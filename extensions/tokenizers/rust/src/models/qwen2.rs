use crate::layers::{Linear, RmsNorm};
use crate::models::Model;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen2Config {
    pub architectures: Vec<String>,
    model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
    pub use_flash_attn: Option<bool>,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn load(config: &Qwen2Config, dtype: DType, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn load(vb: VarBuilder, config: &Qwen2Config) -> Result<Self> {
        let hidden_sz = config.hidden_size;
        let intermediate_sz = config.intermediate_size;
        let gate_proj = Linear::load(vb.pp("gate_proj"), hidden_sz, intermediate_sz, None)?;
        let up_proj = Linear::load(vb.pp("up_proj"), hidden_sz, intermediate_sz, None)?;
        let down_proj = Linear::load(vb.pp("down_proj"), intermediate_sz, hidden_sz, None)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: config.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(
        vb: VarBuilder,
        config: &Qwen2Config,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let hidden_sz = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = Linear::load(vb.pp("q_proj"), hidden_sz, num_heads * head_dim, None)?;
        let k_proj = Linear::load(vb.pp("k_proj"), hidden_sz, num_kv_heads * head_dim, None)?;
        let v_proj = Linear::load(vb.pp("v_proj"), hidden_sz, num_kv_heads * head_dim, None)?;
        let o_proj = Linear::load(vb.pp("o_proj"), num_heads * head_dim, hidden_sz, None)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states)?;

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = attn_weights.broadcast_add(attention_mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        Ok(attn_output)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(
        vb: VarBuilder,
        config: &Qwen2Config,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config, rotary_emb)?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug)]
pub struct Qwen2Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    #[allow(unused)]
    device: Device,
    dtype: DType,
}

impl Qwen2Model {
    pub fn load(vb: VarBuilder, config: &Qwen2Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;
        let rotary_emb = Arc::new(RotaryEmbedding::load(config, vb.dtype(), vb_m.device())?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = DecoderLayer::load(vb_l.pp(layer_idx), config, rotary_emb.clone())?;
            layers.push(layer)
        }
        let norm = RmsNorm::load(vb_m.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(DType::F32)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(DType::F32)?;
        mask.where_cond(&on_true, &on_false)?.to_dtype(self.dtype)
    }
}

impl Model for Qwen2Model {
    fn get_input_names(&self) -> Vec<String> {
        return vec!["input_ids".to_string(), "attention_mask".to_string()];
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_mask = self.prepare_attention_mask(attention_mask)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask.as_ref())?
        }
        xs.apply(&self.norm)
    }
}
