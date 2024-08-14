use crate::layers::{Linear, RmsNorm};
use crate::models::Model;
use crate::utils::repeat_kv;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, ops, rotary_emb, Activation, Embedding, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MistralConfig {
    pub architectures: Vec<String>,
    model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub use_flash_attn: Option<bool>,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            architectures: Vec::new(),
            model_type: Some("mistral".to_string()),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn: Some(false),
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &MistralConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let rope_theta = config.rope_theta as f32;
        let dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / dim as f32))
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

    fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = rotary_emb::rope(k, &cos, &sin)?;
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
    fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
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

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
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
    use_flash_attn: bool,
}

impl Attention {
    fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
        let hidden_sz = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = Linear::load(vb.pp("q_proj"), hidden_sz, num_heads * head_dim, None)?;
        let k_proj = Linear::load(vb.pp("k_proj"), hidden_sz, num_kv_heads * head_dim, None)?;
        let v_proj = Linear::load(vb.pp("v_proj"), hidden_sz, num_kv_heads * head_dim, None)?;
        let o_proj = Linear::load(vb.pp("o_proj"), num_heads * head_dim, hidden_sz, None)?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(config, vb.dtype(), vb.device())?);
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
            use_flash_attn: config.use_flash_attn.unwrap_or(false),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;

        let query_states = self.q_proj.forward(hidden_states)?;
        let key_states = self.k_proj.forward(hidden_states)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb(&query_states, &key_states)?;

        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = ops::softmax_last_dim(&attn_weights)?;
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
    fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps as f32,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps as f32,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let (xs, residual) = self.input_layernorm.forward(xs, Some(&residual))?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let (xs, residual) = self
            .post_attention_layernorm
            .forward(&xs, Some(&residual))?;
        let xs = self.mlp.forward(&xs);
        residual + xs
    }
}

#[derive(Debug)]
pub struct MistralModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: Option<usize>,
    #[allow(unused)]
    pub device: Device,
    dtype: DType,
}

impl MistralModel {
    pub fn load(vb: VarBuilder, config: &MistralConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?; // TODO
        let layers = (0..config.num_hidden_layers)
            .map(|index| DecoderLayer::load(vb.pp(&format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let norm = RmsNorm::load(
            vb.pp("norm"),
            config.hidden_size,
            config.rms_norm_eps as f32,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: config.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_decoder_attention_mask(&self, tgt_len: usize) -> Result<Tensor> {
        let sliding_window = self.sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        mask.expand((1, 1, tgt_len, tgt_len))?.to_dtype(self.dtype)
    }
}

impl Model for MistralModel {
    fn get_input_names(&self) -> Vec<String> {
        return vec!["input_ids".to_string(), "attention_mask".to_string()];
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        _attention_mask: &Tensor,
        _token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(seq_len)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask.as_ref())?
        }
        let xs = xs.narrow(1, seq_len - 1, 1)?;
        let (xs, _residual) = self.norm.forward(&xs, None)?;
        Ok(xs)
    }
}
