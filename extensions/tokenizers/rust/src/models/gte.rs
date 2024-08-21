use crate::layers::{LayerNorm, Linear};
use crate::models::Model;
use candle::{DType, Device, Result, Tensor};
use candle_nn::{embedding, ops, rotary_emb, Activation, Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    Alibi,
    Rope,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GTEConfig {
    pub architectures: Vec<String>,
    model_type: Option<String>,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    pub hidden_act: Activation,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub rope_theta: f64,
    pub use_flash_attn: Option<bool>,
}

impl Default for GTEConfig {
    fn default() -> Self {
        Self {
            architectures: Vec::new(),
            model_type: Some("GTE".to_string()),
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Rope,
            rope_theta: 160_000.,
            use_flash_attn: Some(false),
        }
    }
}

struct GTEEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl GTEEmbeddings {
    fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = LayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings, None)?;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &GTEConfig, dtype: DType, dev: &Device) -> Result<Self> {
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
        let q_embed = rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    intermediate_size: usize,
    up_gate_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let hidden_sz = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let up_gate_proj = Linear::load(
            vb.pp("up_gate_proj"),
            hidden_sz,
            intermediate_size * 2,
            None,
        )?;
        let down_proj = Linear::load(vb.pp("down_proj"), intermediate_size, hidden_sz, None)?;
        Ok(Self {
            intermediate_size,
            up_gate_proj,
            down_proj,
            act_fn: config.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_gate_states = self.up_gate_proj.forward(xs)?;
        let up_states = up_gate_states.narrow(2, 0, self.intermediate_size)?;
        let gate_states =
            up_gate_states.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let gate_states = gate_states.apply(&self.act_fn)?;
        self.down_proj.forward(&(gate_states * up_states)?)
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

struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    hidden_size: usize,
    num_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
}

impl Attention {
    fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let attention_head_size = hidden_size / config.num_attention_heads;
        let qkv_proj = Linear::load(vb.pp("qkv_proj"), hidden_size, hidden_size * 3, None)?;
        let o_proj = Linear::load(vb.pp("o_proj"), hidden_size, hidden_size, None)?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(config, vb.dtype(), vb.device())?);
        Ok(Self {
            qkv_proj,
            o_proj,
            hidden_size,
            num_heads,
            attention_head_size,
            rotary_emb,
            use_flash_attn: config.use_flash_attn.unwrap_or(false),
        })
    }
}

impl Attention {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;

        let qkv = self.qkv_proj.forward(hidden_states)?;

        let qkv = qkv
            .reshape((b_sz, q_len, self.num_heads * 3, self.attention_head_size))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = qkv.narrow(1, 0, self.num_heads)?;
        let k = qkv.narrow(1, self.num_heads, self.num_heads)?;
        let v = qkv.narrow(1, self.num_heads * 2, self.num_heads)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k)?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.attention_head_size as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.attention_head_size as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = attn_weights.broadcast_add(attention_mask)?;
            let attn_weights = ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        Ok(attn_output)
    }
}

struct GTELayer {
    attention: Attention,
    mlp: MLP,
    attention_layer_norm: LayerNorm,
    mlp_layer_norm: LayerNorm,
    span: tracing::Span,
}

impl GTELayer {
    fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let attention = Attention::load(vb.pp("attention"), config)?;
        let mlp = MLP::load(vb.pp("mlp"), config)?;
        let attention_layernorm = LayerNorm::load(
            vb.pp("attn_ln"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;
        let mlp_layernorm = LayerNorm::load(
            vb.pp("mlp_ln"),
            config.hidden_size,
            config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            attention,
            mlp,
            attention_layer_norm: attention_layernorm,
            mlp_layer_norm: mlp_layernorm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }
}

impl GTELayer {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let normed_attention_output = self
            .attention_layer_norm
            .forward(&attention_output, Some(hidden_states))?;
        let mlp_output = self.mlp.forward(&normed_attention_output)?;
        let normed_mlp_output = self
            .mlp_layer_norm
            .forward(&mlp_output, Some(&normed_attention_output))?;
        Ok(normed_mlp_output)
    }
}

struct GTEEncoder {
    layers: Vec<GTELayer>,
    span: tracing::Span,
}

impl GTEEncoder {
    fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| GTELayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(GTEEncoder { layers, span })
    }
}

impl GTEEncoder {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?
        }
        Ok(hidden_states)
    }
}

pub struct GTEModel {
    embeddings: GTEEmbeddings,
    encoder: GTEEncoder,
    #[allow(unused)]
    pub device: Device,
    span: tracing::Span,
}

impl GTEModel {
    pub fn load(vb: VarBuilder, config: &GTEConfig) -> Result<Self> {
        let (embeddings, encoder) = match (
            GTEEmbeddings::load(vb.pp("embeddings"), config),
            GTEEncoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let (Ok(embeddings), Ok(encoder)) = (
                    GTEEmbeddings::load(vb.pp("model.embeddings".to_string()), config),
                    GTEEncoder::load(vb.pp("model.encoder".to_string()), config),
                ) {
                    (embeddings, encoder)
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }
}

impl Model for GTEModel {
    fn get_input_names(&self) -> Vec<String> {
        return vec![
            "input_ids".to_string(),
            "attention_mask".to_string(),
            "token_type_ids".to_string(),
        ];
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids.unwrap())?;
        let sequence_output = self
            .encoder
            .forward(&embedding_output, attention_mask.as_ref())?;
        Ok(sequence_output)
    }
}
