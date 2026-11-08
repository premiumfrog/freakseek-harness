//! Decoder-only transformer for the llama family of GGUF architectures.
//!
//! Covers `llama` (Llama 2/3, Mistral, DeepSeek-R1 Llama distills), `qwen2`
//! (Qwen 2/2.5, DeepSeek-R1 Qwen distills), and `qwen3` (per-head Q/K norm).
//! Weights stay quantized; only the requested embedding rows are dequantized.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

use crate::device::Placement;

/// RoPE pairing convention used by the architecture's GGUF conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RopeStyle {
    /// Adjacent pairs (llama.cpp `LLAMA_ROPE_TYPE_NORM`).
    Interleaved,
    /// Half-split pairs (llama.cpp `LLAMA_ROPE_TYPE_NEOX`).
    Neox,
}

/// Static hyperparameters read from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Hparams {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    rope: RopeStyle,
}

struct Linear {
    w: QMatMul,
    b: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.w.forward(x)?;
        Ok(match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Layer {
    /// Device holding this layer's weights and KV cache.
    dev: Device,
    attn_norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    ffn_norm: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

/// Growable per-layer key/value cache that supports truncation for prefix reuse.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl KvCache {
    /// Move the cached keys and values, keeping only the used positions.
    /// @param dev - destination device.
    fn moved(self, dev: &Device) -> Result<Self> {
        let len = self.len;
        let mv = |t: Option<Tensor>| -> Result<Option<Tensor>> {
            match t {
                Some(_) if len == 0 => Ok(None),
                // Contiguous first, so only the used positions cross the bus.
                Some(t) if !t.device().same_device(dev) => Ok(Some(t.narrow(2, 0, len)?.contiguous()?.to_device(dev)?)),
                other => Ok(other),
            }
        };
        Ok(Self { k: mv(self.k)?, v: mv(self.v)?, len })
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let add = k.dim(2)?;
        let need = self.len + add;
        let cap = self.k.as_ref().map_or(0, |t| t.dim(2).unwrap_or(0));
        if need > cap {
            // Double capacity so per-token appends stay amortized O(1).
            let new_cap = need.max(cap * 2).max(256);
            let (b, h, _, d) = k.dims4()?;
            let nk = Tensor::zeros((b, h, new_cap, d), k.dtype(), k.device())?;
            let nv = Tensor::zeros((b, h, new_cap, d), v.dtype(), v.device())?;
            if let (Some(ok), Some(ov)) = (&self.k, &self.v) {
                if self.len > 0 {
                    nk.slice_set(&ok.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
                    nv.slice_set(&ov.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
                }
            }
            self.k = Some(nk);
            self.v = Some(nv);
        }
        let (Some(ck), Some(cv)) = (self.k.as_ref(), self.v.as_ref()) else { bail!("KV cache allocation failed") };
        ck.slice_set(&k.contiguous()?, 2, self.len)?;
        cv.slice_set(&v.contiguous()?, 2, self.len)?;
        self.len = need;
        Ok((ck.narrow(2, 0, self.len)?, cv.narrow(2, 0, self.len)?))
    }
}

/// Loaded model weights plus the KV cache of the current sequence.
pub struct Model {
    pub hp: Hparams,
    embd: Arc<QTensor>,
    layers: Vec<Layer>,
    out_norm: Tensor,
    output: QMatMul,
    cache: Vec<KvCache>,
    inv_freq: Vec<f32>,
    /// Device of the output head and final norm.
    out_dev: Device,
    placement: Placement,
}

fn md_u32(c: &Content, key: &str) -> Option<usize> {
    c.metadata.get(key).and_then(|v| v.to_u32().ok().map(|x| x as usize).or_else(|| v.to_u64().ok().map(|x| x as usize)))
}

fn md_f32(c: &Content, key: &str) -> Option<f32> {
    c.metadata.get(key).and_then(|v| v.to_f32().ok().or_else(|| v.to_f64().ok().map(|x| x as f32)))
}

/// Read the architecture hyperparameters.
/// @param c - parsed GGUF header.
pub fn hparams(c: &Content) -> Result<Hparams> {
    let arch = c.metadata.get("general.architecture").context("GGUF has no general.architecture")?.to_string()?.clone();
    let rope = match arch.as_str() {
        "llama" | "mistral" | "granite" => RopeStyle::Interleaved,
        "qwen2" | "qwen3" => RopeStyle::Neox,
        other => bail!("unsupported architecture `{other}`; supported: llama, mistral, qwen2, qwen3"),
    };
    let k = |s: &str| format!("{arch}.{s}");
    let n_embd = md_u32(c, &k("embedding_length")).context("missing embedding_length")?;
    let n_head = md_u32(c, &k("attention.head_count")).context("missing attention.head_count")?;
    let n_head_kv = md_u32(c, &k("attention.head_count_kv")).unwrap_or(n_head);
    let head_dim = md_u32(c, &k("attention.key_length")).unwrap_or(n_embd / n_head);
    Ok(Hparams {
        n_layer: md_u32(c, &k("block_count")).context("missing block_count")?,
        n_embd,
        n_head,
        n_head_kv,
        head_dim,
        n_ctx_train: md_u32(c, &k("context_length")).unwrap_or(4096),
        rms_eps: md_f32(c, &k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5) as f64,
        rope_base: md_f32(c, &k("rope.freq_base")).unwrap_or(10000.0),
        rope,
        arch,
    })
}

struct Loader<'a> {
    c: &'a Content,
    r: BufReader<File>,
    /// Device the next tensors load onto.
    dev: Device,
}

impl Loader<'_> {
    fn has(&self, name: &str) -> bool {
        self.c.tensor_infos.contains_key(name)
    }
    fn q(&mut self, name: &str) -> Result<QTensor> {
        self.c.tensor(&mut self.r, name, &self.dev).with_context(|| format!("missing tensor `{name}`"))
    }
    fn f(&mut self, name: &str) -> Result<Tensor> {
        Ok(self.q(name)?.dequantize(&self.dev)?)
    }
    fn mm(&mut self, name: &str) -> Result<QMatMul> {
        Ok(QMatMul::from_qtensor(self.q(name)?)?)
    }
    fn linear(&mut self, prefix: &str) -> Result<Linear> {
        let w = self.mm(&format!("{prefix}.weight"))?;
        let bias = format!("{prefix}.bias");
        let b = if self.has(&bias) { Some(self.f(&bias)?) } else { None };
        Ok(Linear { w, b })
    }
}

impl Model {
    /// Load every weight of a GGUF model into memory.
    /// @param path - GGUF file.
    /// @param content - header parsed from the same file.
    /// @param placement - devices for the layers and output head.
    pub fn load(path: &Path, content: &Content, placement: &Placement) -> Result<Self> {
        let hp = hparams(content)?;
        let cpu = Device::Cpu;
        let mut l = Loader { c: content, r: BufReader::new(File::open(path)?), dev: cpu.clone() };
        // Embedding rows are looked up on the host, so the table stays in system memory.
        let embd = Arc::new(l.q("token_embd.weight")?);
        // The output head follows the last layer, which is the first one offloaded.
        let out_dev = if placement.offloads(hp.n_layer.saturating_sub(1), hp.n_layer) { placement.device.clone() } else { cpu.clone() };
        l.dev = out_dev.clone();
        let out_norm = l.f("output_norm.weight")?;
        let output = if l.has("output.weight") {
            l.mm("output.weight")?
        } else if out_dev.is_cpu() {
            QMatMul::from_arc(embd.clone())?
        } else {
            // Tied output head on the accelerator: a device copy beside the host table.
            l.mm("token_embd.weight")?
        };
        let mut layers = Vec::with_capacity(hp.n_layer);
        for i in 0..hp.n_layer {
            l.dev = if placement.offloads(i, hp.n_layer) { placement.device.clone() } else { cpu.clone() };
            let p = |s: &str| format!("blk.{i}.{s}");
            let q_norm = if l.has(&p("attn_q_norm.weight")) { Some(l.f(&p("attn_q_norm.weight"))?) } else { None };
            let k_norm = if l.has(&p("attn_k_norm.weight")) { Some(l.f(&p("attn_k_norm.weight"))?) } else { None };
            layers.push(Layer {
                dev: l.dev.clone(),
                attn_norm: l.f(&p("attn_norm.weight"))?,
                q: l.linear(&p("attn_q"))?,
                k: l.linear(&p("attn_k"))?,
                v: l.linear(&p("attn_v"))?,
                o: l.linear(&p("attn_output"))?,
                q_norm,
                k_norm,
                ffn_norm: l.f(&p("ffn_norm.weight"))?,
                gate: l.mm(&p("ffn_gate.weight"))?,
                up: l.mm(&p("ffn_up.weight"))?,
                down: l.mm(&p("ffn_down.weight"))?,
            });
        }
        let half = hp.head_dim / 2;
        let mut inv_freq: Vec<f32> =
            (0..half).map(|i| 1.0 / hp.rope_base.powf((2 * i) as f32 / hp.head_dim as f32)).collect();
        // Llama 3.1+ ships long-context frequency factors as a tensor.
        if l.has("rope_freqs.weight") {
            let factors = l.f("rope_freqs.weight")?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (f, s) in inv_freq.iter_mut().zip(factors) {
                *f /= s;
            }
        }
        let cache = (0..hp.n_layer).map(|_| KvCache { k: None, v: None, len: 0 }).collect();
        Ok(Self { hp, embd, layers, out_norm, output, cache, inv_freq, out_dev, placement: placement.clone() })
    }

    /// An empty cache set for this model, for a new sequence slot.
    pub fn empty_cache(&self) -> Vec<KvCache> {
        (0..self.hp.n_layer).map(|_| KvCache { k: None, v: None, len: 0 }).collect()
    }

    /// Swap the active KV cache with `other`, returning the previously active one.
    /// Offloaded layers' caches move to the accelerator on activation and back to
    /// system memory when parked, so parked slots do not occupy VRAM.
    /// @param other - cache set produced by [`Model::empty_cache`] or an earlier swap.
    pub fn swap_cache(&mut self, other: Vec<KvCache>) -> Result<Vec<KvCache>> {
        let incoming = other.into_iter().zip(&self.layers).map(|(c, l)| c.moved(&l.dev)).collect::<Result<Vec<_>>>()?;
        let previous = std::mem::replace(&mut self.cache, incoming);
        previous.into_iter().map(|c| c.moved(&Device::Cpu)).collect()
    }

    /// Discard the active KV cache.
    pub fn clear_cache(&mut self) {
        self.cache = self.empty_cache();
    }

    /// Wire label of the accelerator, or `cpu`.
    pub fn device_label(&self) -> String {
        self.placement.label()
    }

    /// Layers running on the accelerator.
    pub fn offloaded_layers(&self) -> usize {
        self.layers.iter().filter(|l| !l.dev.is_cpu()).count()
    }

    /// Tokens currently held in the KV cache.
    pub fn cached_len(&self) -> usize {
        self.cache.first().map_or(0, |c| c.len)
    }

    /// Drop cached positions at and after `len`, keeping the shared prefix.
    /// @param len - number of leading positions to keep.
    pub fn truncate(&mut self, len: usize) {
        for c in &mut self.cache {
            c.len = c.len.min(len);
        }
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let (vocab, dim) = self.embd.shape().dims2()?;
        let dtype = self.embd.dtype();
        let row_bytes = dim / dtype.block_size() * dtype.type_size();
        let data = self.embd.data()?;
        let mut rows = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let t = t as usize;
            if t >= vocab {
                bail!("token id {t} outside vocabulary of {vocab}");
            }
            let bytes = &data[t * row_bytes..(t + 1) * row_bytes];
            rows.push(qtensor_from_ggml(dtype, bytes, vec![dim], &Device::Cpu)?.dequantize(&Device::Cpu)?);
        }
        Ok(Tensor::stack(&rows, 0)?.unsqueeze(0)?)
    }

    fn rope_tables(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut freqs = Vec::with_capacity(len * half);
        for p in start..start + len {
            for f in &self.inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let t = Tensor::from_vec(freqs, (len, half), &Device::Cpu)?;
        Ok((t.cos()?, t.sin()?))
    }

    /// Run the next tokens through the model, appending them to the KV cache.
    /// @param tokens - tokens following the cached positions.
    /// @returns logits for the last token as a vector of vocabulary size.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let start = self.cached_len();
        let t = tokens.len();
        let hp = self.hp.clone();
        let (cos, sin) = self.rope_tables(start, t)?;
        let mask = if t > 1 {
            // Causal mask over [cached | new] keys for the new queries.
            let total = start + t;
            let m: Vec<f32> = (0..t)
                .flat_map(|i| (0..total).map(move |j| if j > start + i { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            Some(Tensor::from_vec(m, (1, 1, t, total), &Device::Cpu)?)
        } else {
            None
        };
        let mut x = self.embed(tokens)?;
        let scale = 1.0 / (hp.head_dim as f64).sqrt();
        let groups = hp.n_head / hp.n_head_kv;
        // Host-built tables, copied once to the accelerator when the first offloaded layer needs them.
        let (mut cos, mut sin, mut mask) = (cos, sin, mask);
        for (layer, cache) in self.layers.iter().zip(self.cache.iter_mut()) {
            if !x.device().same_device(&layer.dev) {
                x = x.to_device(&layer.dev)?;
                cos = cos.to_device(&layer.dev)?;
                sin = sin.to_device(&layer.dev)?;
                mask = mask.map(|m| m.to_device(&layer.dev)).transpose()?;
            }
            let h = candle_nn::ops::rms_norm(&x, &layer.attn_norm, hp.rms_eps as f32)?;
            let q = layer.q.forward(&h)?.reshape((1, t, hp.n_head, hp.head_dim))?;
            let k = layer.k.forward(&h)?.reshape((1, t, hp.n_head_kv, hp.head_dim))?;
            let v = layer.v.forward(&h)?.reshape((1, t, hp.n_head_kv, hp.head_dim))?.transpose(1, 2)?;
            let q = match &layer.q_norm {
                Some(n) => candle_nn::ops::rms_norm(&q.contiguous()?, n, hp.rms_eps as f32)?,
                None => q,
            };
            let k = match &layer.k_norm {
                Some(n) => candle_nn::ops::rms_norm(&k.contiguous()?, n, hp.rms_eps as f32)?,
                None => k,
            };
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let (q, k) = match hp.rope {
                RopeStyle::Interleaved => (candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?, candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?),
                RopeStyle::Neox => (candle_nn::rotary_emb::rope(&q, &cos, &sin)?, candle_nn::rotary_emb::rope(&k, &cos, &sin)?),
            };
            let (k, v) = cache.append(&k, &v)?;
            let total = k.dim(2)?;
            // Grouped-query attention: fold query heads per KV head instead of copying K/V.
            let q = q.reshape((1, hp.n_head_kv, groups * t, hp.head_dim))?;
            let att = (q.matmul(&k.contiguous()?.t()?)? * scale)?;
            let att = match &mask {
                Some(m) => {
                    let att = att.reshape((1, hp.n_head_kv, groups, t, total))?;
                    att.broadcast_add(&m.unsqueeze(1)?)?.reshape((1, hp.n_head_kv, groups * t, total))?
                }
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let y = att.matmul(&v.contiguous()?)?;
            let y = y.reshape((1, hp.n_head, t, hp.head_dim))?.transpose(1, 2)?.reshape((1, t, hp.n_head * hp.head_dim))?;
            x = (x + layer.o.forward(&y)?)?;
            let h = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, hp.rms_eps as f32)?;
            let g = candle_nn::ops::silu(&layer.gate.forward(&h)?)?;
            let u = layer.up.forward(&h)?;
            x = (x + layer.down.forward(&(g * u)?)?)?;
        }
        let last = x.narrow(1, t - 1, 1)?.to_device(&self.out_dev)?;
        let last = candle_nn::ops::rms_norm(&last, &self.out_norm, hp.rms_eps as f32)?;
        let logits = self.output.forward(&last)?.squeeze(0)?.squeeze(0)?;
        Ok(logits.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }

    /// Vocabulary size of the output head.
    pub fn vocab(&self) -> usize {
        self.embd.shape().dims()[0]
    }
}
