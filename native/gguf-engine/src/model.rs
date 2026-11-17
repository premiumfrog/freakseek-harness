//! Decoder-only transformer for the llama and Qwen families of GGUF architectures.
//!
//! Covers `llama` (Llama 2/3, Mistral, DeepSeek-R1 Llama distills), `qwen2`
//! (Qwen 2/2.5, DeepSeek-R1 Qwen distills), `qwen3` (per-head Q/K norm), `qwen3moe`
//! (routed experts), and `qwen35` / `qwen35moe` (gated attention on every fourth
//! layer, gated delta-net layers elsewhere). Weights stay quantized; only the
//! requested embedding rows are dequantized.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

use crate::device::{ensure_kernels, Placement};
use crate::engine::EngineError;
use crate::moe::{self, Expert, Moe, SharedExpert};
use crate::recurrent::{DeltaDims, DeltaNet, RecState};

/// Recurrent-state checkpoints kept per sequence.
const MAX_CHECKPOINTS: usize = 8;
/// Elements above which a floating-point weight matrix is stored as Q8_0.
const LARGE_FLOAT_MATRIX: usize = 1 << 24;
/// Decoded tokens between automatic checkpoints.
const CHECKPOINT_INTERVAL: usize = 128;

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
    /// Decoder layers the forward pass runs; multi-token-prediction layers are excluded.
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ctx_train: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    rope: RopeStyle,
    /// Leading dimensions of a head that RoPE rotates.
    n_rot: usize,
    /// Interleaved multi-axis RoPE sections; text rotates every pair outside the fourth section.
    rope_sections: Option<[usize; 4]>,
    /// The query projection also emits a per-dimension output gate.
    gated_attn: bool,
    /// Per layer, whether it is a gated delta-net layer.
    recurrent: Vec<bool>,
    delta: Option<DeltaDims>,
    pub n_expert: usize,
    pub n_expert_used: usize,
    expert_scale: f32,
}

impl Hparams {
    /// Layers that keep a KV cache.
    pub fn attention_layers(&self) -> usize {
        self.recurrent.iter().filter(|r| !**r).count()
    }

    /// KV cache bytes per context position with half-precision keys and values.
    pub fn kv_bytes_per_token(&self) -> u64 {
        (self.attention_layers() * 2 * self.n_head_kv * self.head_dim * 2) as u64
    }

    /// Recurrent state bytes per sequence: convolution windows and state matrices.
    pub fn state_bytes(&self) -> u64 {
        let Some(d) = self.delta else { return 0 };
        let per_layer = (d.kernel - 1) * (2 * d.head_dim * d.k_heads + d.head_dim * d.v_heads) + d.v_heads * d.head_dim * d.head_dim;
        (self.recurrent.iter().filter(|r| **r).count() * per_layer * 4) as u64
    }
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

struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
}

enum Mixer {
    Attn(Attn),
    Delta(Box<DeltaNet>),
}

enum Ffn {
    Dense { gate: QMatMul, up: QMatMul, down: QMatMul },
    Moe(Moe),
}

struct Layer {
    /// Device holding this layer's weights and KV cache.
    dev: Device,
    attn_norm: Tensor,
    mixer: Mixer,
    ffn_norm: Tensor,
    ffn: Ffn,
}

/// Growable per-layer key/value cache that supports truncation for prefix reuse.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl KvCache {
    fn empty() -> Self {
        Self { k: None, v: None, len: 0 }
    }

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

/// Memory of one layer for one sequence.
enum LayerCache {
    Kv(KvCache),
    Rec(RecState),
}

/// Memory of one sequence: a KV cache per attention layer, a recurrent state per
/// delta-net layer, and checkpoints of the recurrent states at earlier positions.
pub struct CacheSet {
    layers: Vec<LayerCache>,
    /// Positions evaluated into this set.
    len: usize,
    /// `(position, recurrent states in layer order)`, ascending by position.
    checkpoints: Vec<(usize, Vec<RecState>)>,
}

impl CacheSet {
    fn recurrent(&self) -> bool {
        self.layers.iter().any(|l| matches!(l, LayerCache::Rec(_)))
    }

    /// Record the recurrent states at the current position, thinning older checkpoints.
    fn checkpoint(&mut self) {
        if !self.recurrent() || self.checkpoints.last().is_some_and(|c| c.0 == self.len) {
            return;
        }
        let states = self.layers.iter().filter_map(|l| if let LayerCache::Rec(s) = l { Some(s.clone()) } else { None }).collect();
        self.checkpoints.push((self.len, states));
        if self.checkpoints.len() > MAX_CHECKPOINTS {
            // Drop the interior checkpoint whose neighbours are closest, keeping the first and the newest.
            let n = self.checkpoints.len();
            let drop = (1..n - 1).min_by_key(|i| self.checkpoints[i + 1].0 - self.checkpoints[i - 1].0).unwrap_or(1);
            self.checkpoints.remove(drop);
        }
    }

    /// Cut the sequence back to at most `len` positions.
    /// @returns the positions kept: `len` for attention-only models, otherwise the newest checkpoint at or before it.
    fn truncate(&mut self, len: usize) -> usize {
        if len >= self.len {
            return self.len;
        }
        let keep = if self.recurrent() {
            self.checkpoints.retain(|c| c.0 <= len);
            let (pos, states) = match self.checkpoints.last() {
                Some((pos, states)) => (*pos, Some(states.clone())),
                None => (0, None),
            };
            let mut states = states.map(Vec::into_iter);
            for l in &mut self.layers {
                if let LayerCache::Rec(s) = l {
                    match states.as_mut().and_then(Iterator::next) {
                        Some(saved) => *s = saved,
                        None => s.reset(),
                    }
                }
            }
            pos
        } else {
            len
        };
        for l in &mut self.layers {
            if let LayerCache::Kv(c) = l {
                c.len = c.len.min(keep);
            }
        }
        self.len = keep;
        keep
    }
}

/// Recurrent checkpoints written to a saved prompt cache: the newest, which bound its size.
const SAVED_CHECKPOINTS: usize = 4;

impl CacheSet {
    /// A copy that shares no storage with this set, so appending to one never changes the other.
    pub fn deep_copy(&self) -> Result<Self> {
        let layers = self
            .layers
            .iter()
            .map(|l| {
                Ok(match l {
                    LayerCache::Kv(kv) => LayerCache::Kv(KvCache {
                        k: kv.k.as_ref().map(Tensor::copy).transpose()?,
                        v: kv.v.as_ref().map(Tensor::copy).transpose()?,
                        len: kv.len,
                    }),
                    LayerCache::Rec(s) => LayerCache::Rec(s.clone()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, len: self.len, checkpoints: self.checkpoints.clone() })
    }

    /// Positions evaluated into this set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no position has been evaluated.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Write the set to a safetensors file: KV caches trimmed to their positions, recurrent states, and the
    /// newest [`SAVED_CHECKPOINTS`] checkpoints.
    /// @param path - destination file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let cpu = Device::Cpu;
        let mut out: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let vec = |v: &[f32]| Tensor::from_slice(v, v.len(), &cpu);
        out.insert("len".into(), Tensor::new(&[self.len as u32], &cpu)?);
        for (i, l) in self.layers.iter().enumerate() {
            match l {
                LayerCache::Kv(kv) => {
                    if let (Some(k), Some(v)) = (&kv.k, &kv.v) {
                        if kv.len > 0 {
                            out.insert(format!("kv.{i}.k"), k.narrow(2, 0, kv.len)?.to_device(&cpu)?.contiguous()?);
                            out.insert(format!("kv.{i}.v"), v.narrow(2, 0, kv.len)?.to_device(&cpu)?.contiguous()?);
                        }
                    }
                }
                LayerCache::Rec(s) => {
                    let (conv, state) = s.parts();
                    out.insert(format!("rec.{i}.conv"), vec(conv)?);
                    out.insert(format!("rec.{i}.s"), vec(state)?);
                }
            }
        }
        let kept = &self.checkpoints[self.checkpoints.len().saturating_sub(SAVED_CHECKPOINTS)..];
        out.insert("checkpoints".into(), Tensor::new(kept.iter().map(|c| c.0 as u32).collect::<Vec<_>>(), &cpu)?);
        for (n, (_, states)) in kept.iter().enumerate() {
            for (j, s) in states.iter().enumerate() {
                let (conv, state) = s.parts();
                out.insert(format!("ckpt.{n}.{j}.conv"), vec(conv)?);
                out.insert(format!("ckpt.{n}.{j}.s"), vec(state)?);
            }
        }
        candle_core::safetensors::save(&out, path)?;
        Ok(())
    }
}

/// Loaded model weights plus the cache of the current sequence.
pub struct Model {
    pub hp: Hparams,
    embd: Arc<QTensor>,
    layers: Vec<Layer>,
    out_norm: Tensor,
    output: QMatMul,
    cache: CacheSet,
    inv_freq: Vec<f32>,
    /// Device of the output head and final norm.
    out_dev: Device,
    placement: Placement,
    /// Where expert use counts are saved for the next load; `None` without routed experts.
    profile: Option<PathBuf>,
    /// When set, [`Model::forward`] appends the last position's hidden-state norm after each layer.
    pub trace: Option<Vec<f32>>,
    /// Wall-clock time per forward phase since the last [`Model::take_phases`].
    phases: PhaseTimes,
}

/// Wall-clock time spent in each forward phase. On a GPU, kernels queue asynchronously, so a phase also absorbs the
/// wait for earlier queued work at its first host read; the figures locate stalls rather than isolate kernel cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimes {
    pub attention: Duration,
    pub recurrent: Duration,
    pub ffn: Duration,
    pub head: Duration,
}

fn md_u32(c: &Content, key: &str) -> Option<usize> {
    c.metadata.get(key).and_then(|v| v.to_u32().ok().map(|x| x as usize).or_else(|| v.to_u64().ok().map(|x| x as usize)))
}

fn md_f32(c: &Content, key: &str) -> Option<f32> {
    c.metadata.get(key).and_then(|v| v.to_f32().ok().or_else(|| v.to_f64().ok().map(|x| x as f32)))
}

fn md_usize_array(c: &Content, key: &str) -> Option<Vec<usize>> {
    let values = c.metadata.get(key)?.to_vec().ok()?;
    values.iter().map(|v| v.to_u32().ok().map(|x| x as usize).or_else(|| v.to_i32().ok().map(|x| x.max(0) as usize))).collect()
}

/// Read the architecture hyperparameters.
/// @param c - parsed GGUF header.
pub fn hparams(c: &Content) -> Result<Hparams> {
    let arch = c.metadata.get("general.architecture").context("GGUF has no general.architecture")?.to_string()?.clone();
    let rope = match arch.as_str() {
        "llama" | "mistral" | "granite" => RopeStyle::Interleaved,
        "qwen2" | "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe" => RopeStyle::Neox,
        other => bail!("unsupported architecture `{other}`; supported: llama, mistral, qwen2, qwen3, qwen3moe, qwen35, qwen35moe"),
    };
    let hybrid = matches!(arch.as_str(), "qwen35" | "qwen35moe");
    let k = |s: &str| format!("{arch}.{s}");
    let need = |s: &str| md_u32(c, &k(s)).with_context(|| format!("missing {s}"));
    let n_embd = need("embedding_length")?;
    let n_head = need("attention.head_count")?;
    let n_head_kv = md_u32(c, &k("attention.head_count_kv")).unwrap_or(n_head);
    let head_dim = md_u32(c, &k("attention.key_length")).unwrap_or(n_embd / n_head);
    // Multi-token-prediction blocks are stored after the decoder layers and are not part of the forward pass.
    let n_layer = need("block_count")?.saturating_sub(md_u32(c, &k("nextn_predict_layers")).unwrap_or(0));
    let recurrent = if hybrid {
        let listed = c.metadata.get(&k("attention.recurrent_layers")).and_then(|v| v.to_vec().ok());
        match listed {
            Some(flags) => (0..n_layer).map(|i| flags.get(i).and_then(|f| f.to_bool().ok()).unwrap_or(false)).collect(),
            None => {
                let interval = md_u32(c, &k("full_attention_interval")).unwrap_or(4).max(1);
                (0..n_layer).map(|i| (i + 1) % interval != 0).collect()
            }
        }
    } else {
        vec![false; n_layer]
    };
    let delta = if hybrid {
        let dims = DeltaDims {
            head_dim: need("ssm.state_size")?,
            k_heads: need("ssm.group_count")?,
            v_heads: need("ssm.time_step_rank")?,
            kernel: need("ssm.conv_kernel")?,
        };
        if dims.head_dim == 0 || dims.k_heads == 0 || dims.v_heads % dims.k_heads != 0 || dims.kernel == 0 {
            bail!("invalid gated delta-net dimensions {dims:?}");
        }
        Some(dims)
    } else {
        None
    };
    let rope_sections = match md_usize_array(c, &k("rope.dimension_sections")) {
        Some(s) if hybrid && s.len() >= 3 => Some([s[0], s[1], s[2], s.get(3).copied().unwrap_or(0)]),
        _ => None,
    };
    let n_rot = md_u32(c, &k("rope.dimension_count")).filter(|_| hybrid).unwrap_or(head_dim).min(head_dim);
    let n_expert = md_u32(c, &k("expert_count")).unwrap_or(0);
    let n_expert_used = md_u32(c, &k("expert_used_count")).unwrap_or(0);
    if arch.ends_with("moe") && (n_expert == 0 || n_expert_used == 0 || n_expert_used > n_expert) {
        bail!("invalid expert counts: {n_expert_used} used of {n_expert}");
    }
    Ok(Hparams {
        n_layer,
        n_embd,
        n_head,
        n_head_kv,
        head_dim,
        n_ctx_train: md_u32(c, &k("context_length")).unwrap_or(4096),
        rms_eps: md_f32(c, &k("attention.layer_norm_rms_epsilon")).unwrap_or(1e-5) as f64,
        rope_base: md_f32(c, &k("rope.freq_base")).unwrap_or(10000.0),
        rope,
        n_rot,
        rope_sections,
        gated_attn: hybrid,
        recurrent,
        delta,
        n_expert,
        n_expert_used,
        expert_scale: md_f32(c, &k("expert_weights_scale")).filter(|s| *s != 0.0).unwrap_or(1.0),
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
    /// A small tensor as host floats.
    fn host(&mut self, name: &str) -> Result<Vec<f32>> {
        let t = self.c.tensor(&mut self.r, name, &Device::Cpu).with_context(|| format!("missing tensor `{name}`"))?;
        Ok(t.dequantize(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
    }
    /// A weight matrix. Large half- and single-precision matrices, such as a BF16 output head, are
    /// stored as Q8_0: candle would otherwise expand them to F32, four times their size in the file.
    fn mm(&mut self, name: &str) -> Result<QMatMul> {
        let t = self.q(name)?;
        let float = matches!(t.dtype(), GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32);
        let dims = t.shape().dims().to_vec();
        if float && dims.len() == 2 && dims[0] * dims[1] >= LARGE_FLOAT_MATRIX && dims[1] % GgmlDType::Q8_0.block_size() == 0 {
            let host = self.c.tensor(&mut self.r, name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
            return Ok(QMatMul::from_qtensor(QTensor::quantize_onto(&host, GgmlDType::Q8_0, &self.dev)?)?);
        }
        Ok(QMatMul::from_qtensor(t)?)
    }
    fn linear(&mut self, prefix: &str) -> Result<Linear> {
        let w = self.mm(&format!("{prefix}.weight"))?;
        let bias = format!("{prefix}.bias");
        let b = if self.has(&bias) { Some(self.f(&bias)?) } else { None };
        Ok(Linear { w, b })
    }
    /// Bytes a tensor occupies in the file; 0 when absent.
    fn bytes(&self, name: &str) -> u64 {
        self.c.tensor_infos.get(name).map_or(0, |i| (i.shape.elem_count() / i.ggml_dtype.block_size() * i.ggml_dtype.type_size()) as u64)
    }
    /// Bytes of one routed expert of layer `i`.
    fn expert_bytes(&self, i: usize, n_expert: usize) -> u64 {
        ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"].iter().map(|s| self.bytes(&format!("blk.{i}.{s}.weight"))).sum::<u64>()
            / n_expert.max(1) as u64
    }
    /// Split a stacked `(n_expert, rows, cols)` tensor into one matrix per expert on that expert's device.
    /// @param name - stacked tensor.
    /// @param devs - device per expert.
    /// @param halves - 2 when each expert stacks a gate matrix above an up matrix, otherwise 1.
    /// @returns per expert, `halves` matrices.
    fn expert_matrices(&mut self, name: &str, devs: &[Device], halves: usize) -> Result<Vec<Vec<QMatMul>>> {
        let stacked = self.c.tensor(&mut self.r, name, &Device::Cpu).with_context(|| format!("missing tensor `{name}`"))?;
        let (n, rows, cols) = stacked.shape().dims3().with_context(|| format!("`{name}` is not a stack of matrices"))?;
        let dtype = stacked.dtype();
        if n != devs.len() || cols % dtype.block_size() != 0 || rows % halves != 0 {
            bail!("`{name}` has shape ({n}, {rows}, {cols}), which does not split into {} experts", devs.len());
        }
        let data = stacked.data()?;
        let part = rows / halves * (cols / dtype.block_size() * dtype.type_size());
        let mut out = Vec::with_capacity(n);
        for (e, dev) in devs.iter().enumerate() {
            let mut matrices = Vec::with_capacity(halves);
            for h in 0..halves {
                let at = (e * halves + h) * part;
                matrices.push(QMatMul::from_qtensor(qtensor_from_ggml(dtype, &data[at..at + part], vec![rows / halves, cols], dev)?)?);
            }
            out.push(matrices);
        }
        Ok(out)
    }
}

/// Expert use counts saved by an earlier run, one line of counts per layer.
fn read_profile(path: &Path, n_layer: usize, n_expert: usize) -> Vec<Vec<u32>> {
    let saved = std::fs::read_to_string(path).unwrap_or_default();
    let mut rows: Vec<Vec<u32>> = saved.lines().map(|l| l.split_whitespace().filter_map(|x| x.parse().ok()).collect()).collect();
    if rows.len() != n_layer || rows.iter().any(|r: &Vec<u32>| r.len() != n_expert) {
        rows = vec![vec![0; n_expert]; n_layer];
    }
    rows
}

impl Model {
    /// Load every weight of a GGUF model into memory.
    /// @param path - GGUF file.
    /// @param content - header parsed from the same file.
    /// @param placement - devices for the layers and output head.
    pub fn load(path: &Path, content: &Content, placement: &Placement) -> Result<Self> {
        Self::load_with_progress(path, content, placement, &mut |_, _| {})
    }

    /// [`Model::load`], reporting each loaded layer.
    /// @param progress - called with `(layers loaded, total layers)` after each layer.
    pub fn load_with_progress(path: &Path, content: &Content, placement: &Placement, progress: &mut dyn FnMut(usize, usize)) -> Result<Self> {
        let hp = hparams(content)?;
        let cpu = Device::Cpu;
        let mut l = Loader { c: content, r: BufReader::new(File::open(path)?), dev: cpu.clone() };
        let is_moe = hp.n_expert > 0;
        if !placement.device.is_cpu() {
            let mut dtypes: Vec<_> = content.tensor_infos.values().filter(|i| i.shape.rank() >= 2).map(|i| i.ggml_dtype).collect();
            dtypes.sort_by_key(|d| format!("{d:?}"));
            dtypes.dedup();
            ensure_kernels(&placement.device, &dtypes)?;
        }
        let profile = is_moe.then(|| PathBuf::from(format!("{}.routing", path.display())));
        // With routed experts and no explicit layer count, every layer's shared weights go to the
        // accelerator and experts fill the remaining memory, most used first.
        let split = is_moe && !placement.device.is_cpu() && placement.gpu_layers.is_none();
        let on_gpu = |i: usize| if split { true } else { placement.offloads(i, hp.n_layer) };
        let expert_plan: Option<Vec<Vec<bool>>> = match (split, placement.vram_budget) {
            (true, Some(budget)) => {
                let per_expert: Vec<u64> = (0..hp.n_layer).map(|i| l.expert_bytes(i, hp.n_expert)).collect();
                let experts: u64 = per_expert.iter().map(|b| b * hp.n_expert as u64).sum();
                let total: u64 = content.tensor_infos.keys().map(|name| l.bytes(name)).sum();
                let shared = total.saturating_sub(experts).saturating_sub(l.bytes("token_embd.weight"));
                let hits = read_profile(profile.as_deref().unwrap_or(path), hp.n_layer, hp.n_expert);
                Some(moe::plan(&hits, &per_expert, budget.saturating_sub(shared)))
            }
            _ => None,
        };
        // Embedding rows are looked up on the host, so the table stays in system memory.
        let embd = Arc::new(l.q("token_embd.weight")?);
        // The output head follows the last layer, which is the first one offloaded.
        let out_dev = if on_gpu(hp.n_layer.saturating_sub(1)) { placement.device.clone() } else { cpu.clone() };
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
            l.dev = if on_gpu(i) { placement.device.clone() } else { cpu.clone() };
            let p = |s: &str| format!("blk.{i}.{s}");
            let attn_norm = l.f(&p("attn_norm.weight"))?;
            let mixer = match (hp.recurrent[i], hp.delta) {
                (true, Some(dims)) => Mixer::Delta(Box::new(DeltaNet {
                    dims,
                    qkv: l.mm(&p("attn_qkv.weight"))?,
                    gate: l.mm(&p("attn_gate.weight"))?,
                    beta: l.mm(&p("ssm_beta.weight"))?,
                    alpha: l.mm(&p("ssm_alpha.weight"))?,
                    out: l.mm(&p("ssm_out.weight"))?,
                    conv: l.host(&p("ssm_conv1d.weight"))?,
                    dt: l.host(&p("ssm_dt.bias"))?,
                    a: l.host(&p("ssm_a"))?,
                    norm: l.f(&p("ssm_norm.weight"))?,
                    eps: hp.rms_eps as f32,
                })),
                _ => Mixer::Attn(Attn {
                    q_norm: if l.has(&p("attn_q_norm.weight")) { Some(l.f(&p("attn_q_norm.weight"))?) } else { None },
                    k_norm: if l.has(&p("attn_k_norm.weight")) { Some(l.f(&p("attn_k_norm.weight"))?) } else { None },
                    q: l.linear(&p("attn_q"))?,
                    k: l.linear(&p("attn_k"))?,
                    v: l.linear(&p("attn_v"))?,
                    o: l.linear(&p("attn_output"))?,
                }),
            };
            if let Mixer::Delta(d) = &mixer {
                let dd = d.dims;
                let conv_dim = 2 * dd.head_dim * dd.k_heads + dd.head_dim * dd.v_heads;
                if d.conv.len() != conv_dim * dd.kernel || d.dt.len() != dd.v_heads || d.a.len() != dd.v_heads {
                    bail!("layer {i}: gated delta-net tensors do not match the ssm.* metadata");
                }
            }
            let ffn_norm = l.f(&p(if hp.gated_attn { "post_attention_norm.weight" } else { "ffn_norm.weight" }))?;
            let ffn = if is_moe {
                let layer_dev = l.dev.clone();
                let devs: Vec<Device> = (0..hp.n_expert)
                    .map(|e| match &expert_plan {
                        Some(plan) if !plan[i][e] => cpu.clone(),
                        _ => layer_dev.clone(),
                    })
                    .collect();
                let (gates, ups): (Vec<QMatMul>, Vec<QMatMul>) = if l.has(&p("ffn_gate_up_exps.weight")) {
                    l.expert_matrices(&p("ffn_gate_up_exps.weight"), &devs, 2)?.into_iter().map(|mut m| (m.remove(0), m.remove(0))).unzip()
                } else {
                    let g = l.expert_matrices(&p("ffn_gate_exps.weight"), &devs, 1)?.into_iter().map(|mut m| m.remove(0));
                    let u = l.expert_matrices(&p("ffn_up_exps.weight"), &devs, 1)?.into_iter().map(|mut m| m.remove(0));
                    (g.collect(), u.collect())
                };
                let downs = l.expert_matrices(&p("ffn_down_exps.weight"), &devs, 1)?.into_iter().map(|mut m| m.remove(0));
                let experts = gates
                    .into_iter()
                    .zip(ups)
                    .zip(downs)
                    .zip(devs)
                    .map(|(((gate, up), down), dev)| Expert { dev, gate, up, down, hits: AtomicU32::new(0) })
                    .collect();
                let shared = if l.has(&p("ffn_up_shexp.weight")) {
                    Some(SharedExpert {
                        gate: l.mm(&p("ffn_gate_shexp.weight"))?,
                        up: l.mm(&p("ffn_up_shexp.weight"))?,
                        down: l.mm(&p("ffn_down_shexp.weight"))?,
                        gate_inp: l.f(&p("ffn_gate_inp_shexp.weight"))?.flatten_all()?.unsqueeze(0)?,
                    })
                } else {
                    None
                };
                Ffn::Moe(Moe { router: l.mm(&p("ffn_gate_inp.weight"))?, experts, used: hp.n_expert_used, normalize: true, scale: hp.expert_scale, shared })
            } else {
                Ffn::Dense { gate: l.mm(&p("ffn_gate.weight"))?, up: l.mm(&p("ffn_up.weight"))?, down: l.mm(&p("ffn_down.weight"))? }
            };
            layers.push(Layer { dev: l.dev.clone(), attn_norm, mixer, ffn_norm, ffn });
            progress(i + 1, hp.n_layer);
        }
        let half = hp.n_rot / 2;
        let mut inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / hp.rope_base.powf((2 * i) as f32 / hp.n_rot as f32)).collect();
        if let Some(s) = hp.rope_sections {
            // Text positions advance the first three axes together and leave the fourth at zero.
            let total = (s[0] + s[1] + s[2] + s[3]).max(1);
            for (i, f) in inv_freq.iter_mut().enumerate() {
                let sector = i % total;
                if sector >= 3 * s[sector % 3] {
                    *f = 0.0;
                }
            }
        }
        // Llama 3.1+ ships long-context frequency factors as a tensor.
        if l.has("rope_freqs.weight") {
            let factors = l.host("rope_freqs.weight")?;
            for (f, s) in inv_freq.iter_mut().zip(factors) {
                *f /= s;
            }
        }
        let mut model = Self { hp, embd, layers, out_norm, output, cache: CacheSet { layers: Vec::new(), len: 0, checkpoints: Vec::new() }, inv_freq, out_dev, placement: placement.clone(), profile, trace: None, phases: PhaseTimes::default() };
        model.cache = model.empty_cache();
        Ok(model)
    }

    /// An empty cache set for this model, for a new sequence slot.
    pub fn empty_cache(&self) -> CacheSet {
        let layers = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Attn(_) => LayerCache::Kv(KvCache::empty()),
                Mixer::Delta(d) => LayerCache::Rec(RecState::new(&d.dims)),
            })
            .collect();
        CacheSet { layers, len: 0, checkpoints: Vec::new() }
    }

    /// Swap the active cache with `other`, returning the previously active one.
    /// Offloaded layers' KV caches move to the accelerator on activation and back to
    /// system memory when parked, so parked slots do not occupy VRAM.
    /// @param other - cache set produced by [`Model::empty_cache`] or an earlier swap.
    pub fn swap_cache(&mut self, other: CacheSet) -> Result<CacheSet> {
        let to = |set: CacheSet, devs: &mut dyn Iterator<Item = Device>| -> Result<CacheSet> {
            let layers = set
                .layers
                .into_iter()
                .zip(devs)
                .map(|(c, dev)| Ok(match c {
                    LayerCache::Kv(kv) => LayerCache::Kv(kv.moved(&dev)?),
                    rec => rec,
                }))
                .collect::<Result<Vec<_>>>()?;
            Ok(CacheSet { layers, len: set.len, checkpoints: set.checkpoints })
        };
        let incoming = to(other, &mut self.layers.iter().map(|l| l.dev.clone()))?;
        let previous = std::mem::replace(&mut self.cache, incoming);
        to(previous, &mut std::iter::repeat(Device::Cpu))
    }

    /// Write the active cache to a file [`Model::load_cache`] reads.
    /// @param path - destination safetensors file.
    pub fn save_cache(&self, path: &Path) -> Result<()> {
        self.cache.save(path)
    }

    /// Read a cache set written by [`CacheSet::save`] for this model.
    /// @param path - safetensors file.
    /// @returns the set in system memory; activate it with [`Model::swap_cache`].
    pub fn load_cache(&self, path: &Path) -> Result<CacheSet> {
        let t = candle_core::safetensors::load(path, &Device::Cpu)?;
        let get = |name: &str| t.get(name).with_context(|| format!("saved prompt cache has no `{name}`"));
        let floats = |name: &str| -> Result<Vec<f32>> { Ok(get(name)?.to_dtype(DType::F32)?.to_vec1::<f32>()?) };
        let len = get("len")?.to_vec1::<u32>()?.first().copied().unwrap_or(0) as usize;
        let mut set = self.empty_cache();
        let mut dims = Vec::new();
        for (i, (l, layer)) in set.layers.iter_mut().zip(&self.layers).enumerate() {
            match (l, &layer.mixer) {
                (LayerCache::Kv(kv), _) => {
                    if let (Some(k), Some(v)) = (t.get(&format!("kv.{i}.k")), t.get(&format!("kv.{i}.v"))) {
                        if k.dim(2)? != len || v.dim(2)? != len {
                            bail!("saved prompt cache layer {i} holds {} positions, not {len}", k.dim(2)?);
                        }
                        *kv = KvCache { k: Some(k.clone()), v: Some(v.clone()), len };
                    }
                }
                (LayerCache::Rec(s), Mixer::Delta(d)) => {
                    *s = RecState::from_parts(&d.dims, floats(&format!("rec.{i}.conv"))?, floats(&format!("rec.{i}.s"))?)
                        .with_context(|| format!("saved prompt cache layer {i} does not match the model"))?;
                    dims.push(d.dims);
                }
                _ => bail!("saved prompt cache does not match the model's layers"),
            }
        }
        set.len = len;
        for (n, pos) in get("checkpoints")?.to_vec1::<u32>()?.into_iter().enumerate() {
            let states = dims
                .iter()
                .enumerate()
                .map(|(j, d)| {
                    RecState::from_parts(d, floats(&format!("ckpt.{n}.{j}.conv"))?, floats(&format!("ckpt.{n}.{j}.s"))?)
                        .context("saved prompt cache checkpoint does not match the model")
                })
                .collect::<Result<Vec<_>>>()?;
            set.checkpoints.push((pos as usize, states));
        }
        Ok(set)
    }

    /// Discard the active cache.
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

    /// Routed experts on the accelerator and in total; `(0, 0)` without routed experts.
    pub fn offloaded_experts(&self) -> (usize, usize) {
        let experts = self.layers.iter().filter_map(|l| if let Ffn::Moe(m) = &l.ffn { Some(&m.experts) } else { None }).flatten();
        experts.fold((0, 0), |(gpu, all), e| (gpu + usize::from(!e.dev.is_cpu()), all + 1))
    }

    /// Tokens currently held in the cache.
    pub fn cached_len(&self) -> usize {
        self.cache.len
    }

    /// Drop cached positions at and after `len`, keeping the shared prefix.
    /// A model with recurrent layers can only return to a checkpoint, so it may keep fewer positions.
    /// @param len - number of leading positions to keep.
    /// @returns the positions kept; the caller evaluates the sequence again from there.
    pub fn truncate(&mut self, len: usize) -> usize {
        self.cache.truncate(len)
    }

    /// Record a point [`Model::truncate`] can return to; a no-op without recurrent layers.
    pub fn checkpoint(&mut self) {
        self.cache.checkpoint();
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let vocab = self.embd.shape().dims2()?.0;
        if let Some(t) = tokens.iter().find(|t| **t as usize >= vocab) {
            bail!("token id {t} outside vocabulary of {vocab}");
        }
        // One batched row lookup that dequantizes only the requested rows.
        let ids = Tensor::new(tokens, &Device::Cpu)?;
        Ok(self.embd.embedding(&ids)?.to_dtype(DType::F32)?.unsqueeze(0)?)
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

    /// Run the next tokens through the model, appending them to the cache.
    /// @param tokens - tokens following the cached positions.
    /// @returns logits for the last token as a vector of vocabulary size.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let start = self.cached_len();
        let t = tokens.len();
        let (cos, sin) = self.rope_tables(start, t)?;
        // Causal mask over [cached | new] keys for the new queries, built with tensor ops on the device that uses it.
        let causal = |dev: &Device| -> Result<Option<Tensor>> {
            if t < 2 {
                return Ok(None);
            }
            let total = start + t;
            let queries = Tensor::arange(start as u32, total as u32, dev)?.reshape((t, 1))?;
            let keys = Tensor::arange(0u32, total as u32, dev)?.reshape((1, total))?;
            let future = keys.broadcast_gt(&queries)?;
            let blocked = Tensor::full(f32::NEG_INFINITY, (t, total), dev)?;
            Ok(Some(future.where_cond(&blocked, &Tensor::zeros((t, total), DType::F32, dev)?)?.reshape((1, 1, t, total))?))
        };
        let mask = causal(&Device::Cpu)?;
        let mut x = self.embed(tokens)?;
        let eps = self.hp.rms_eps as f32;
        // Host-built tables, copied once to the accelerator when the first offloaded layer needs them.
        let (mut cos, mut sin, mut mask) = (cos, sin, mask);
        for (layer, cache) in self.layers.iter().zip(self.cache.layers.iter_mut()) {
            if !x.device().same_device(&layer.dev) {
                x = x.to_device(&layer.dev)?;
                cos = cos.to_device(&layer.dev)?;
                sin = sin.to_device(&layer.dev)?;
                mask = causal(&layer.dev)?;
            }
            let h = candle_nn::ops::rms_norm(&x, &layer.attn_norm, eps)?;
            let started = Instant::now();
            let y = match (&layer.mixer, cache) {
                (Mixer::Attn(a), LayerCache::Kv(kv)) => attend(&self.hp, a, &h, kv, &cos, &sin, mask.as_ref())?,
                (Mixer::Delta(d), LayerCache::Rec(state)) => d.forward(&h, state, &layer.dev)?,
                _ => bail!("cache set does not belong to this model"),
            };
            match layer.mixer {
                Mixer::Attn(_) => self.phases.attention += started.elapsed(),
                Mixer::Delta(_) => self.phases.recurrent += started.elapsed(),
            }
            x = (x + y)?;
            let h = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, eps)?;
            let started = Instant::now();
            let y = match &layer.ffn {
                Ffn::Dense { gate, up, down } => down.forward(&(candle_nn::ops::silu(&gate.forward(&h)?)? * up.forward(&h)?)?)?,
                Ffn::Moe(m) => m.forward(&h, &layer.dev)?,
            };
            self.phases.ffn += started.elapsed();
            x = (x + y)?;
            if let Some(trace) = &mut self.trace {
                trace.push(x.narrow(1, t - 1, 1)?.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt());
            }
        }
        self.cache.len += t;
        if t > 1 || self.cache.len % CHECKPOINT_INTERVAL == 0 {
            self.cache.checkpoint();
        }
        let started = Instant::now();
        let last = x.narrow(1, t - 1, 1)?.to_device(&self.out_dev)?;
        let last = candle_nn::ops::rms_norm(&last, &self.out_norm, eps)?;
        let logits = self.output.forward(&last)?.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        self.phases.head += started.elapsed();
        // Sampling from non-finite logits repeats one token forever; fail the request instead.
        if self.trace.is_none() && logits.iter().any(|l| !l.is_finite()) {
            return Err(EngineError { code: "NUMERIC_FAILURE", message: format!("the model produced non-finite logits on {}; reload it on the CPU", self.placement.label()) }.into());
        }
        Ok(logits)
    }

    /// Phase times accumulated since the last call, which resets them.
    pub fn take_phases(&mut self) -> PhaseTimes {
        std::mem::take(&mut self.phases)
    }

    /// Vocabulary size of the output head.
    pub fn vocab(&self) -> usize {
        self.embd.shape().dims()[0]
    }
}

impl Drop for Model {
    /// Add this run's expert use counts to the profile beside the model file, so the next load
    /// places the most used experts on the accelerator. A read-only models folder loses the update.
    fn drop(&mut self) {
        let Some(path) = &self.profile else { return };
        let mut rows = read_profile(path, self.hp.n_layer, self.hp.n_expert);
        let mut changed = false;
        for (row, layer) in rows.iter_mut().zip(&self.layers) {
            if let Ffn::Moe(m) = &layer.ffn {
                for (count, expert) in row.iter_mut().zip(&m.experts) {
                    let hits = expert.hits.load(Ordering::Relaxed);
                    changed |= hits > 0;
                    // Halve on overflow so recent use keeps its weight.
                    *count = count.checked_add(hits).unwrap_or(u32::MAX / 2);
                }
            }
        }
        if changed {
            let text: String = rows.iter().map(|r| r.iter().map(u32::to_string).collect::<Vec<_>>().join(" ") + "\n").collect();
            let _ = std::fs::write(path, text);
        }
    }
}

/// Rotate the leading `n_rot` dimensions of each head by position.
fn rotate(hp: &Hparams, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let apply = |x: &Tensor| -> Result<Tensor> {
        Ok(match hp.rope {
            RopeStyle::Interleaved => candle_nn::rotary_emb::rope_i(x, cos, sin)?,
            RopeStyle::Neox => candle_nn::rotary_emb::rope(x, cos, sin)?,
        })
    };
    if hp.n_rot == hp.head_dim {
        return apply(x);
    }
    let rotated = apply(&x.narrow(3, 0, hp.n_rot)?.contiguous()?)?;
    Ok(Tensor::cat(&[&rotated, &x.narrow(3, hp.n_rot, hp.head_dim - hp.n_rot)?], 3)?)
}

/// Causal self-attention over the cached and new positions.
/// @param h - normalized layer input, `(1, t, n_embd)`.
/// @param cache - keys and values of the earlier positions; the new ones are appended.
/// @param mask - causal mask for a multi-token step.
/// @returns `(1, t, n_embd)`.
fn attend(hp: &Hparams, a: &Attn, h: &Tensor, cache: &mut KvCache, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let t = h.dim(1)?;
    let eps = hp.rms_eps as f32;
    let scale = 1.0 / (hp.head_dim as f64).sqrt();
    let groups = hp.n_head / hp.n_head_kv;
    let (q, gate) = if hp.gated_attn {
        // The projection emits [query | gate] per head.
        let full = a.q.forward(h)?.reshape((1, t, hp.n_head, 2 * hp.head_dim))?;
        let gate = full.narrow(3, hp.head_dim, hp.head_dim)?.contiguous()?.reshape((1, t, hp.n_head * hp.head_dim))?;
        (full.narrow(3, 0, hp.head_dim)?, Some(gate))
    } else {
        (a.q.forward(h)?.reshape((1, t, hp.n_head, hp.head_dim))?, None)
    };
    let k = a.k.forward(h)?.reshape((1, t, hp.n_head_kv, hp.head_dim))?;
    let v = a.v.forward(h)?.reshape((1, t, hp.n_head_kv, hp.head_dim))?.transpose(1, 2)?;
    let q = match &a.q_norm {
        Some(n) => candle_nn::ops::rms_norm(&q.contiguous()?, n, eps)?,
        None => q,
    };
    let k = match &a.k_norm {
        Some(n) => candle_nn::ops::rms_norm(&k.contiguous()?, n, eps)?,
        None => k,
    };
    let q = rotate(hp, &q.transpose(1, 2)?.contiguous()?, cos, sin)?;
    let k = rotate(hp, &k.transpose(1, 2)?.contiguous()?, cos, sin)?;
    // On a GPU the cache holds half-precision keys and values, halving the memory a long context needs.
    let kv_dtype = if h.device().is_cpu() { DType::F32 } else { DType::F16 };
    let (k, v) = cache.append(&k.to_dtype(kv_dtype)?, &v.to_dtype(kv_dtype)?)?;
    let total = k.dim(2)?;
    // Grouped-query attention: fold query heads per KV head instead of copying K/V.
    let q = q.reshape((1, hp.n_head_kv, groups * t, hp.head_dim))?.to_dtype(kv_dtype)?;
    // The cached keys and values are strided views of the cache buffers; copying them would cost a pass
    // over the whole context per layer per token.
    let att = (q.matmul(&k.t()?)?.to_dtype(DType::F32)? * scale)?;
    let att = match mask {
        Some(m) => {
            let att = att.reshape((1, hp.n_head_kv, groups, t, total))?;
            att.broadcast_add(&m.unsqueeze(1)?)?.reshape((1, hp.n_head_kv, groups * t, total))?
        }
        None => att,
    };
    let att = candle_nn::ops::softmax_last_dim(&att)?.to_dtype(kv_dtype)?;
    let y = att.matmul(&v)?.to_dtype(DType::F32)?;
    let y = y.reshape((1, hp.n_head, t, hp.head_dim))?.transpose(1, 2)?.reshape((1, t, hp.n_head * hp.head_dim))?;
    let y = match gate {
        Some(g) => (y * candle_nn::ops::sigmoid(&g)?)?,
        None => y,
    };
    a.o.forward(&y)
}
