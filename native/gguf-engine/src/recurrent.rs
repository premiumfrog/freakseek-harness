//! Gated delta-net mixer of the Qwen 3.5 family (`qwen35`, `qwen35moe`).
//!
//! A recurrent layer replaces attention with a per-head state matrix updated once
//! per token, preceded by a short causal depthwise convolution. The state has a
//! fixed size, so it cannot be cut back to an earlier position the way a KV cache
//! can; [`RecState`] is cloned into checkpoints instead. The recurrence runs on the
//! host: it is sequential per token and parallel only across heads.

use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use rayon::prelude::*;

/// Dimensions of one gated delta-net layer.
#[derive(Debug, Clone, Copy)]
pub struct DeltaDims {
    /// Width of one key, query, or value head (`ssm.state_size`).
    pub head_dim: usize,
    /// Key/query heads (`ssm.group_count`).
    pub k_heads: usize,
    /// Value heads (`ssm.time_step_rank`).
    pub v_heads: usize,
    /// Convolution taps (`ssm.conv_kernel`).
    pub kernel: usize,
}

impl DeltaDims {
    fn key_dim(&self) -> usize {
        self.head_dim * self.k_heads
    }
    fn value_dim(&self) -> usize {
        self.head_dim * self.v_heads
    }
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// Recurrent memory of one layer for one sequence, held in system memory.
#[derive(Debug, Clone)]
pub struct RecState {
    /// Last `kernel - 1` convolution inputs, row per position.
    conv: Vec<f32>,
    /// Per value head, a `head_dim x head_dim` matrix stored as rows of the transposed state.
    s: Vec<f32>,
}

impl RecState {
    /// Zeroed state for a new sequence.
    /// @param d - layer dimensions.
    pub fn new(d: &DeltaDims) -> Self {
        Self { conv: vec![0.0; (d.kernel - 1) * d.conv_dim()], s: vec![0.0; d.v_heads * d.head_dim * d.head_dim] }
    }

    /// The convolution window and state matrices, for saving.
    pub fn parts(&self) -> (&[f32], &[f32]) {
        (&self.conv, &self.s)
    }

    /// Rebuild a state from saved parts.
    /// @param d - layer dimensions the parts must match.
    /// @returns `None` when the sizes do not match `d`.
    pub fn from_parts(d: &DeltaDims, conv: Vec<f32>, s: Vec<f32>) -> Option<Self> {
        let empty = Self::new(d);
        (conv.len() == empty.conv.len() && s.len() == empty.s.len()).then_some(Self { conv, s })
    }

    /// Return to the state of an empty sequence.
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.s.fill(0.0);
    }
}

/// Weights of one gated delta-net layer.
pub struct DeltaNet {
    pub dims: DeltaDims,
    pub qkv: QMatMul,
    pub gate: QMatMul,
    pub beta: QMatMul,
    pub alpha: QMatMul,
    pub out: QMatMul,
    /// Depthwise kernel, `conv_dim` rows of `kernel` taps.
    pub conv: Vec<f32>,
    pub dt: Vec<f32>,
    pub a: Vec<f32>,
    pub norm: Tensor,
    pub eps: f32,
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Dot product with eight independent accumulators, which the compiler maps to SIMD lanes; a single running sum
/// stays scalar because floating-point addition is not reassociated.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

impl DeltaNet {
    /// Mix `t` new positions, advancing `state`.
    /// @param h - normalized layer input, `(1, t, n_embd)` on `dev`.
    /// @param state - recurrent memory of the sequence at the position before `h`.
    /// @param dev - device of this layer's weights.
    /// @returns the mixer output, `(1, t, n_embd)` on `dev`.
    pub fn forward(&self, h: &Tensor, state: &mut RecState, dev: &Device) -> Result<Tensor> {
        let d = self.dims;
        let t = h.dim(1)?;
        let (s, key, conv_dim, k) = (d.head_dim, d.key_dim(), d.conv_dim(), d.kernel);
        let mixed = host(&self.qkv.forward(h)?)?;
        let beta = host(&self.beta.forward(h)?)?;
        let alpha = host(&self.alpha.forward(h)?)?;

        // Causal depthwise convolution over [carried inputs | new inputs], then SiLU.
        let mut input = std::mem::take(&mut state.conv);
        input.extend_from_slice(&mixed);
        let mut conv = vec![0f32; t * conv_dim];
        conv.par_chunks_mut(conv_dim).enumerate().for_each(|(p, row)| {
            for (c, out) in row.iter_mut().enumerate() {
                let taps = &self.conv[c * k..(c + 1) * k];
                *out = silu((0..k).map(|i| input[(p + i) * conv_dim + c] * taps[i]).sum());
            }
        });
        state.conv = input[t * conv_dim..].to_vec();

        // Queries and keys are unit vectors per head.
        conv.par_chunks_mut(conv_dim).for_each(|row| {
            for head in row[..2 * key].chunks_mut(s) {
                let scale = 1.0 / (dot(head, head) + self.eps).sqrt();
                head.iter_mut().for_each(|x| *x *= scale);
            }
        });

        let scale = 1.0 / (s as f32).sqrt();
        let mut out = vec![0f32; t * d.value_dim()];
        // One task per value head; each owns its state matrix and writes a disjoint column block of `out`.
        let heads: Vec<Vec<f32>> = state
            .s
            .par_chunks_mut(s * s)
            .enumerate()
            .map(|(head, m)| {
                let kq = head % d.k_heads;
                let mut y = vec![0f32; t * s];
                let mut delta = vec![0f32; s];
                for p in 0..t {
                    let row = &conv[p * conv_dim..(p + 1) * conv_dim];
                    let q = &row[kq * s..(kq + 1) * s];
                    let kk = &row[key + kq * s..key + (kq + 1) * s];
                    let v = &row[2 * key + head * s..2 * key + (head + 1) * s];
                    let b = 1.0 / (1.0 + (-beta[p * d.v_heads + head]).exp());
                    let x = alpha[p * d.v_heads + head] + self.dt[head];
                    // softplus, linear above the point where exp would overflow.
                    let decay = (if x > 20.0 { x } else { x.exp().ln_1p() } * self.a[head]).exp();
                    m.iter_mut().for_each(|w| *w *= decay);
                    for j in 0..s {
                        delta[j] = (v[j] - dot(&m[j * s..(j + 1) * s], kk)) * b;
                    }
                    for j in 0..s {
                        let dj = delta[j];
                        m[j * s..(j + 1) * s].iter_mut().zip(kk).for_each(|(w, kv)| *w += dj * kv);
                    }
                    for j in 0..s {
                        y[p * s + j] = dot(&m[j * s..(j + 1) * s], q) * scale;
                    }
                }
                y
            })
            .collect();
        for (head, y) in heads.iter().enumerate() {
            for p in 0..t {
                out[p * d.value_dim() + head * s..p * d.value_dim() + (head + 1) * s].copy_from_slice(&y[p * s..(p + 1) * s]);
            }
        }

        let y = Tensor::from_vec(out, (1, t, d.v_heads, s), dev)?;
        let y = candle_nn::ops::rms_norm(&y, &self.norm, self.eps)?;
        let z = self.gate.forward(h)?.reshape((1, t, d.v_heads, s))?;
        let y = (y * candle_nn::ops::silu(&z)?)?.reshape((1, t, d.value_dim()))?;
        Ok(self.out.forward(&y)?)
    }
}
