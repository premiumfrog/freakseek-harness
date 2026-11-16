//! Mixture-of-experts feed-forward block (`qwen3moe`, `qwen35moe`).
//!
//! A router picks `used` experts per token. Each expert keeps its own quantized
//! matrices on its own device, so the most used experts can sit in VRAM while the
//! rest run from system memory; a token's rows travel to an expert's device only
//! when that expert is selected.

use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};

/// One routed expert.
pub struct Expert {
    pub dev: Device,
    pub gate: QMatMul,
    pub up: QMatMul,
    pub down: QMatMul,
    /// Tokens routed here since load; feeds the placement profile.
    pub hits: AtomicU32,
}

/// Always-on expert added to the routed output, scaled by a learned gate.
pub struct SharedExpert {
    pub gate: QMatMul,
    pub up: QMatMul,
    pub down: QMatMul,
    /// Row vector whose dot product with the input is the gate logit.
    pub gate_inp: Tensor,
}

/// Weights of one MoE block.
pub struct Moe {
    pub router: QMatMul,
    pub experts: Vec<Expert>,
    pub used: usize,
    /// Divide the selected weights by their sum.
    pub normalize: bool,
    /// Multiplier on the selected weights; 1 leaves them unchanged.
    pub scale: f32,
    pub shared: Option<SharedExpert>,
}

fn ffn(gate: &QMatMul, up: &QMatMul, down: &QMatMul, x: &Tensor) -> Result<Tensor> {
    Ok(down.forward(&(candle_nn::ops::silu(&gate.forward(x)?)? * up.forward(x)?)?)?)
}

/// Pick the `used` largest router probabilities of one token.
/// @param logits - router output for the token.
/// @param used - experts to select.
/// @param normalize - divide the selected probabilities by their sum.
/// @param scale - multiplier applied afterwards.
/// @returns `(expert, weight)` pairs.
pub fn route(logits: &[f32], used: usize, normalize: bool, scale: f32) -> Vec<(usize, f32)> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x));
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| exp[*b].total_cmp(&exp[*a]).then(a.cmp(b)));
    order.truncate(used);
    let picked: Vec<(usize, f32)> = order.into_iter().map(|e| (e, exp[e] / total)).collect();
    // The floor is the smallest positive half-precision normal, as in llama.cpp.
    let sum = if normalize { picked.iter().map(|(_, w)| w).sum::<f32>().max(6.103_515_6e-5) } else { 1.0 };
    picked.into_iter().map(|(e, w)| (e, w / sum * scale)).collect()
}

impl Moe {
    /// Apply the block to `t` positions.
    /// @param h - normalized layer input, `(1, t, n_embd)` on `dev`.
    /// @param dev - device of the router and of the result.
    /// @returns `(1, t, n_embd)` on `dev`.
    pub fn forward(&self, h: &Tensor, dev: &Device) -> Result<Tensor> {
        let (_, t, n_embd) = h.dims3()?;
        let x = h.reshape((t, n_embd))?;
        let n = self.experts.len();
        let logits = self.router.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        // Rows and weights routed to each expert, so every expert runs one batched product.
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut weights: Vec<Vec<f32>> = vec![Vec::new(); n];
        for p in 0..t {
            for (e, w) in route(&logits[p * n..(p + 1) * n], self.used, self.normalize, self.scale) {
                rows[e].push(p as u32);
                weights[e].push(w);
            }
        }
        let mut out = match &self.shared {
            Some(s) => {
                let gate = candle_nn::ops::sigmoid(&x.broadcast_mul(&s.gate_inp)?.sum_keepdim(1)?)?;
                ffn(&s.gate, &s.up, &s.down, &x)?.broadcast_mul(&gate)?
            }
            None => Tensor::zeros((t, n_embd), x.dtype(), dev)?,
        };
        // Rows cross to the host once, and only when a selected expert lives there.
        let mut on_host: Option<Tensor> = None;
        for (e, expert) in self.experts.iter().enumerate() {
            if rows[e].is_empty() {
                continue;
            }
            expert.hits.fetch_add(rows[e].len() as u32, Ordering::Relaxed);
            let count = rows[e].len();
            let src = if expert.dev.same_device(dev) {
                &x
            } else {
                if on_host.is_none() {
                    on_host = Some(x.to_device(&expert.dev)?);
                }
                on_host.as_ref().unwrap_or(&x)
            };
            let all = count == t;
            let xe = if all { src.clone() } else { src.index_select(&Tensor::from_vec(rows[e].clone(), count, &expert.dev)?, 0)? };
            let w = Tensor::from_vec(std::mem::take(&mut weights[e]), (count, 1), &expert.dev)?;
            let y = ffn(&expert.gate, &expert.up, &expert.down, &xe)?.broadcast_mul(&w)?.to_device(dev)?;
            out = if all { (out + y)? } else { out.index_add(&Tensor::from_vec(std::mem::take(&mut rows[e]), count, dev)?, &y, 0)? };
        }
        Ok(out.reshape((1, t, n_embd))?)
    }
}

/// Choose which experts go to the accelerator.
/// @param hits - per layer, per expert use counts from an earlier run; empty rows mean no history.
/// @param bytes - per layer, bytes of one expert.
/// @param budget - accelerator bytes available for experts.
/// @returns per layer, per expert, whether it is placed on the accelerator.
pub fn plan(hits: &[Vec<u32>], bytes: &[u64], budget: u64) -> Vec<Vec<bool>> {
    let mut placed: Vec<Vec<bool>> = hits.iter().map(|l| vec![false; l.len()]).collect();
    let mut order: Vec<(usize, usize)> = hits.iter().enumerate().flat_map(|(l, row)| (0..row.len()).map(move |e| (l, e))).collect();
    // Most used first; without history, low expert indexes first so every layer gets an equal share.
    order.sort_by(|a, b| hits[b.0][b.1].cmp(&hits[a.0][a.1]).then(a.1.cmp(&b.1)).then(a.0.cmp(&b.0)));
    let mut left = budget;
    for (l, e) in order {
        if bytes[l] <= left {
            left -= bytes[l];
            placed[l][e] = true;
        }
    }
    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_normalizes_the_selected_weights() {
        let picked = route(&[0.0, 2.0, 1.0, -1.0], 2, true, 1.0);
        assert_eq!(picked.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2]);
        assert!((picked.iter().map(|p| p.1).sum::<f32>() - 1.0).abs() < 1e-6);
        let raw = route(&[0.0, 2.0, 1.0, -1.0], 2, false, 1.0);
        assert!(raw.iter().map(|p| p.1).sum::<f32>() < 1.0);
    }

    #[test]
    fn plan_prefers_used_experts_within_the_budget() {
        let placed = plan(&[vec![1, 9, 0], vec![5, 0, 7]], &[10, 10], 30);
        assert_eq!(placed, vec![vec![false, true, false], vec![true, false, true]]);
        let cold = plan(&[vec![0, 0], vec![0, 0]], &[10, 10], 20);
        assert_eq!(cold, vec![vec![true, false], vec![true, false]]);
    }
}
