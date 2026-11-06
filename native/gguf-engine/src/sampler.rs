//! Token sampling: repetition penalty, temperature, top-k, top-p, and min-p.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

/// Sampling controls for one generation; absent fields use the listed defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SamplingParams {
    /// 0 selects greedy decoding.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    /// Window of recent tokens the repetition penalty considers.
    pub repeat_last_n: usize,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.1, repeat_last_n: 64, seed: None }
    }
}

/// Stateful sampler for one generation.
pub struct Sampler {
    p: SamplingParams,
    rng: StdRng,
}

impl Sampler {
    /// @param p - sampling controls.
    pub fn new(p: SamplingParams) -> Self {
        let rng = match p.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { p, rng }
    }

    /// Choose the next token.
    /// @param logits - last-position logits, modified in place.
    /// @param history - tokens of the sequence so far.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        if self.p.repeat_penalty != 1.0 {
            let start = history.len().saturating_sub(self.p.repeat_last_n);
            for &t in &history[start..] {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / self.p.repeat_penalty } else { *l * self.p.repeat_penalty };
                }
            }
        }
        if self.p.temperature <= 0.0 {
            return argmax(logits);
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        let k = if self.p.top_k == 0 { logits.len() } else { self.p.top_k.min(logits.len()) };
        idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let max = logits[idx[0]];
        let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) / self.p.temperature).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let floor = self.p.min_p * probs[0];
        let mut cum = 0.0;
        let mut keep = 0;
        for &p in &probs {
            if keep > 0 && (p < floor || cum >= self.p.top_p) {
                break;
            }
            cum += p;
            keep += 1;
        }
        let total: f32 = probs[..keep].iter().sum();
        let mut r = self.rng.gen::<f32>() * total;
        for (j, &p) in probs[..keep].iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return idx[j] as u32;
            }
        }
        idx[keep - 1] as u32
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, repeat_penalty: 1.0, ..Default::default() });
        assert_eq!(s.sample(&mut [0.1, 3.0, 2.0], &[]), 1);
    }

    #[test]
    fn repeat_penalty_demotes_seen_token() {
        let mut s = Sampler::new(SamplingParams { temperature: 0.0, repeat_penalty: 2.0, ..Default::default() });
        assert_eq!(s.sample(&mut [0.0, 3.0, 2.0], &[1]), 2);
    }

    #[test]
    fn seeded_sampling_stays_within_top_k() {
        let mut s = Sampler::new(SamplingParams { top_k: 2, seed: Some(7), repeat_penalty: 1.0, min_p: 0.0, ..Default::default() });
        for _ in 0..50 {
            let t = s.sample(&mut [5.0, 4.9, -10.0, -10.0], &[]);
            assert!(t < 2);
        }
    }
}
