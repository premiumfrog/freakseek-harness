//! Forward-pass parity with candle-transformers' reference GGUF models.
//!
//! Each case writes a small random GGUF model, then compares the engine's
//! logits with the reference implementation for prefill, incremental decoding,
//! and KV-cache prefix reuse.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::{Device, Tensor};
use dsh_gguf_engine::device::Placement;
use dsh_gguf_engine::model::Model;

mod common;

use common::{content, write_model};

/// Reference model behind one interface: feed tokens at a position, return last logits.
enum Reference {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
    Qwen3(candle_transformers::models::quantized_qwen3::ModelWeights),
}

impl Reference {
    fn load(arch: &str, path: &Path) -> Self {
        let c = content(path);
        let mut r = BufReader::new(File::open(path).unwrap());
        let d = Device::Cpu;
        match arch {
            "llama" => Self::Llama(candle_transformers::models::quantized_llama::ModelWeights::from_gguf(c, &mut r, &d).unwrap()),
            "qwen2" => Self::Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(c, &mut r, &d).unwrap()),
            "qwen3" => Self::Qwen3(candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(c, &mut r, &d).unwrap()),
            _ => unreachable!(),
        }
    }

    fn forward(&mut self, tokens: &[u32], pos: usize) -> Vec<f32> {
        let x = Tensor::new(tokens, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let y = match self {
            Self::Llama(m) => m.forward(&x, pos),
            Self::Qwen2(m) => m.forward(&x, pos),
            Self::Qwen3(m) => m.forward(&x, pos),
        }
        .unwrap();
        y.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(worst / scale < 2e-3, "{what}: max abs diff {worst} vs scale {scale}");
}

fn parity(arch: &str, tied: bool) {
    let path = write_model(arch, tied, false);
    let c = content(&path);
    let mut engine = Model::load(&path, &c, &Placement::cpu()).unwrap();
    let mut reference = Reference::load(arch, &path);
    let prompt = [1u32, 17, 42, 5, 88, 23, 9];
    assert_close(&engine.forward(&prompt).unwrap(), &reference.forward(&prompt, 0), &format!("{arch} prefill"));
    let mut pos = prompt.len();
    for t in [3u32, 64, 12] {
        assert_close(&engine.forward(&[t]).unwrap(), &reference.forward(&[t], pos), &format!("{arch} decode @{pos}"));
        pos += 1;
    }
    // Prefix reuse: truncating to 4 positions then refeeding must equal a fresh prefill.
    engine.truncate(4);
    let refeed = [88u32, 23, 9, 3];
    let mut fresh = Reference::load(arch, &path);
    let full: Vec<u32> = prompt[..4].iter().chain(&refeed).copied().collect();
    assert_close(&engine.forward(&refeed).unwrap(), &fresh.forward(&full, 0), &format!("{arch} prefix reuse"));
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn llama_matches_reference() {
    parity("llama", false);
}

#[test]
fn llama_tied_output_matches_reference() {
    parity("llama", true);
}

#[test]
fn qwen2_matches_reference() {
    parity("qwen2", false);
}

#[test]
fn qwen3_matches_reference() {
    parity("qwen3", false);
}
