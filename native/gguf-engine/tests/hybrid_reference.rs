//! Forward-pass parity of the Qwen 3.5 architectures with llama.cpp.
//!
//! Each case writes a deterministic model and compares the engine's logits with a
//! fixture llama.cpp produced for the same file and token sequence. To refresh a
//! fixture, run with `DSH_WRITE_HYBRID_MODELS=<dir>` to write `<arch>.gguf`
//! there, then evaluate `PROMPT` followed by `DECODE` with llama.cpp and store
//! the last-position logits of each step, one step per line.

use std::path::Path;

use dsh_gguf_engine::device::Placement;
use dsh_gguf_engine::model::Model;

mod common;

use common::{content, write_hybrid};

const PROMPT: [u32; 9] = [1, 17, 42, 5, 88, 23, 9, 200, 131];
const DECODE: [u32; 3] = [3, 64, 12];

fn fixture(arch: &str) -> Vec<Vec<f32>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/fixtures/{arch}.logits"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect()).collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
    let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(worst / scale < 2e-3, "{what}: max abs diff {worst} vs scale {scale}");
}

fn parity(arch: &str) {
    if let Ok(dir) = std::env::var("DSH_WRITE_HYBRID_MODELS") {
        write_hybrid(arch, &Path::new(&dir).join(format!("{arch}.gguf")));
        return;
    }
    let dir = std::env::temp_dir().join(format!("dsh-gguf-hybrid-{}-{arch}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.gguf");
    write_hybrid(arch, &path);
    let expected = fixture(arch);
    let mut engine = Model::load(&path, &content(&path), &Placement::cpu()).unwrap();
    assert_close(&engine.forward(&PROMPT).unwrap(), &expected[0], &format!("{arch} prefill"));
    for (i, t) in DECODE.iter().enumerate() {
        assert_close(&engine.forward(&[*t]).unwrap(), &expected[i + 1], &format!("{arch} decode {i}"));
    }
    // Chunked prefill carries the convolution window and the recurrent state across calls.
    let mut chunked = Model::load(&path, &content(&path), &Placement::cpu()).unwrap();
    chunked.forward(&PROMPT[..4]).unwrap();
    assert_close(&chunked.forward(&PROMPT[4..]).unwrap(), &expected[0], &format!("{arch} chunked prefill"));
    // Prefix reuse returns to the checkpoint after the first chunk, not to the requested position.
    assert_eq!(chunked.truncate(6), 4);
    assert_close(&chunked.forward(&PROMPT[4..]).unwrap(), &expected[0], &format!("{arch} checkpoint reuse"));
    // Without a checkpoint at or before the cut, the sequence restarts.
    assert_eq!(chunked.truncate(2), 0);
    assert_close(&chunked.forward(&PROMPT).unwrap(), &expected[0], &format!("{arch} restart"));
    // A parked sequence resumes where it stopped.
    let parked = chunked.swap_cache(chunked.empty_cache()).unwrap();
    chunked.forward(&[7, 8]).unwrap();
    chunked.swap_cache(parked).unwrap();
    assert_close(&chunked.forward(&[DECODE[0]]).unwrap(), &expected[1], &format!("{arch} resumed slot"));
    drop((engine, chunked));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn qwen35_matches_llama_cpp() {
    parity("qwen35");
}

#[test]
fn qwen35moe_matches_llama_cpp() {
    parity("qwen35moe");
}
