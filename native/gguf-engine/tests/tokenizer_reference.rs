//! Tokenizer parity with llama.cpp's `test-tokenizer-0` reference vocabularies.
//!
//! Set `DSH_GGUF_VOCAB_DIR` to a directory holding llama.cpp's
//! `models/ggml-vocab-<name>.gguf{,.inp,.out}` files; the test skips otherwise.

use std::fs::File;
use std::path::PathBuf;

use candle_core::quantized::gguf_file::Content;
use dsh_gguf_engine::tokenizer::Tokenizer;

fn check(name: &str) {
    let Some(dir) = std::env::var_os("DSH_GGUF_VOCAB_DIR").map(PathBuf::from) else {
        eprintln!("skipping {name}: DSH_GGUF_VOCAB_DIR is unset");
        return;
    };
    let base = dir.join(format!("ggml-vocab-{name}.gguf"));
    let content = Content::read(&mut File::open(&base).unwrap()).unwrap();
    let tok = Tokenizer::from_gguf(&content).unwrap();
    let inputs = std::fs::read_to_string(base.with_extension("gguf.inp")).unwrap();
    let outputs = std::fs::read_to_string(base.with_extension("gguf.out")).unwrap();
    let inputs: Vec<&str> = inputs.split("\n__ggml_vocab_test__\n").collect();
    let outputs: Vec<&str> = outputs.split('\n').collect();
    let mut failures = 0;
    for (i, text) in inputs.iter().enumerate() {
        let text = if i + 1 == inputs.len() { text.strip_suffix("\n__ggml_vocab_test__").unwrap_or(text) } else { text };
        let expected: Vec<u32> = outputs[i].split_whitespace().map(|t| t.parse().unwrap()).collect();
        let got = tok.encode_plain_text(text);
        if got != expected {
            failures += 1;
            eprintln!("{name} #{i} {text:?}\n  expected {expected:?}\n  got      {got:?}");
        }
    }
    assert_eq!(failures, 0, "{name}: {failures} of {} cases differ", inputs.len());
}

#[test]
fn llama_spm() { check("llama-spm") }
#[test]
fn llama_bpe() { check("llama-bpe") }
#[test]
fn qwen2() { check("qwen2") }
#[test]
fn phi_3() { check("phi-3") }
#[test]
fn gpt_2() { check("gpt-2") }
