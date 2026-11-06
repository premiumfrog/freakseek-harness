//! Synthetic GGUF fixtures shared by the integration tests.
#![allow(dead_code)]

use std::fs::File;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

/// Vocabulary: 256 byte-level symbols plus ChatML control tokens.
pub const VOCAB: usize = 259;
const EMBD: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const FFN: usize = 128;
const LAYERS: usize = 2;

fn rand_q(shape: &[usize], dtype: GgmlDType, scale: f32) -> QTensor {
    let t = (Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap() * scale as f64).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

fn ones_ish(n: usize) -> QTensor {
    let t = (Tensor::randn(0f32, 0.1f32, n, &Device::Cpu).unwrap() + 1.0).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Write a random model of `arch` to a temporary path.
pub fn write_model(arch: &str, tied: bool, tokenizer: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dsh-gguf-ref-{}-{arch}-{tied}-{tokenizer}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.gguf");
    let k = |s: &str| format!("{arch}.{s}");
    let mut md: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(arch.into())),
        (k("block_count"), Value::U32(LAYERS as u32)),
        (k("embedding_length"), Value::U32(EMBD as u32)),
        (k("feed_forward_length"), Value::U32(FFN as u32)),
        (k("context_length"), Value::U32(512)),
        (k("attention.head_count"), Value::U32(HEADS as u32)),
        (k("attention.head_count_kv"), Value::U32(KV_HEADS as u32)),
        (k("attention.key_length"), Value::U32(HEAD_DIM as u32)),
        (k("attention.value_length"), Value::U32(HEAD_DIM as u32)),
        (k("rope.dimension_count"), Value::U32(HEAD_DIM as u32)),
        (k("rope.freq_base"), Value::F32(10000.0)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-5)),
        // Selects F32 compute in the qwen3 reference, which defaults to F16.
        ("general.dtype".into(), Value::U32(0)),
    ];
    let q = GgmlDType::Q8_0;
    let mut tensors: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), rand_q(&[VOCAB, EMBD], q, 0.5)),
        ("output_norm.weight".into(), ones_ish(EMBD)),
    ];
    if !tied {
        tensors.push(("output.weight".into(), rand_q(&[VOCAB, EMBD], q, 0.2)));
    }
    for i in 0..LAYERS {
        let p = |s: &str| format!("blk.{i}.{s}");
        tensors.push((p("attn_norm.weight"), ones_ish(EMBD)));
        tensors.push((p("attn_q.weight"), rand_q(&[HEADS * HEAD_DIM, EMBD], q, 0.2)));
        tensors.push((p("attn_k.weight"), rand_q(&[KV_HEADS * HEAD_DIM, EMBD], q, 0.2)));
        tensors.push((p("attn_v.weight"), rand_q(&[KV_HEADS * HEAD_DIM, EMBD], q, 0.2)));
        tensors.push((p("attn_output.weight"), rand_q(&[EMBD, HEADS * HEAD_DIM], q, 0.2)));
        tensors.push((p("ffn_norm.weight"), ones_ish(EMBD)));
        tensors.push((p("ffn_gate.weight"), rand_q(&[FFN, EMBD], q, 0.2)));
        tensors.push((p("ffn_up.weight"), rand_q(&[FFN, EMBD], q, 0.2)));
        tensors.push((p("ffn_down.weight"), rand_q(&[EMBD, FFN], q, 0.2)));
        if arch == "qwen2" {
            tensors.push((p("attn_q.bias"), rand_q(&[HEADS * HEAD_DIM], GgmlDType::F32, 0.1)));
            tensors.push((p("attn_k.bias"), rand_q(&[KV_HEADS * HEAD_DIM], GgmlDType::F32, 0.1)));
            tensors.push((p("attn_v.bias"), rand_q(&[KV_HEADS * HEAD_DIM], GgmlDType::F32, 0.1)));
        }
        if arch == "qwen3" {
            tensors.push((p("attn_q_norm.weight"), ones_ish(HEAD_DIM)));
            tensors.push((p("attn_k_norm.weight"), ones_ish(HEAD_DIM)));
        }
    }
    if tokenizer {
        md.extend(tokenizer_metadata());
    }
    let md_ref: Vec<(&str, &Value)> = md.iter().map(|(a, b)| (a.as_str(), b)).collect();
    let t_ref: Vec<(&str, &QTensor)> = tensors.iter().map(|(a, b)| (a.as_str(), b)).collect();
    let mut f = File::create(&path).unwrap();
    gguf_file::write(&mut f, &md_ref, &t_ref).unwrap();
    path
}

pub fn content(path: &Path) -> gguf_file::Content {
    gguf_file::Content::read(&mut File::open(path).unwrap()).unwrap()
}

/// Byte-level BPE tokenizer metadata with no merges and ChatML control tokens.
fn tokenizer_metadata() -> Vec<(String, Value)> {
    let mut table = Vec::new();
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table.push(if printable { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(255 + n).unwrap() });
    }
    let mut tokens: Vec<Value> = table.iter().map(|c| Value::String(c.to_string())).collect();
    let mut types: Vec<Value> = vec![Value::I32(1); 256];
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        tokens.push(Value::String(s.into()));
        types.push(Value::I32(3));
    }
    vec![
        ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), Value::String("qwen2".into())),
        ("tokenizer.ggml.tokens".into(), Value::Array(tokens)),
        ("tokenizer.ggml.token_type".into(), Value::Array(types)),
        ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(257)),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ("general.name".into(), Value::String("Synthetic test model".into())),
    ]
}
