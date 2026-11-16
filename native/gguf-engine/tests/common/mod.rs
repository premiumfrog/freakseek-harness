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

/// Deterministic generator, so a model written twice has identical weights.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn tensor(&mut self, shape: &[usize], dtype: GgmlDType, scale: f32, offset: f32) -> QTensor {
        let data: Vec<f32> = (0..shape.iter().product()).map(|_| self.next() * scale + offset).collect();
        QTensor::quantize(&Tensor::from_vec(data, shape, &Device::Cpu).unwrap(), dtype).unwrap()
    }
}

/// Write a deterministic Qwen 3.5 model: three gated delta-net layers and one gated attention layer.
/// @param arch - `qwen35` (dense feed-forward) or `qwen35moe` (routed and shared experts).
/// @param path - destination file.
pub fn write_hybrid(arch: &str, path: &Path) {
    const LAYERS: usize = 4;
    const STATE: usize = 16;
    const K_HEADS: usize = 2;
    const V_HEADS: usize = 4;
    const KERNEL: usize = 4;
    const EXPERTS: usize = 4;
    const EXPERT_FFN: usize = 32;
    let (key, value) = (STATE * K_HEADS, STATE * V_HEADS);
    let moe = arch == "qwen35moe";
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
        (k("rope.dimension_sections"), Value::Array(vec![Value::U32(2); 4])),
        (k("rope.freq_base"), Value::F32(10000.0)),
        (k("attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
        (k("full_attention_interval"), Value::U32(4)),
        (k("ssm.conv_kernel"), Value::U32(KERNEL as u32)),
        (k("ssm.state_size"), Value::U32(STATE as u32)),
        (k("ssm.group_count"), Value::U32(K_HEADS as u32)),
        (k("ssm.time_step_rank"), Value::U32(V_HEADS as u32)),
        (k("ssm.inner_size"), Value::U32(value as u32)),
    ];
    if moe {
        md.push((k("expert_count"), Value::U32(EXPERTS as u32)));
        md.push((k("expert_used_count"), Value::U32(2)));
        md.push((k("expert_feed_forward_length"), Value::U32(EXPERT_FFN as u32)));
        md.push((k("expert_shared_feed_forward_length"), Value::U32(EXPERT_FFN as u32)));
    }
    md.extend(tokenizer_metadata());
    // llama.cpp reads the element type of the merges array, which an empty array does not carry.
    for (key, value) in &mut md {
        if key == "tokenizer.ggml.merges" {
            *value = Value::Array(vec![Value::String("a b".into())]);
        }
    }
    let (f, q) = (GgmlDType::F32, GgmlDType::Q8_0);
    let mut r = Lcg(42);
    let mut tensors: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), r.tensor(&[VOCAB, EMBD], f, 0.5, 0.0)),
        ("output_norm.weight".into(), r.tensor(&[EMBD], f, 0.1, 1.0)),
    ];
    for i in 0..LAYERS {
        let p = |s: &str| format!("blk.{i}.{s}");
        tensors.push((p("attn_norm.weight"), r.tensor(&[EMBD], f, 0.1, 1.0)));
        tensors.push((p("post_attention_norm.weight"), r.tensor(&[EMBD], f, 0.1, 1.0)));
        if (i + 1) % 4 == 0 {
            tensors.push((p("attn_q.weight"), r.tensor(&[HEADS * HEAD_DIM * 2, EMBD], f, 0.2, 0.0)));
            tensors.push((p("attn_k.weight"), r.tensor(&[KV_HEADS * HEAD_DIM, EMBD], f, 0.2, 0.0)));
            tensors.push((p("attn_v.weight"), r.tensor(&[KV_HEADS * HEAD_DIM, EMBD], f, 0.2, 0.0)));
            tensors.push((p("attn_output.weight"), r.tensor(&[EMBD, HEADS * HEAD_DIM], f, 0.2, 0.0)));
            tensors.push((p("attn_q_norm.weight"), r.tensor(&[HEAD_DIM], f, 0.1, 1.0)));
            tensors.push((p("attn_k_norm.weight"), r.tensor(&[HEAD_DIM], f, 0.1, 1.0)));
        } else {
            tensors.push((p("attn_qkv.weight"), r.tensor(&[2 * key + value, EMBD], f, 0.3, 0.0)));
            tensors.push((p("attn_gate.weight"), r.tensor(&[value, EMBD], f, 0.3, 0.0)));
            tensors.push((p("ssm_conv1d.weight"), r.tensor(&[2 * key + value, KERNEL], f, 0.5, 0.0)));
            tensors.push((p("ssm_dt.bias"), r.tensor(&[V_HEADS], f, 0.5, 0.0)));
            tensors.push((p("ssm_a"), r.tensor(&[V_HEADS], f, 0.3, -0.5)));
            tensors.push((p("ssm_beta.weight"), r.tensor(&[V_HEADS, EMBD], f, 0.3, 0.0)));
            tensors.push((p("ssm_alpha.weight"), r.tensor(&[V_HEADS, EMBD], f, 0.3, 0.0)));
            tensors.push((p("ssm_norm.weight"), r.tensor(&[STATE], f, 0.1, 1.0)));
            tensors.push((p("ssm_out.weight"), r.tensor(&[EMBD, value], f, 0.2, 0.0)));
        }
        if moe {
            tensors.push((p("ffn_gate_inp.weight"), r.tensor(&[EXPERTS, EMBD], f, 0.5, 0.0)));
            tensors.push((p("ffn_gate_exps.weight"), r.tensor(&[EXPERTS, EXPERT_FFN, EMBD], q, 0.2, 0.0)));
            tensors.push((p("ffn_up_exps.weight"), r.tensor(&[EXPERTS, EXPERT_FFN, EMBD], q, 0.2, 0.0)));
            tensors.push((p("ffn_down_exps.weight"), r.tensor(&[EXPERTS, EMBD, EXPERT_FFN], q, 0.2, 0.0)));
            tensors.push((p("ffn_gate_inp_shexp.weight"), r.tensor(&[EMBD], f, 0.3, 0.0)));
            tensors.push((p("ffn_gate_shexp.weight"), r.tensor(&[EXPERT_FFN, EMBD], f, 0.2, 0.0)));
            tensors.push((p("ffn_up_shexp.weight"), r.tensor(&[EXPERT_FFN, EMBD], f, 0.2, 0.0)));
            tensors.push((p("ffn_down_shexp.weight"), r.tensor(&[EMBD, EXPERT_FFN], f, 0.2, 0.0)));
        } else {
            tensors.push((p("ffn_gate.weight"), r.tensor(&[FFN, EMBD], f, 0.2, 0.0)));
            tensors.push((p("ffn_up.weight"), r.tensor(&[FFN, EMBD], f, 0.2, 0.0)));
            tensors.push((p("ffn_down.weight"), r.tensor(&[EMBD, FFN], f, 0.2, 0.0)));
        }
    }
    let md_ref: Vec<(&str, &Value)> = md.iter().map(|(a, b)| (a.as_str(), b)).collect();
    let t_ref: Vec<(&str, &QTensor)> = tensors.iter().map(|(a, b)| (a.as_str(), b)).collect();
    gguf_file::write(&mut File::create(path).unwrap(), &md_ref, &t_ref).unwrap();
}
