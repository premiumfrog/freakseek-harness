//! Prefill and decode throughput on a Qwen2.5-0.5B-shaped Q4_0/Q8_0 model with random weights.
//!
//! Ignored by default; run with `cargo test --release --test throughput -- --ignored --nocapture`.

use std::fs::File;
use std::time::Instant;

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use dsh_gguf_engine::model::Model;

fn q(shape: &[usize], dtype: GgmlDType) -> QTensor {
    let t = (Tensor::randn(0f32, 1f32, shape, &Device::Cpu).unwrap() * 0.02).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

#[test]
#[ignore = "benchmark"]
fn qwen2_half_billion_shape() {
    let (embd, heads, kv, hd, ffn, layers, vocab) = (896usize, 14usize, 2usize, 64usize, 4864usize, 24usize, 32768usize);
    let path = std::env::temp_dir().join(format!("dsh-gguf-throughput-{}.gguf", std::process::id()));
    let a = "qwen2";
    let md: Vec<(String, Value)> = vec![
        ("general.architecture".into(), Value::String(a.into())),
        (format!("{a}.block_count"), Value::U32(layers as u32)),
        (format!("{a}.embedding_length"), Value::U32(embd as u32)),
        (format!("{a}.context_length"), Value::U32(4096)),
        (format!("{a}.attention.head_count"), Value::U32(heads as u32)),
        (format!("{a}.attention.head_count_kv"), Value::U32(kv as u32)),
        (format!("{a}.rope.freq_base"), Value::F32(1_000_000.0)),
        (format!("{a}.attention.layer_norm_rms_epsilon"), Value::F32(1e-6)),
    ];
    let k = GgmlDType::Q4_0;
    let ones = |n: usize| QTensor::quantize(&Tensor::ones(n, candle_core::DType::F32, &Device::Cpu).unwrap(), GgmlDType::F32).unwrap();
    let mut t: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), q(&[vocab, embd], GgmlDType::Q8_0)),
        ("output_norm.weight".into(), ones(embd)),
    ];
    for i in 0..layers {
        let p = |s: &str| format!("blk.{i}.{s}");
        t.push((p("attn_norm.weight"), ones(embd)));
        t.push((p("attn_q.weight"), q(&[heads * hd, embd], GgmlDType::Q8_0)));
        t.push((p("attn_k.weight"), q(&[kv * hd, embd], GgmlDType::Q8_0)));
        t.push((p("attn_v.weight"), q(&[kv * hd, embd], GgmlDType::Q8_0)));
        t.push((p("attn_q.bias"), ones(heads * hd)));
        t.push((p("attn_k.bias"), ones(kv * hd)));
        t.push((p("attn_v.bias"), ones(kv * hd)));
        t.push((p("attn_output.weight"), q(&[embd, heads * hd], GgmlDType::Q8_0)));
        t.push((p("ffn_norm.weight"), ones(embd)));
        t.push((p("ffn_gate.weight"), q(&[ffn, embd], k)));
        t.push((p("ffn_up.weight"), q(&[ffn, embd], k)));
        t.push((p("ffn_down.weight"), q(&[embd, ffn], GgmlDType::Q8_0)));
    }
    let mr: Vec<(&str, &Value)> = md.iter().map(|(a, b)| (a.as_str(), b)).collect();
    let tr: Vec<(&str, &QTensor)> = t.iter().map(|(a, b)| (a.as_str(), b)).collect();
    gguf_file::write(&mut File::create(&path).unwrap(), &mr, &tr).unwrap();
    let content = gguf_file::Content::read(&mut File::open(&path).unwrap()).unwrap();
    let mut model = Model::load(&path, &content).unwrap();
    let prompt: Vec<u32> = (0..512).map(|i| (i * 37 % vocab) as u32).collect();
    let start = Instant::now();
    for chunk in prompt.chunks(256) {
        model.forward(chunk).unwrap();
    }
    let prefill = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let steps = 64;
    for i in 0..steps {
        model.forward(&[(i * 11 % vocab) as u32]).unwrap();
    }
    let decode = start.elapsed().as_secs_f64();
    println!("prefill: {:.1} tokens/s, decode: {:.1} tokens/s", prompt.len() as f64 / prefill, steps as f64 / decode);
    std::fs::remove_file(&path).ok();
}
