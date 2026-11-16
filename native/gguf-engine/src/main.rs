//! `dsh-gguf-engine`: offline GGUF inference driven by JSON lines on stdin/stdout.
//!
//! The process opens no sockets and makes no network requests. Every request is
//! one JSON object per stdin line carrying a caller-chosen `id`; every reply is
//! one JSON object per stdout line echoing it. See `PROTOCOL.md`.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::{json, Value as Json};

use dsh_gguf_engine::device::{self, Placement};
use dsh_gguf_engine::engine::{self, Engine, EngineError, Event, Request};
use dsh_gguf_engine::sampler::SamplingParams;

/// Wire protocol revision; the host refuses an engine reporting another value.
const PROTOCOL_VERSION: u32 = 1;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum Command {
    Hello,
    Devices,
    Inspect { path: PathBuf },
    #[serde(rename_all = "camelCase")]
    Load {
        path: PathBuf,
        context_length: Option<usize>,
        slots: Option<usize>,
        device: Option<String>,
        gpu_layers: Option<usize>,
        prompt_cache_dir: Option<PathBuf>,
    },
    Unload,
    Status,
    #[serde(rename_all = "camelCase")]
    Generate {
        messages: Json,
        #[serde(default)]
        tools: Json,
        max_tokens: Option<usize>,
        #[serde(default)]
        stop: Vec<String>,
        #[serde(default)]
        sampling: SamplingParams,
        thinking: Option<bool>,
        #[serde(default)]
        slot: String,
        tool_grammar: Option<bool>,
    },
    Cancel { target: String },
    Shutdown,
}

#[derive(Deserialize)]
struct Envelope {
    id: String,
    #[serde(flatten)]
    command: Json,
}

/// Serialized stdout writer shared by the reader and worker threads.
#[derive(Clone)]
struct Out(Arc<Mutex<std::io::Stdout>>);

impl Out {
    fn send(&self, v: Json) {
        let mut o = self.0.lock().unwrap_or_else(|e| e.into_inner());
        // A closed stdout means the host is gone; the reader loop ends on stdin EOF.
        let _ = writeln!(o, "{v}");
        let _ = o.flush();
    }
    fn ok(&self, id: &str, value: Json) {
        self.send(json!({ "id": id, "type": "result", "ok": true, "value": value }));
    }
    fn fail(&self, id: &str, err: &anyhow::Error) {
        let (code, message) = match err.downcast_ref::<EngineError>() {
            Some(e) => (e.code, e.message.clone()),
            None => ("ENGINE_ERROR", format!("{err:#}")),
        };
        self.send(json!({ "id": id, "type": "result", "ok": false, "error": { "code": code, "message": message } }));
    }
}

struct Job {
    id: String,
    command: Command,
    cancel: Arc<AtomicBool>,
}

fn worker(rx: mpsc::Receiver<Job>, out: Out) {
    let mut engine: Option<Engine> = None;
    for job in rx {
        let id = job.id.as_str();
        match job.command {
            Command::Load { path, context_length, slots, device, gpu_layers, prompt_cache_dir } => {
                engine = None;
                let started = std::time::Instant::now();
                let progress_out = out.clone();
                let progress_id = id.to_string();
                let mut last = std::time::Instant::now();
                let mut progress = move |done: usize, total: usize| {
                    // At most ten updates a second, and always the last layer.
                    if done == total || last.elapsed().as_millis() >= 100 {
                        last = std::time::Instant::now();
                        progress_out.send(json!({ "id": progress_id, "type": "progress", "stage": "load", "done": done, "total": total }));
                    }
                };
                let mut load = |spec: Option<&str>| {
                    Placement::resolve(spec, gpu_layers).and_then(|p| Engine::load_with_progress(&path, context_length, slots.unwrap_or(4), &p, &mut progress))
                };
                let mut warning = None;
                let mut loaded = load(device.as_deref());
                // With no device preference, a GPU that computes wrong results gives way to the CPU.
                if let Err(e) = &loaded {
                    let mismatch = e.downcast_ref::<EngineError>().is_some_and(|e| e.code == "GPU_KERNEL_MISMATCH");
                    if mismatch && matches!(device.as_deref(), None | Some("auto")) {
                        warning = Some(e.to_string());
                        loaded = load(Some("cpu"));
                    }
                }
                match loaded {
                    Ok(mut e) => {
                        if let Some(dir) = &prompt_cache_dir {
                            e.set_prompt_cache(dir);
                        }
                        let mut result = placement_facts(&e);
                        result["facts"] = json!(e.facts);
                        result["vocab"] = json!(e.vocab());
                        result["loadMs"] = json!(started.elapsed().as_millis() as u64);
                        if let Some(w) = warning {
                            result["warning"] = json!(w);
                        }
                        out.ok(id, result);
                        engine = Some(e);
                    }
                    Err(e) => out.fail(id, &e),
                }
            }
            Command::Unload => {
                engine = None;
                out.ok(id, Json::Null);
            }
            Command::Status => out.ok(
                id,
                match &engine {
                    Some(e) => {
                        let mut result = placement_facts(e);
                        result["loaded"] = json!(true);
                        result["path"] = json!(e.facts.path);
                        result["cachedTokens"] = json!(e.cached_tokens());
                        result
                    }
                    None => json!({ "loaded": false }),
                },
            ),
            Command::Generate { messages, tools, max_tokens, stop, sampling, thinking, slot, tool_grammar } => {
                let Some(e) = engine.as_mut() else {
                    out.fail(id, &EngineError { code: "NO_MODEL", message: "no model is loaded".into() }.into());
                    continue;
                };
                let req = Request { messages: &messages, tools: &tools, max_tokens: max_tokens.unwrap_or(usize::MAX), stop, sampling, thinking, slot, tool_grammar: tool_grammar.unwrap_or(true) };
                let mut emit = |ev: Event| match ev {
                    Event::Start { prompt_tokens, cached_tokens, opens_reasoning } => out.send(json!({
                        "id": id, "type": "start", "promptTokens": prompt_tokens, "cachedTokens": cached_tokens, "opensReasoning": opens_reasoning,
                    })),
                    Event::Delta(text) => out.send(json!({ "id": id, "type": "delta", "text": text })),
                    Event::Prefill { done, total } => out.send(json!({ "id": id, "type": "progress", "stage": "prefill", "done": done, "total": total })),
                    Event::Decode { tokens, tokens_per_second, context_used } => out.send(json!({
                        "id": id, "type": "progress", "stage": "decode", "done": tokens, "tokensPerSecond": tokens_per_second, "contextUsed": context_used,
                    })),
                };
                match e.generate(req, &job.cancel, &mut emit) {
                    Ok(o) => out.ok(id, json!(o)),
                    Err(err) => out.fail(id, &err),
                }
            }
            Command::Hello | Command::Devices | Command::Inspect { .. } | Command::Cancel { .. } | Command::Shutdown => unreachable!("handled by the reader"),
        }
    }
}

/// Where a loaded model runs: device, offloaded layers and experts, and the device's memory.
fn placement_facts(e: &Engine) -> Json {
    let (gpu_experts, experts) = e.offloaded_experts();
    let label = e.device_label();
    let memory = label.strip_prefix("cuda:").and_then(|n| n.parse::<usize>().ok()).and_then(|ordinal| {
        dsh_gguf_engine::device::devices().ok()?.into_iter().find(|d| d.ordinal == ordinal)
    });
    json!({
        "contextLength": e.context_length, "device": label, "gpuLayers": e.offloaded_layers(),
        "gpuExperts": gpu_experts, "experts": experts,
        "vramTotalBytes": memory.as_ref().map(|m| m.total_bytes), "vramFreeBytes": memory.as_ref().map(|m| m.free_bytes),
    })
}

/// Run the same tokens through a host copy and an accelerator copy of a model and print where they diverge.
/// @param path - GGUF file.
/// @returns the process exit code: 0 when both agree, 1 when they diverge or the accelerator is unavailable.
fn selftest(path: &std::path::Path) -> i32 {
    use dsh_gguf_engine::model::Model;
    let tokens: Vec<u32> = (1..=24).collect();
    let run = |placement: &Placement| -> anyhow::Result<(Vec<f32>, Vec<f32>, String)> {
        let content = candle_core::quantized::gguf_file::Content::read(&mut std::fs::File::open(path)?)?;
        let mut model = Model::load(path, &content, placement)?;
        model.trace = Some(Vec::new());
        let mut logits = model.forward(&tokens)?;
        for t in [5u32, 9, 13] {
            logits = model.forward(&[t])?;
        }
        let (gpu, all) = model.offloaded_experts();
        let label = format!("{} ({} layers, {gpu}/{all} experts offloaded)", model.device_label(), model.offloaded_layers());
        Ok((model.trace.take().unwrap_or_default(), logits, label))
    };
    let top = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |p| p.0);
    println!("selftest: evaluating on the CPU");
    let host = match run(&Placement::cpu()) {
        Ok(r) => r,
        Err(e) => {
            println!("selftest: CPU run failed: {e:#}");
            return 1;
        }
    };
    println!("selftest: CPU top token {}, finite logits: {}", top(&host.1), host.1.iter().all(|l| l.is_finite()));
    let accel = match Placement::resolve(Some("auto"), None).and_then(|p| if p.device.is_cpu() { anyhow::bail!("no supported GPU") } else { run(&p) }) {
        Ok(r) => r,
        Err(e) => {
            println!("selftest: GPU run unavailable: {e:#}");
            return 1;
        }
    };
    println!("selftest: evaluating on {}", accel.2);
    let layers = host.0.len() / 4;
    let mut first_bad = None;
    for (i, (a, b)) in host.0.iter().zip(&accel.0).enumerate() {
        let error = (a - b).abs() / a.abs().max(1e-6);
        if !(error <= 0.05) && first_bad.is_none() {
            first_bad = Some((i / layers.max(1), i % layers.max(1), *a, *b));
        }
    }
    match first_bad {
        Some((step, layer, a, b)) => {
            println!("selftest: FAILED at step {step}, layer {layer}: hidden-state norm {a} on the CPU, {b} on the GPU");
            1
        }
        None if top(&host.1) != top(&accel.1) => {
            println!("selftest: FAILED: top token {} on the CPU, {} on the GPU", top(&host.1), top(&accel.1));
            1
        }
        None => {
            println!("selftest: passed; CPU and GPU agree on every layer");
            0
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => {
                if let Some(n) = args.next() {
                    std::env::set_var("RAYON_NUM_THREADS", n);
                }
            }
            "--selftest" => {
                let Some(path) = args.next() else {
                    eprintln!("dsh-gguf-engine: --selftest needs a model path");
                    std::process::exit(2);
                };
                std::process::exit(selftest(std::path::Path::new(&path)));
            }
            "--version" => {
                println!("dsh-gguf-engine {} protocol {PROTOCOL_VERSION}", env!("CARGO_PKG_VERSION"));
                return;
            }
            other => {
                eprintln!("dsh-gguf-engine: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let out = Out(Arc::new(Mutex::new(std::io::stdout())));
    let (tx, rx) = mpsc::channel::<Job>();
    let worker_out = out.clone();
    let handle = std::thread::spawn(move || worker(rx, worker_out));
    // Cancellation flags of queued and running generations, by request id.
    let flags: Arc<Mutex<Vec<(String, Arc<AtomicBool>)>>> = Arc::default();
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let env: Envelope = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                out.send(json!({ "id": null, "type": "result", "ok": false, "error": { "code": "BAD_REQUEST", "message": e.to_string() } }));
                continue;
            }
        };
        let command: Command = match serde_json::from_value(env.command) {
            Ok(c) => c,
            Err(e) => {
                out.send(json!({ "id": env.id, "type": "result", "ok": false, "error": { "code": "BAD_REQUEST", "message": e.to_string() } }));
                continue;
            }
        };
        match command {
            Command::Hello => out.ok(&env.id, json!({ "protocol": PROTOCOL_VERSION, "version": env!("CARGO_PKG_VERSION"), "cuda": device::cuda_compiled() })),
            // Device queries touch no model and answer even while a generation runs.
            Command::Devices => match device::devices() {
                Ok(list) => out.ok(&env.id, json!({
                    "cuda": device::cuda_compiled(), "minComputeCapability": device::min_compute_capability(), "devices": list,
                })),
                Err(e) => out.fail(&env.id, &e),
            },
            // Header reads are cheap and must not wait behind a running generation.
            Command::Inspect { path } => match engine::inspect(&path) {
                Ok((_, facts)) => out.ok(&env.id, json!(facts)),
                Err(e) => out.fail(&env.id, &e),
            },
            Command::Cancel { target } => {
                let mut f = flags.lock().unwrap_or_else(|e| e.into_inner());
                f.retain(|(_, flag)| Arc::strong_count(flag) > 1);
                for (id, flag) in f.iter() {
                    if *id == target {
                        flag.store(true, Ordering::Relaxed);
                    }
                }
                out.ok(&env.id, Json::Null);
            }
            Command::Shutdown => {
                out.ok(&env.id, Json::Null);
                break;
            }
            command => {
                let cancel = Arc::new(AtomicBool::new(false));
                flags.lock().unwrap_or_else(|e| e.into_inner()).push((env.id.clone(), cancel.clone()));
                if tx.send(Job { id: env.id, command, cancel }).is_err() {
                    break;
                }
            }
        }
    }
    // Stop the running generation so shutdown does not wait for it.
    for (_, flag) in flags.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        flag.store(true, Ordering::Relaxed);
    }
    drop(tx);
    let _ = handle.join();
}
