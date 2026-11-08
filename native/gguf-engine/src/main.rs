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
    Inspect { path: PathBuf },
    #[serde(rename_all = "camelCase")]
    Load { path: PathBuf, context_length: Option<usize>, slots: Option<usize>, device: Option<String>, gpu_layers: Option<usize> },
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
            Command::Load { path, context_length, slots, device, gpu_layers } => {
                engine = None;
                let loaded = Placement::resolve(device.as_deref(), gpu_layers).and_then(|p| Engine::load(&path, context_length, slots.unwrap_or(4), &p));
                match loaded {
                    Ok(e) => {
                        out.ok(id, json!({
                            "facts": e.facts, "contextLength": e.context_length, "vocab": e.vocab(),
                            "device": e.device_label(), "gpuLayers": e.offloaded_layers(),
                        }));
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
                    Some(e) => json!({
                        "loaded": true, "path": e.facts.path, "contextLength": e.context_length,
                        "device": e.device_label(), "gpuLayers": e.offloaded_layers(),
                    }),
                    None => json!({ "loaded": false }),
                },
            ),
            Command::Generate { messages, tools, max_tokens, stop, sampling, thinking, slot } => {
                let Some(e) = engine.as_mut() else {
                    out.fail(id, &EngineError { code: "NO_MODEL", message: "no model is loaded".into() }.into());
                    continue;
                };
                let req = Request { messages: &messages, tools: &tools, max_tokens: max_tokens.unwrap_or(usize::MAX), stop, sampling, thinking, slot };
                let mut emit = |ev: Event| match ev {
                    Event::Start { prompt_tokens, cached_tokens, opens_reasoning } => out.send(json!({
                        "id": id, "type": "start", "promptTokens": prompt_tokens, "cachedTokens": cached_tokens, "opensReasoning": opens_reasoning,
                    })),
                    Event::Delta(text) => out.send(json!({ "id": id, "type": "delta", "text": text })),
                };
                match e.generate(req, &job.cancel, &mut emit) {
                    Ok(o) => out.ok(id, json!(o)),
                    Err(err) => out.fail(id, &err),
                }
            }
            Command::Hello | Command::Inspect { .. } | Command::Cancel { .. } | Command::Shutdown => unreachable!("handled by the reader"),
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
