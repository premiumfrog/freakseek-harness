//! The JSON-lines protocol of the built `dsh-gguf-engine` binary.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_dsh-gguf-engine"))
            .args(["--threads", "2"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self { child, stdin, stdout }
    }

    fn send(&mut self, v: Value) {
        writeln!(self.stdin, "{v}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn next(&mut self) -> Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    /// Read lines until the result for `id`, returning every line for that id.
    fn until_result(&mut self, id: &str) -> Vec<Value> {
        let mut seen = Vec::new();
        loop {
            let v = self.next();
            if v["id"] != id {
                continue;
            }
            let done = v["type"] == "result";
            seen.push(v);
            if done {
                return seen;
            }
        }
    }
}

#[test]
fn load_generate_reuse_cache_and_shutdown() {
    let path = common::write_model("qwen2", false, true);
    let mut e = Engine::spawn();
    e.send(json!({"id": "h", "op": "hello"}));
    assert_eq!(e.until_result("h")[0]["value"]["protocol"], 1);

    e.send(json!({"id": "i", "op": "inspect", "path": path}));
    let facts = &e.until_result("i")[0]["value"];
    assert_eq!(facts["architecture"], "qwen2");
    assert_eq!(facts["supported"], true);
    assert_eq!(facts["name"], "Synthetic test model");

    e.send(json!({"id": "g0", "op": "generate", "messages": []}));
    assert_eq!(e.until_result("g0")[0]["error"]["code"], "NO_MODEL");

    e.send(json!({"id": "l", "op": "load", "path": path, "contextLength": 512}));
    let loaded = e.until_result("l");
    assert_eq!(loaded[0]["ok"], true, "{loaded:?}");
    assert_eq!(loaded[0]["value"]["contextLength"], 512);

    let request = |id: &str| json!({
        "id": id, "op": "generate", "maxTokens": 6,
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}],
        "sampling": {"temperature": 0.0, "repeatPenalty": 1.0},
    });
    e.send(request("g1"));
    let first = e.until_result("g1");
    assert_eq!(first[0]["type"], "start");
    assert_eq!(first[0]["cachedTokens"], 0);
    let result = first.last().unwrap();
    assert_eq!(result["ok"], true, "{result:?}");
    let reason = result["value"]["finishReason"].as_str().unwrap();
    assert!(reason == "length" || reason == "stop");
    let text1: String = first.iter().filter(|v| v["type"] == "delta").map(|v| v["text"].as_str().unwrap()).collect();

    // The same conversation reuses every prompt position but the last and reproduces greedy output.
    e.send(request("g2"));
    let second = e.until_result("g2");
    let prompt = second[0]["promptTokens"].as_u64().unwrap();
    assert_eq!(second[0]["cachedTokens"].as_u64().unwrap(), prompt - 1);
    let text2: String = second.iter().filter(|v| v["type"] == "delta").map(|v| v["text"].as_str().unwrap()).collect();
    assert_eq!(text1, text2);

    e.send(json!({"id": "s", "op": "status"}));
    assert_eq!(e.until_result("s")[0]["value"]["loaded"], true);

    e.send(json!({"id": "x", "op": "nonsense"}));
    assert_eq!(e.until_result("x")[0]["error"]["code"], "BAD_REQUEST");

    e.send(json!({"id": "u", "op": "unload"}));
    assert_eq!(e.until_result("u")[0]["ok"], true);
    e.send(json!({"id": "q", "op": "shutdown"}));
    assert_eq!(e.until_result("q")[0]["ok"], true);
    assert!(e.child.wait().unwrap().success());
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn cancel_stops_a_running_generation() {
    let path = common::write_model("llama", false, true);
    let mut e = Engine::spawn();
    e.send(json!({"id": "l", "op": "load", "path": path, "contextLength": 512}));
    assert_eq!(e.until_result("l")[0]["ok"], true);
    e.send(json!({"id": "g", "op": "generate", "maxTokens": 100000, "messages": [{"role": "user", "content": "go"}],
        "sampling": {"temperature": 0.0, "repeatPenalty": 1.0}}));
    assert_eq!(e.next()["type"], "start");
    e.send(json!({"id": "c", "op": "cancel", "target": "g"}));
    let lines = e.until_result("g");
    let reason = lines.last().unwrap()["value"]["finishReason"].as_str().unwrap().to_string();
    // A random model may also hit its context window or EOS before the cancel lands.
    assert!(["cancelled", "length", "stop"].contains(&reason.as_str()), "{reason}");
    drop(e.stdin);
    assert!(e.child.wait().unwrap().success());
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn slots_keep_prefixes_across_interleaved_requests() {
    let path = common::write_model("qwen2", false, true);
    let mut e = Engine::spawn();
    e.send(json!({"id": "l", "op": "load", "path": path, "contextLength": 512, "slots": 2}));
    assert_eq!(e.until_result("l")[0]["ok"], true);
    let gen = |id: &str, slot: &str, text: &str| json!({
        "id": id, "op": "generate", "maxTokens": 2, "slot": slot,
        "messages": [{"role": "user", "content": text}],
        "sampling": {"temperature": 0.0, "repeatPenalty": 1.0},
    });
    e.send(gen("a1", "conversation", "a long conversation prompt"));
    let prompt = e.until_result("a1")[0]["promptTokens"].as_u64().unwrap();
    e.send(gen("b1", "title", "something else entirely"));
    assert_eq!(e.until_result("b1")[0]["cachedTokens"], 0);
    e.send(gen("a2", "conversation", "a long conversation prompt"));
    assert_eq!(e.until_result("a2")[0]["cachedTokens"].as_u64().unwrap(), prompt - 1);
    // A third slot evicts the least recently used one ("title").
    e.send(gen("c1", "other", "third"));
    e.until_result("c1");
    e.send(gen("b2", "title", "something else entirely"));
    assert_eq!(e.until_result("b2")[0]["cachedTokens"], 0);
    drop(e.stdin);
    assert!(e.child.wait().unwrap().success());
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}
