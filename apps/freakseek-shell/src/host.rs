//! Supervisor of the bundled FreakSeek runtime (`desktop-host` in stdio control mode).
//!
//! The runtime writes control events as stdout lines prefixed with `@freakseek:` and
//! reads JSON commands from stdin. Every other stdout and stderr line goes to
//! `data/logs/runtime.log`.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::layout::Layout;

/// Prefix of a control event line, shared with `apps/desktop-host/src/control-channel.ts`.
pub const EVENT_PREFIX: &str = "@freakseek:";

/// Control events the shell acts on.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum HostEvent {
    Starting,
    Ready { url: String },
    Fatal { message: String, diagnostic: Option<String> },
    ShutdownComplete,
    /// Events the shell does not act on (diagnostics, sessions).
    #[serde(other)]
    Other,
}

/// What the supervisor reports to the window.
#[derive(Debug, Clone, PartialEq)]
pub enum Notice {
    Event(HostEvent),
    /// The runtime closed its stdout (it exited); `tail` holds its last log lines.
    Exited { tail: String },
}

/// Parse one stdout line into a control event, if it is one.
/// @param line - a raw stdout line.
pub fn parse_event(line: &str) -> Option<HostEvent> {
    serde_json::from_str(line.trim_end().strip_prefix(EVENT_PREFIX)?).ok()
}

/// The running runtime process.
pub struct Host {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
}

impl Host {
    /// Start the runtime and forward its notices.
    /// @param layout - portable folder paths.
    /// @param notify - receives every control event and the final exit, on reader threads.
    pub fn spawn(layout: &Layout, notify: impl Fn(Notice) + Send + Sync + 'static) -> std::io::Result<Self> {
        let log = Arc::new(Mutex::new(
            OpenOptions::new().create(true).append(true).open(layout.logs.join("runtime.log"))?,
        ));
        let mut command = Command::new(&layout.node);
        command
            .arg("--expose-internals")
            .arg(&layout.host_entry)
            .arg(&layout.runtime_app)
            .arg(&layout.profile)
            .current_dir(&layout.root)
            .env("FREAKSEEK_SHELL", "stdio")
            .env("DSH_HOME", &layout.data)
            // Agent definitions and memories default to the user profile; keep them in the portable folder.
            .env("DSH_AGENTS_HOME", layout.data.join("agents"))
            .env("DSH_LOCAL_MODELS_ROOT", &layout.models)
            .env("DSH_GGUF_ENGINE_DIR", &layout.engine)
            .env("DSH_TELEMETRY_DISABLED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().ok_or_else(|| std::io::Error::other("runtime stdout unavailable"))?;
        let stderr = child.stderr.take().ok_or_else(|| std::io::Error::other("runtime stderr unavailable"))?;
        let tail = Arc::new(Mutex::new(Vec::<String>::new()));
        let notify = Arc::new(notify);

        {
            let (log, tail) = (log.clone(), tail.clone());
            thread::spawn(move || copy_lines(BufReader::new(stderr), &log, &tail));
        }
        let host = Self { child: Mutex::new(child), stdin: Mutex::new(stdin) };
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                match parse_event(&line) {
                    Some(event) => notify(Notice::Event(event)),
                    None => record(&log, &tail, &line),
                }
            }
            // stdout closes when the runtime exits.
            let tail = tail.lock().map(|lines| lines.join("\n")).unwrap_or_default();
            notify(Notice::Exited { tail });
        });
        Ok(host)
    }

    /// The exit code once the process has exited, waiting up to `wait` for it.
    /// @param wait - how long to wait for the process to finish exiting.
    pub fn exit_code(&self, wait: Duration) -> Option<i32> {
        let deadline = Instant::now() + wait;
        loop {
            let status = self.child.lock().ok()?.try_wait().ok()?;
            if let Some(status) = status {
                return status.code();
            }
            if Instant::now() >= deadline {
                return None;
            }
            thread::sleep(Duration::from_millis(25));
        }
    }

    /// Send one JSON command line.
    /// @param command - a serializable command such as `{"type":"shutdown"}`.
    pub fn command(&self, command: &serde_json::Value) {
        if let Ok(mut stdin) = self.stdin.lock() {
            if let Some(pipe) = stdin.as_mut() {
                // A closed pipe means the runtime already exited; shutdown handles that.
                let _ = writeln!(pipe, "{command}").and_then(|()| pipe.flush());
            }
        }
    }

    /// Ask the runtime to stop, then wait up to `grace` before killing it.
    /// @param grace - time the runtime gets to save state and exit.
    pub fn shutdown(&self, grace: Duration) {
        self.command(&serde_json::json!({ "type": "shutdown" }));
        let deadline = Instant::now() + grace;
        loop {
            let Ok(mut child) = self.child.lock() else { return };
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {}
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
            }
            drop(child);
            thread::sleep(Duration::from_millis(50));
        }
        // Closing stdin also ends the runtime's control channel if it is still reading.
        if let Ok(mut stdin) = self.stdin.lock() {
            stdin.take();
        }
    }
}

fn record(log: &Mutex<File>, tail: &Mutex<Vec<String>>, line: &str) {
    if let Ok(mut file) = log.lock() {
        let _ = writeln!(file, "{line}");
    }
    if let Ok(mut lines) = tail.lock() {
        lines.push(line.to_string());
        let excess = lines.len().saturating_sub(80);
        lines.drain(..excess);
    }
}

fn copy_lines(reader: impl BufRead, log: &Mutex<File>, tail: &Mutex<Vec<String>>) {
    for line in reader.lines().map_while(Result::ok) {
        record(log, tail, &line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prefixed_control_lines_only() {
        assert_eq!(
            parse_event("@freakseek:{\"type\":\"ready\",\"url\":\"http://127.0.0.1:5/?token=t\",\"injections\":[]}\r"),
            Some(HostEvent::Ready { url: "http://127.0.0.1:5/?token=t".into() }),
        );
        assert_eq!(
            parse_event("@freakseek:{\"type\":\"fatal\",\"message\":\"boom\"}"),
            Some(HostEvent::Fatal { message: "boom".into(), diagnostic: None }),
        );
        assert_eq!(parse_event("@freakseek:{\"type\":\"plugin-operation\",\"record\":{}}"), Some(HostEvent::Other));
        assert_eq!(parse_event("dsh web: listening"), None);
        assert_eq!(parse_event("@freakseek:not json"), None);
    }
}
