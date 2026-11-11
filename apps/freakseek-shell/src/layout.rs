//! The portable folder layout the shell runs from.
//!
//! ```text
//! FreakSeek/
//!   FreakSeek(.exe)        this shell
//!   runtime/node(.exe)     bundled Node.js
//!   runtime/app/           deployed desktop-host package tree (lib/index.js, node_modules/)
//!   engine/                GGUF engines (cpu/, cuda/)
//!   models/                model collections (one subfolder each)
//!   data/                  conversations, settings, profile, logs (DSH_HOME)
//! ```

use std::path::{Path, PathBuf};

/// Every path the shell and the runtime it starts read.
#[derive(Debug, Clone)]
pub struct Layout {
    pub root: PathBuf,
    pub node: PathBuf,
    pub runtime_app: PathBuf,
    pub host_entry: PathBuf,
    pub engine: PathBuf,
    pub models: PathBuf,
    pub data: PathBuf,
    pub profile: PathBuf,
    pub logs: PathBuf,
}

impl Layout {
    /// Resolve the layout under `root`.
    /// @param root - the portable folder.
    pub fn under(root: &Path) -> Self {
        let runtime = root.join("runtime");
        let bundled = runtime.join(if cfg!(windows) { "node.exe" } else { "node" });
        // A development tree without a bundled Node uses the one on PATH.
        let node = if bundled.is_file() { bundled } else { PathBuf::from("node") };
        let runtime_app = runtime.join("app");
        let data = root.join("data");
        Self {
            root: root.to_path_buf(),
            node,
            host_entry: runtime_app.join("lib").join("index.js"),
            runtime_app,
            engine: root.join("engine"),
            models: root.join("models"),
            profile: data.join("profile"),
            logs: data.join("logs"),
            data,
        }
    }

    /// The folder holding the running executable, or `FREAKSEEK_ROOT` when set.
    pub fn locate() -> std::io::Result<Self> {
        if let Some(root) = std::env::var_os("FREAKSEEK_ROOT") {
            return Ok(Self::under(Path::new(&root)));
        }
        let exe = std::env::current_exe()?;
        let root = exe.parent().ok_or_else(|| std::io::Error::other("executable has no parent folder"))?;
        Ok(Self::under(root))
    }

    /// Create the writable folders the runtime expects.
    pub fn prepare(&self) -> std::io::Result<()> {
        for dir in [&self.models, &self.data, &self.logs] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_every_path_under_the_folder() {
        let root = Path::new("/portable/FreakSeek");
        let layout = Layout::under(root);
        assert_eq!(layout.host_entry, root.join("runtime/app/lib/index.js"));
        assert_eq!(layout.models, root.join("models"));
        assert_eq!(layout.profile, root.join("data/profile"));
        // No bundled Node in a nonexistent folder: PATH lookup.
        assert_eq!(layout.node, PathBuf::from("node"));
    }
}
