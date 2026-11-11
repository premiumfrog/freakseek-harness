//! Generates the Tauri context and the permissions of the shell's own commands.

fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(&["pick_directory", "pick_gguf"])),
    )
    .expect("tauri-build failed");
}
