//! FreakSeek desktop shell.
//!
//! Starts the bundled runtime from the portable folder and shows a local splash window until the
//! runtime reports its loopback URL. The UI then opens in a new window whose first load is that
//! URL: the launch-token exchange sets a `SameSite=Strict` session cookie, which a browser does not
//! send on a redirect that a page from another site (the splash) initiated. Windows may navigate
//! only to loopback and the shell's own assets, so nothing a page does opens a remote site.
//! Closing the UI window hides it to the tray; Quit stops the runtime first.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod host;
mod layout;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, DragDropEvent, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_dialog::DialogExt;

use host::{Host, HostEvent, Notice};
use layout::Layout;

const MAIN: &str = "main";
const SPLASH: &str = "splash";
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Page-side bridges the FreakSeek UI looks for; installed only on the runtime's loopback page.
const BRIDGES: &str = r#"
(() => {
  if (location.protocol !== 'http:' || location.hostname !== '127.0.0.1') return
  const invoke = (command) => (window.__TAURI__?.core?.invoke ?? window.__TAURI_INTERNALS__.invoke)(command)
  window.__DSH_DIRECTORY_PICKER__ = { pick: () => invoke('pick_directory') }
  window.__DSH_GGUF_PICKER__ = { pick: () => invoke('pick_gguf') }
})()
"#;

/// Shell state shared by window, tray, and runtime callbacks.
#[derive(Default)]
struct Shell {
    host: Mutex<Option<Arc<Host>>>,
    quitting: Mutex<bool>,
}

/// Whether the window may navigate to `url`: the shell's assets and loopback only.
/// @param url - the requested navigation target.
fn navigation_allowed(url: &url::Url) -> bool {
    match url.scheme() {
        "tauri" | "about" | "data" | "blob" => true,
        "http" | "https" => matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]" | "tauri.localhost")),
        _ => false,
    }
}

/// Open a folder in the platform file manager.
/// @param path - folder to show.
fn reveal(path: &std::path::Path) {
    let program = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    // A missing file manager only means the folder is not shown.
    let _ = std::process::Command::new(program).arg(path).spawn();
}

fn show(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(MAIN).or_else(|| app.get_webview_window(SPLASH)) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Stop the runtime and exit the process.
fn quit(app: &AppHandle) {
    let shell = app.state::<Shell>();
    if let Ok(mut quitting) = shell.quitting.lock() {
        if *quitting {
            return;
        }
        *quitting = true;
    }
    let host = shell.host.lock().ok().and_then(|mut host| host.take());
    let app = app.clone();
    std::thread::spawn(move || {
        if let Some(host) = host {
            host.shutdown(SHUTDOWN_GRACE);
        }
        app.exit(0);
    });
}

fn js_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into())
}

fn show_fatal(window: &WebviewWindow, message: &str, diagnostic: &str) {
    let script = format!("window.freakseekFatal({}, {})", js_string(message), js_string(diagnostic));
    // The splash page may have been replaced by the UI; return to it before reporting.
    if window.url().map(|url| url.scheme() == "http" && url.host_str() == Some("127.0.0.1")).unwrap_or(false) {
        if let Ok(splash) = splash_url(window) {
            let _ = window.navigate(splash);
            std::thread::sleep(Duration::from_millis(300));
        }
    }
    let _ = window.eval(&script);
    let _ = window.show();
}

fn splash_url(window: &WebviewWindow) -> tauri::Result<url::Url> {
    let base = if cfg!(windows) { "http://tauri.localhost/index.html" } else { "tauri://localhost/index.html" };
    let _ = window;
    url::Url::parse(base).map_err(|error| tauri::Error::Anyhow(error.into()))
}

/// The UI window when it exists, otherwise the splash window.
fn current_window(app: &AppHandle) -> Option<WebviewWindow> {
    app.get_webview_window(MAIN).or_else(|| app.get_webview_window(SPLASH))
}

/// Open the UI window with `url` as its first load and wire its window events.
/// @param app - application handle.
/// @param url - the runtime's authenticated loopback URL.
fn open_main(app: &AppHandle, url: url::Url) -> tauri::Result<()> {
    let window = WebviewWindowBuilder::new(app, MAIN, WebviewUrl::External(url))
        .title("FreakSeek")
        .inner_size(1280.0, 840.0)
        .min_inner_size(720.0, 480.0)
        .initialization_script(BRIDGES)
        .on_navigation(navigation_allowed)
        .build()?;
    let events = window.clone();
    window.on_window_event(move |event| match event {
        WindowEvent::CloseRequested { api, .. } => {
            // The tray keeps FreakSeek running; Quit stops it.
            api.prevent_close();
            let _ = events.hide();
        }
        WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) => {
            let models = dropped_models(paths);
            if !models.is_empty() {
                let detail = serde_json::json!({ "paths": models });
                let _ = events.eval(&format!("window.dispatchEvent(new CustomEvent('freakseek:gguf-dropped', {{ detail: {detail} }}))"));
            }
        }
        _ => {}
    });
    Ok(())
}

/// Start the runtime and wire its notices to the window.
fn start_runtime(app: &AppHandle, layout: &Layout, window: WebviewWindow) {
    let ready_app = app.clone();
    if !layout.host_entry.is_file() {
        show_fatal(
            &window,
            "The FreakSeek runtime is missing from this folder.",
            &format!("Expected {}. Re-extract the complete FreakSeek folder.", layout.host_entry.display()),
        );
        return;
    }
    let notify_window = window.clone();
    let shell_app = app.clone();
    let spawned = Host::spawn(layout, move |notice| match notice {
        Notice::Event(HostEvent::Ready { url }) => match url::Url::parse(&url) {
            Ok(target) if navigation_allowed(&target) => match open_main(&ready_app, target) {
                Ok(()) => {
                    let _ = notify_window.destroy();
                }
                Err(error) => show_fatal(&notify_window, "The FreakSeek window could not be opened.", &error.to_string()),
            },
            _ => show_fatal(&notify_window, "The runtime reported an address the shell will not open.", &url),
        },
        Notice::Event(HostEvent::Fatal { message, diagnostic }) => {
            show_fatal(&current_window(&ready_app).unwrap_or(notify_window.clone()), &message, diagnostic.as_deref().unwrap_or(&message));
        }
        Notice::Event(_) => {}
        Notice::Exited { tail } => {
            let shell = shell_app.state::<Shell>();
            if shell.quitting.lock().map(|quitting| *quitting).unwrap_or(true) {
                return;
            }
            let code = shell.host.lock().ok().and_then(|host| host.as_ref().and_then(|host| host.exit_code(Duration::from_secs(2))));
            let message = match code {
                Some(code) => format!("The FreakSeek runtime stopped unexpectedly (exit code {code})."),
                None => "The FreakSeek runtime stopped unexpectedly.".to_string(),
            };
            show_fatal(&current_window(&shell_app).unwrap_or(notify_window.clone()), &message, if tail.is_empty() { &message } else { &tail });
        }
    });
    match spawned {
        Ok(host) => {
            if let Ok(mut slot) = app.state::<Shell>().host.lock() {
                *slot = Some(Arc::new(host));
            }
        }
        Err(error) => show_fatal(
            &window,
            "The FreakSeek runtime could not be started.",
            &format!("{error}\nNode: {}\nEntry: {}", layout.node.display(), layout.host_entry.display()),
        ),
    }
}

/// Native folder dialog for the UI's workspace picker.
#[tauri::command]
async fn pick_directory(app: AppHandle) -> Option<String> {
    tauri::async_runtime::spawn_blocking(move || app.dialog().file().blocking_pick_folder())
        .await
        .ok()
        .flatten()
        .and_then(|path| path.into_path().ok())
        .map(|path| path.display().to_string())
}

/// Native `.gguf` file dialog for the composer's model control.
#[tauri::command]
async fn pick_gguf(app: AppHandle) -> Option<String> {
    tauri::async_runtime::spawn_blocking(move || {
        app.dialog().file().add_filter("GGUF models", &["gguf"]).blocking_pick_file()
    })
    .await
    .ok()
    .flatten()
    .and_then(|path| path.into_path().ok())
    .map(|path| path.display().to_string())
}

fn dropped_models(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")))
        .map(|path| path.display().to_string())
        .collect()
}

fn main() {
    let layout = match Layout::locate() {
        Ok(layout) => layout,
        Err(error) => {
            eprintln!("FreakSeek: cannot locate its folder: {error}");
            std::process::exit(1);
        }
    };
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show(app)))
        .plugin(tauri_plugin_dialog::init())
        .manage(Shell::default())
        .invoke_handler(tauri::generate_handler![pick_directory, pick_gguf])
        .setup(move |app| {
            let handle = app.handle().clone();
            let window = WebviewWindowBuilder::new(app, SPLASH, WebviewUrl::App("index.html".into()))
                .title("FreakSeek")
                .inner_size(1280.0, 840.0)
                .min_inner_size(720.0, 480.0)
                .on_navigation(navigation_allowed)
                .build()?;
            let splash_app = handle.clone();
            window.on_window_event(move |event| {
                // Closing the splash (startup or a startup failure) quits; the runtime may still be starting.
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    quit(&splash_app);
                }
            });

            let show_item = MenuItem::with_id(app, "show", "Show FreakSeek", true, None::<&str>)?;
            let models_item = MenuItem::with_id(app, "models", "Open models folder", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit FreakSeek", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_item, &models_item, &quit_item])?;
            let models_dir = layout.models.clone();
            let mut tray = TrayIconBuilder::with_id("freakseek")
                .tooltip("FreakSeek")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(move |app, event| match event.id().as_ref() {
                    "show" => show(app),
                    "models" => reveal(&models_dir),
                    "quit" => quit(app),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                        show(tray.app_handle());
                    }
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;

            if let Err(error) = layout.prepare() {
                show_fatal(&window, "FreakSeek cannot write to its folder.", &format!("{error}\nFolder: {}", layout.root.display()));
                return Ok(());
            }
            start_runtime(&handle, &layout, window);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("FreakSeek: failed to build the window")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { api, code, .. } = event {
                // Only Quit (which stops the runtime first) exits; other exit requests keep the tray alive.
                let quitting = app.state::<Shell>().quitting.lock().map(|quitting| *quitting).unwrap_or(true);
                if !quitting && code.is_none() {
                    api.prevent_exit();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigation_stays_on_loopback_and_shell_assets() {
        for allowed in ["http://127.0.0.1:4317/", "http://localhost:1/x", "tauri://localhost/index.html", "http://tauri.localhost/index.html", "about:blank"] {
            assert!(navigation_allowed(&url::Url::parse(allowed).unwrap()), "{allowed}");
        }
        for blocked in ["https://example.com/", "http://192.168.1.2/", "http://127.0.0.1.example.com/", "file:///etc/passwd", "ftp://127.0.0.1/"] {
            assert!(!navigation_allowed(&url::Url::parse(blocked).unwrap()), "{blocked}");
        }
    }

    #[test]
    fn forwards_only_gguf_drops() {
        let paths = [PathBuf::from("/m/a.GGUF"), PathBuf::from("/m/b.txt"), PathBuf::from("/m/c.gguf")];
        assert_eq!(dropped_models(&paths), vec!["/m/a.GGUF".to_string(), "/m/c.gguf".to_string()]);
    }
}
