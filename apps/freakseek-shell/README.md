---
description: "The FreakSeek desktop shell: a Rust (Tauri) window that starts the bundled runtime from the portable folder; for users of the portable build and maintainers changing how the window, tray, and runtime connect."
---

# FreakSeek shell

`FreakSeek(.exe)` is the desktop application of the portable FreakSeek folder. It is a [Tauri](https://tauri.app) v2 shell over the operating system's web view (WebView2 on Windows), replacing the Electron app in `apps/desktop`. It starts the bundled runtime ([`desktop-host`](../desktop-host/src/index.ts)) with the folder's Node.js, waits for the runtime's loopback URL, and shows the FreakSeek UI from it.

## Folder layout

| Path | Contents |
|---|---|
| `FreakSeek(.exe)` | This shell |
| `runtime/node(.exe)` | Bundled Node.js |
| `runtime/app/` | Deployed `@deepseek-ai/dsh-desktop-host` tree; the shell runs `runtime/app/lib/index.js` |
| `engine/cpu/`, `engine/cuda/` | GGUF engines |
| `models/` | Model collections, one subfolder each |
| `data/` | Conversations, settings, the desktop profile (`data/profile`), and `data/logs/runtime.log` |
| `freakseek(.cmd)` | CLI launcher with the same folder environment |

The shell passes `DSH_HOME=data`, `DSH_LOCAL_MODELS_ROOT=models`, `DSH_GGUF_ENGINE_DIR=engine`, `DSH_TELEMETRY_DISABLED=1`, and `FREAKSEEK_SHELL=stdio` to the runtime. `FREAKSEEK_ROOT` overrides the folder, which is otherwise the directory of the executable. [`scripts/assemble-portable.ts`](../../scripts/assemble-portable.ts) (`pnpm run assemble:portable`) builds the folder after `pnpm run build`.

## Runtime link

The runtime runs in stdio control mode ([`control-channel.ts`](../desktop-host/src/control-channel.ts)): it writes control events as stdout lines prefixed with `@freakseek:` (`starting`, `ready` with the authenticated URL, `fatal`, `shutdown-complete`) and reads JSON commands (`shutdown`, `quit-inspection`) from stdin; other output goes to `data/logs/runtime.log`. In stdio mode the runtime creates the desktop profile in `data/profile` on first launch. Quitting sends `shutdown` and kills the runtime after 5 seconds.

## Windows and navigation

A splash window shows startup progress and, on failure, the diagnostic with a Copy button. On `ready`, the UI opens in a new window whose first load is the authenticated URL: the runtime's launch-token exchange sets a `SameSite=Strict` session cookie, which the web view would not send on a redirect initiated from the splash page. Both windows may navigate only to loopback hosts and the shell's own assets.

The loopback page gets two bridges through an initialization script: `window.__DSH_DIRECTORY_PICKER__.pick()` and `window.__DSH_GGUF_PICKER__.pick()`, which call the `pick_directory` and `pick_gguf` commands (native dialogs). The `runtime-pickers` capability grants exactly these two commands to `http://127.0.0.1:*`. Dropping `.gguf` files onto the window dispatches a `freakseek:gguf-dropped` DOM event with `{ paths }` on the page.

Closing the UI window hides it to the tray; the tray offers Show FreakSeek, Open models folder, and Quit FreakSeek. A second launch focuses the running instance.

## Build

```sh
cd apps/freakseek-shell
cargo build --release            # target/release/FreakSeek(.exe)
cargo test
```

Windows needs the MSVC toolchain and the WebView2 runtime (preinstalled on Windows 10 and 11). Linux needs WebKitGTK 4.1, GTK 3, and libayatana-appindicator development packages.

## Known Limitations and Deferred Work

- The native drag-and-drop handler is enabled so dropped `.gguf` paths reach the shell; while it is, the page receives no HTML file drops, so dropping attachments onto the composer does not work in this shell.
- Desktop-only Electron bridges (window controls, file actions, keyboard shortcuts registry, updates) are not provided; the UI runs with its Web behavior.
- Office skills are unavailable because the portable folder ships no primary runtime.
- `icons/` holds a placeholder mark until the FreakSeek artwork lands.
