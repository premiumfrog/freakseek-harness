---
description: "The Local models settings section for users picking, loading, and unloading GGUF models in the Web and desktop clients."
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-settings-local-models

## Summary

`@deepseek-ai/dsh-client-ui-settings-local-models` adds **Settings → Local models** to the desktop client. The section lists every GGUF file the Host's [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.md) provider found, with its architecture, size, quantization, and tool support; **Load** loads one into the local engine and makes it the default for new conversations, and **Unload** releases its memory. The page also shows the scanned model folders and any problem. In the desktop app, **Load from file…** and the composer's model control open the system file dialog: choosing any `.gguf` file loads it and switches the current conversation to it.

## Table of Contents

- [Use this package](#use-this-package)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

The [`dsh-offline-app`](../../bundle/offline-app/README.md) bundle mounts the row. The section reads the `localModels` Remote and refreshes on `llm/models-updated` for the `local` route, so a load started in another window or a rescan updates every open page. A deployment without the provider shows that it serves no local models.

The file flows exist only where the desktop preload exposes `__DSH_GGUF_PICKER__` (the native dialog filtered to `.gguf`). There the plugin also occupies the composer's `conversation.input.model` seat, which is single, so the composition must not also mount `ui-model-selection`. A chosen path goes to `localModels.loadFile`, which validates it on the Host and remembers it across restarts; the plugin then calls `session.selectModel` with the `local` provider and the returned id. A failure from the dialog, the load, or the switch appears as the seat's tooltip and on the page.

-----

<a id="model-experience"></a>
## Model Experience

None, as the page changes which model serves new conversations without constructing model input.

#### KV Cache effect

Loading a different model discards the engine's cached prefix.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Model folders are edited in the profile configuration (`modelDirectories`), not on the page.
- The composer's model control shows the loaded local model, not each conversation's saved selection, and offers no reasoning-effort choice.
- Imported files cannot be removed from the catalog on the page; edit `importedModelsFile` instead.

<a id="dev-note"></a>
### Dev Note

None.
