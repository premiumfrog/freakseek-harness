---
description: "The Local models settings section for users picking, loading, and unloading GGUF models in the Web and desktop clients."
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-settings-local-models

## Summary

`@deepseek-ai/dsh-client-ui-settings-local-models` adds **Settings → Local models** to the Web and desktop clients. The section lists every GGUF file the Host's [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.md) provider found, with its architecture, size, quantization, and tool and reasoning support; **Load** loads one into the local engine and makes it the default for new conversations, and **Unload** releases its memory. The page also shows the scanned model folders and any engine or folder problem.

## Table of Contents

- [Use this package](#use-this-package)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

The [`dsh-offline`](../../bundle/offline/README.md) bundle mounts the row. The section reads the `localModels` Remote and refreshes on `llm/models-updated` for the `local` route, so a load started in another window or a rescan updates every open page. A deployment without the provider shows that it serves no local models.

-----

<a id="model-experience"></a>
## Model Experience

None, as the page changes which model serves new conversations without constructing model input.

#### KV Cache effect

Loading a different model discards the engine's cached prefix.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Model folders are edited in the profile configuration (`modelDirectories`), not on the page.

<a id="dev-note"></a>
### Dev Note

None.
