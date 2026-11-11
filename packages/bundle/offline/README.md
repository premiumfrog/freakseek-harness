---
description: "The offline profile bundle for users who run dsh and the desktop app entirely on local GGUF models with no remote service."
kind: "package-bundle"
---

# @deepseek-ai/dsh-offline

## Summary

`@deepseek-ai/dsh-offline` is the offline layer of FreakSeek's two surfaces: the `cli` profile (`dsh-base`, `dsh-headless`, `dsh-offline`) and the desktop app (`dsh-base`, `dsh-web-app`, `dsh-offline`, [`dsh-offline-app`](../offline-app/README.md)). It removes every `dsh-base` row that contacts a remote service on its own, mounts [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.md) as the only model route, and replaces DeepSeek-API web search with a direct [DuckDuckGo](../../web/web-search-duckduckgo/README.md) query. The only outbound traffic left is model-initiated: the web search tool sends its query text, and the web fetch tool retrieves the URL the model names.

## Table of Contents

- [Use this bundle](#use-this-bundle)
- [What it removes](#what-it-removes)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-bundle"></a>
## Use this bundle

```sh
dsh cli "summarize this repository"     # one task on a local model, printed to the terminal
dsh cli --resume <session> "and now…"   # continue that Session
dsh --profile cli --dump-config
```

Put `.gguf` files in `$DSH_HOME/models`. In the desktop app, open **Settings → Local models** and select **Load**; loading a model makes it the default for new conversations. The CLI uses the only supported model in the folders; with several, `DSH_LOCAL_MODEL` names one by id, such as `qwen/qwen2.5-7b-instruct-q4_k_m`. `DSH_LOCAL_MODELS_ROOT` sets the models root whose subfolders are collections, `DSH_LOCAL_MODEL_DIRS` (path-list separated) replaces collections with fixed directories, `DSH_GGUF_ENGINE_DIR` names a folder with `cuda/` and `cpu/` engine builds, `DSH_GGUF_ENGINE` names one engine executable, `DSH_GGUF_DEVICE` and `DSH_GGUF_GPU_LAYERS` set the provider's `device` and `gpuLayers`, and `DSH_LOCAL_MODEL` names the default model id before one is loaded.

To remove the remaining model-initiated traffic as well, disable the web tool in the profile patch:

```yaml
- id: tool-web
  disabled: true
```

<a id="what-it-removes"></a>
## What it removes

| Rows | Remote service |
|---|---|
| `llm-deepseek`, `llm-deepseek-account`, `llm-pi-ai`, `deepseek-llm-api-extensions`, `plugin-package-inventory-deepseek`, `session-log-deepseek` | DeepSeek and third-party model APIs, request metadata, session-log upload |
| `deepseek-account` | DeepSeek account sign-in |
| `otel`, `session-telemetry-otel` | Telemetry |
| `plugin-manager`, `tool-plugin-manager` | npm-backed plugin installation |
| `web-search-deepseek`, `web-deepseek-api` | Web search through the DeepSeek API |

[`dsh-offline-app`](../offline-app/README.md) removes the Web app's account, telemetry, and plugin-installation rows for the desktop app, which also confines its own Electron session to loopback and local resources, covering update checks and spell-check dictionary downloads.

<a id="model-experience"></a>
## Model Experience

None, as the bundle only selects rows; the local model route and the web tools own every model-visible input.

#### KV Cache effect

The bundle adds nothing to the request prefix; the local route's prefix reuse applies.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- MCP servers a user configures keep their own network access.

<a id="dev-note"></a>
### Dev Note

None.
