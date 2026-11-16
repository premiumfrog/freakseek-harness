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

It also disables model-facing tools that one local model cannot use well; see [Model Experience](#model-experience).

[`dsh-offline-app`](../offline-app/README.md) removes the Web app's account, telemetry, and plugin-installation rows for the desktop app, which also confines its own Electron session to loopback and local resources, covering update checks and spell-check dictionary downloads.

### Compaction for local context windows

The bundle sizes automatic compaction for local windows of 8k to 256k tokens: a conversation is summarized at 75% of the window, less the reply reservation and 2,048 tokens of headroom, keeping the newest 20% verbatim. The shared default headroom of 65,536 tokens exceeds a 32k window and leaves no budget.

<a id="model-experience"></a>
## Model Experience

### Lean tool surface and persona

#### What the model sees

The system prompt opens with `You are FreakSeek, a coding agent running on a local model on the user's computer.` in place of the upstream identity line (an agent preset's persona section follows it in the desktop app). The declared tools are `bash` or `pwsh` without `run_in_background`, `read`, `write`, `edit`, `glob`, `grep`, `web_search`, `web_fetch`, `todo_write`, and `skill`. The bundle disables the subagent, agent-team, workflow, goal, background-job, and working-directory tools and `read_image`, together with the guidance sections those tools contribute; the desktop's standard agent preset applies the same selection and also drops the schedule tool. `DSH_LOCAL_FULL_TOOLS=1` keeps every tool.

#### Token effect

With the Qwen 2.5 tokenizer the `cli` profile's first request falls from about 5,460 prompt tokens to about 2,200: the removed tool declarations account for about 3,100 tokens (`workflow` alone 866) and their guidance sections for the rest.

#### KV Cache effect

The tool list and system prompt stay fixed for a Session, so the shorter prefix is still reused in full by every later request; only the first request of a Session evaluates it.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- MCP servers a user configures keep their own network access.

<a id="dev-note"></a>
### Dev Note

None.
