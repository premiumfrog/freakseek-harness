---
description: "The offline profile bundle for users who run dsh and the desktop app entirely on local GGUF models with no remote service."
kind: "package-bundle"
---

# @deepseek-ai/dsh-offline

## Summary

`@deepseek-ai/dsh-offline` is the last layer of the `offline` profile (`dsh-base`, `dsh-web-app`, `dsh-offline`) and of the desktop app. It removes every row that contacts a remote service on its own, mounts [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.md) as the only model route, and replaces DeepSeek-API web search with a direct [DuckDuckGo](../../web/web-search-duckduckgo/README.md) query. The only outbound traffic left is model-initiated: the web search tool sends its query text, and the web fetch tool retrieves the URL the model names.

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
dsh offline                     # the Web app on local models
dsh --profile offline --dump-config
```

Put `.gguf` files in `$DSH_HOME/models`, open **Settings → Local models**, and select **Load**. Loading a model makes it the default for new conversations; the composer's model picker lists every supported file. `DSH_LOCAL_MODEL_DIRS` (path-list separated) replaces the scanned directories, `DSH_GGUF_ENGINE` names the engine executable, `DSH_GGUF_DEVICE` and `DSH_GGUF_GPU_LAYERS` set the provider's `device` and `gpuLayers`, and `DSH_LOCAL_MODEL` names the default model id before one is loaded.

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
| `deepseek-account`, `account-controller`, `ui-settings-account` | DeepSeek account sign-in |
| `otel`, `session-telemetry-otel`, `desktop-product-telemetry`, `product-analytics` | Telemetry and product analytics |
| `plugin-manager`, `tool-plugin-manager`, `ui-plugin-manager` | npm-backed plugin installation |
| `web-search-deepseek`, `web-deepseek-api` | Web search through the DeepSeek API |

The desktop app also confines its own Electron session to loopback and local resources, which covers update checks and spell-check dictionary downloads.

<a id="model-experience"></a>
## Model Experience

None, as the bundle only selects rows; the local model route and the web tools own every model-visible input.

#### KV Cache effect

The bundle adds nothing to the request prefix; the local route's prefix reuse applies.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- MCP servers a user configures keep their own network access.
- Sidebar browser tabs browse the web on the user's request; they use separate partitions that the desktop offline policy does not confine.

<a id="dev-note"></a>
### Dev Note

None.
