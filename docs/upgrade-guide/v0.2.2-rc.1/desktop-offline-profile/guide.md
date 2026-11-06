---
kind: upgrade-guide
description: "The desktop app now runs only local GGUF models and no longer offers DeepSeek or third-party model providers, account sign-in, telemetry, or plugin installation."
---

# Desktop runs on local models only

## Change

The desktop profile stacks the new `@deepseek-ai/dsh-offline` bundle after `dsh-base` and `dsh-web-app`. It disables every row that contacts a remote service on its own: DeepSeek and pi-ai model routes, DeepSeek request metadata and session-log upload, account sign-in, telemetry and product analytics, plugin installation, and DeepSeek-API web search. Models come from GGUF files through `@deepseek-ai/dsh-llm-local-gguf`, and web search goes directly to DuckDuckGo. The welcome screen no longer asks for a DeepSeek sign-in or API key, and the app's own Electron session reaches only loopback and local resources.

Desktop users who chatted with DeepSeek or another hosted provider are affected: those providers no longer appear. Sessions recorded earlier stay readable, but continuing one sends its next turn to the selected local model. The `web` profile and the npm CLI are unchanged; `dsh offline` runs the same composition outside the desktop app.

## Migration

1. Copy one or more `.gguf` chat models into `$DSH_HOME/models` (default `~/.dsh/models`).
2. Open **Settings → Local models** and select **Load** on a model. It becomes the default model for new conversations.
3. To keep using a hosted provider, run the `web` profile with `dsh web` instead of the desktop app.

Confirm the migration by sending a message: the reply streams while the network is disconnected, and **Settings → Local models** shows the model as loaded.
