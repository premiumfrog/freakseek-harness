---
kind: upgrade-guide
description: "FreakSeek ships only the desktop app and the cli profile, both on local GGUF models; the browser Web UI, hosted model providers, account sign-in, telemetry, and plugin installation are gone."
---

# Desktop and CLI run on local models only; no browser UI

## Change

The desktop profile stacks the new `@deepseek-ai/dsh-offline` and `@deepseek-ai/dsh-offline-app` bundles after `dsh-base` and `dsh-web-app`, and the new `cli` profile stacks `dsh-offline` after `dsh-base` and `dsh-headless`. It disables every row that contacts a remote service on its own: DeepSeek and pi-ai model routes, DeepSeek request metadata and session-log upload, account sign-in, telemetry and product analytics, plugin installation, and DeepSeek-API web search. Models come from GGUF files through `@deepseek-ai/dsh-llm-local-gguf`, and web search goes directly to DuckDuckGo. The welcome screen no longer asks for a DeepSeek sign-in or API key, the app's own Electron session reaches only loopback and local resources, and the sidebar web-browser tabs are removed.

Desktop users who chatted with DeepSeek or another hosted provider are affected: those providers no longer appear. Sessions recorded earlier stay readable, but continuing one sends its next turn to the selected local model. The `dsh` launcher refuses every profile that serves the browser Web UI, including `web` and custom profiles that list `dsh-web-app`, and the earlier `offline` profile is removed; `dsh cli "<task>"` runs one task on a local model in the terminal.

## Migration

1. Copy one or more `.gguf` chat models into `$DSH_HOME/models` (default `~/.dsh/models`).
2. Open **Settings → Local models** and select **Load** on a model. It becomes the default model for new conversations.
3. Replace `dsh web` and `dsh offline` launches with the desktop app, or with `dsh cli "<task>"` for terminal tasks. Hosted providers are not available in either surface.

Confirm the migration by sending a message: the reply streams while the network is disconnected, and **Settings → Local models** shows the model as loaded.
