---
description: "The desktop layer of the offline composition for maintainers changing which Web app rows the FreakSeek desktop app removes or adds."
kind: "package-bundle"
---

# @deepseek-ai/dsh-offline-app

## Summary

`@deepseek-ai/dsh-offline-app` is the last layer of the desktop composition (`dsh-base`, `dsh-web-app`, [`dsh-offline`](../offline/README.md), `dsh-offline-app`). It removes the rows [`dsh-web-app`](../web-app/README.md) contributes that contact a remote service on their own, removes the sidebar web-browser tabs and the catalog model selector, and adds the [Local models](../../client/ui-settings-local-models/README.md) settings page with its file-dialog model control and [code mode](../../client/ui-code-mode/README.md). Only the desktop host boots this composition; the `dsh` launcher refuses profiles that serve the Web app.

## Table of Contents

- [What it changes](#what-it-changes)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="what-it-changes"></a>
## What it changes

| Rows | Change |
|---|---|
| `account-controller`, `ui-settings-account` | Removed: DeepSeek account sign-in |
| `desktop-product-telemetry`, `product-analytics` | Removed: product telemetry and analytics |
| `ui-plugin-manager` | Removed: npm-backed plugin installation page |
| `ui-sidebar-browser` | Removed: sidebar web-browser tabs |
| `ui-model-selection` | Removed: catalog model selector and `/model`; the composer's model control opens a `.gguf` file dialog instead |
| `ui-settings-local-models` | Added: list, load, and unload local GGUF models, and the composer's file-dialog model control |
| `ui-code-mode` | Added: the Code toggle that opens the workspace files page beside a terminal |

<a id="model-experience"></a>
## Model Experience

None, as the bundle only selects rows; no row it touches reaches a model request.

#### KV Cache effect

The bundle adds nothing to the request prefix.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- None.

<a id="dev-note"></a>
### Dev Note

None.
