---
description: "Code mode for the dsh web client: a Session header toggle that opens the workspace files page and a terminal beside the conversation; for users and maintainers of the coding layout."
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-code-mode

## Summary

`@deepseek-ai/dsh-client-ui-code-mode` adds a **Code** toggle to the Session header. Pressing it puts the Session in code mode: the right Sidebar expands with the workspace files page in one pane and a terminal in a second pane beside it, so the conversation, the tree, and a shell sit side by side. Pressing it again collapses the Sidebar; the tabs stay open for the next entry. The FreakSeek desktop composition mounts it through [`dsh-offline-app`](../../bundle/offline-app/README.md).

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount it beside [`ui-sidebar-right`](../ui-sidebar-right/README.md), [`ui-sidebar-files`](../ui-sidebar-files/README.md), and [`ui-sidebar-terminal`](../ui-sidebar-terminal/README.md):

```yaml
- id: ui-code-mode
  name: '@deepseek-ai/dsh-client-ui-code-mode'
```

The toggle sits in the header's utilities row, left of the Schedule catalog, and shows its pressed state while the Session is in code mode. `Ctrl+Shift+E` (`⌘⇧E` on macOS) runs the same command in the desktop app and can be rebound in the shortcut settings. Entering code mode reveals a files page or terminal that is already open instead of opening another; a missing tab type is skipped, so the toggle still expands the Sidebar when, for example, no terminal provider is mounted.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

| File | Role |
|---|---|
| [`src/client/index.ts`](src/client/index.ts) | Plugin entry: copy, the shortcut command, and the header registration |
| [`src/client/controller.ts`](src/client/controller.ts) | `CodeModeController`: per-Session state and the enter and leave operations over `ctx.sidebarRight` |
| [`src/client/CodeModeButton.tsx`](src/client/CodeModeButton.tsx) | The header toggle |
| [`src/client/locales.ts`](src/client/locales.ts) | Copy in the eight shipped locales |

Entering opens the `files` page through `ctx.sidebarRight.openTab`, which expands the column, then focuses the Session's open `terminal` tab from `ctx.sidebarRight.openTabs` or opens one with `preferNewPane: true`, which splits the pane when there is room. Leaving collapses the column with `toggleExpanded`. The pressed state is the controller's own per-Session set; pressing while the column was collapsed some other way enters code mode again instead of leaving it.

</details>

-----

<a id="model-experience"></a>
## Model Experience

None, as the package only arranges right-Sidebar tabs; nothing it does reaches a model request.

#### KV Cache effect

None.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Code mode state lives in memory: a reload restores the Sidebar layout but shows the toggle released.
- The changed-files review opens from a turn's changed-files card, because a review tab addresses one turn; code mode does not open it.
- Collapsing the Sidebar with its own control leaves the toggle pressed until the next press, which re-enters code mode.

<a id="dev-note"></a>
### Dev Note

`tests/apply.client.spec.ts` registers the plugin against a fake right Sidebar and asserts the tabs each press opens.
