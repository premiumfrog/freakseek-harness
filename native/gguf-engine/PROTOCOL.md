---
description: "The stdio JSON-lines protocol of dsh-gguf-engine, for maintainers writing or debugging an engine host."
kind: "package-reference"
---

# dsh-gguf-engine protocol

Protocol revision **1**. A host writes one JSON object per line to stdin and reads one JSON object per line from stdout. Every request carries a host-chosen string `id`; every reply echoes it. stderr carries only diagnostics.

## Replies

| `type` | Fields | Meaning |
|---|---|---|
| `result` | `ok: true`, `value` | The request finished |
| `result` | `ok: false`, `error: { code, message }` | The request failed; `code` is stable |
| `start` | `promptTokens`, `cachedTokens`, `opensReasoning` | `generate` only: the prompt is ready; `cachedTokens` positions were reused from the previous request; `opensReasoning` means the rendered prompt ends inside an opened `<think>` block |
| `delta` | `text` | `generate` only: the next piece of output text, always valid UTF-8 |

A request line that is not valid JSON is answered with `id: null` and `BAD_REQUEST`.

## Requests

| `op` | Fields | `value` on success | Ordering |
|---|---|---|---|
| `hello` | — | `{ protocol, version }` | Answered immediately |
| `inspect` | `path` | Header facts: `name`, `architecture`, `supported`, `unsupportedReason`, `contextLength`, `parameterCount`, `fileType`, `fileBytes`, `hasChatTemplate`, `templateTools`, `templateReasoning` | Answered immediately, even during a generation |
| `load` | `path`, optional `contextLength`, optional `slots` (default 4) | `{ facts, contextLength, vocab }`; the window is clamped to 256 … the trained length | Queued |
| `unload` | — | `null` | Queued |
| `status` | — | `{ loaded, path?, contextLength? }` | Queued |
| `generate` | `messages`, optional `tools`, `maxTokens`, `stop`, `sampling`, `thinking`, `slot` | `{ finishReason, promptTokens, cachedTokens, completionTokens }`; `finishReason` is `stop`, `length`, or `cancelled` | Queued |
| `cancel` | `target` | `null` | Answered immediately; the target generation stops at its next token |
| `shutdown` | — | `null` | Answered immediately; queued work is cancelled and the process exits |

Closing stdin has the same effect as `shutdown`.

`messages` and `tools` use the OpenAI chat format that Hugging Face chat templates read: `role`, `content`, optional `reasoning_content`, `tool_calls` (with object `arguments`), `tool_call_id`, and `name`; tools are `{ type: 'function', function: { name, description, parameters } }`. `sampling` holds `temperature` (0 is greedy), `topK`, `topP`, `minP`, `repeatPenalty`, `repeatLastN`, and `seed`. `thinking` is passed to templates as `enable_thinking`. `slot` (default `""`) names the KV cache the request uses: each slot keeps its own cache and reuses the prefix its previous request shares, and loading keeps at most `slots` caches, evicting the least recently used.

## Error codes

| Code | Meaning |
|---|---|
| `BAD_REQUEST` | Malformed line or unknown `op` |
| `NO_MODEL` | `generate` before a successful `load` |
| `UNSUPPORTED_MODEL` | The architecture is not implemented |
| `CONTEXT_WINDOW_EXCEEDED` | The rendered prompt does not fit the loaded window |
| `ENGINE_ERROR` | Any other failure, such as an unreadable file or a template error |
