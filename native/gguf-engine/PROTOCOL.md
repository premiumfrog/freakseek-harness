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
| `hello` | — | `{ protocol, version, cuda }`; `cuda` is whether the build can use CUDA devices | Answered immediately |
| `devices` | — | `{ cuda, minComputeCapability, devices: [{ ordinal, name, computeCapability, totalBytes, freeBytes, supported }] }`; `minComputeCapability` is `major * 10 + minor` of the kernels (75 for release builds), and `supported` marks devices that meet it. A CPU build, or a driver with no device, reports an empty list | Answered immediately |
| `inspect` | `path` | Header facts: `name`, `architecture`, `supported`, `unsupportedReason`, `contextLength`, `parameterCount`, `fileType`, `fileBytes`, `hasChatTemplate`, `templateTools`, `templateReasoning` | Answered immediately, even during a generation |
| `load` | `path`, optional `contextLength`, optional `slots` (default 4), optional `device` (`auto` default, `cpu`, `cuda`, `cuda:<n>`), optional `gpuLayers` (trailing layers on the GPU; default all) | `{ facts, contextLength, vocab, device, gpuLayers, gpuExperts, experts, vramTotalBytes, vramFreeBytes, loadMs, warning? }`; the window is clamped to 256 … the trained length; `device` is `cpu` or `cuda:<n>` and `gpuLayers` counts offloaded layers | Queued |
| `unload` | — | `null` | Queued |
| `status` | — | `{ loaded, path?, contextLength?, device?, gpuLayers?, gpuExperts?, experts?, vramTotalBytes?, vramFreeBytes?, cachedTokens? }` | Queued |
| `generate` | `messages`, optional `tools`, `maxTokens`, `stop`, `sampling`, `thinking`, `slot` | `{ finishReason, promptTokens, cachedTokens, completionTokens }`; `finishReason` is `stop`, `length`, or `cancelled` | Queued |
| `cancel` | `target` | `null` | Answered immediately; the target generation stops at its next token |
| `shutdown` | — | `null` | Answered immediately; queued work is cancelled and the process exits |

Closing stdin has the same effect as `shutdown`.

`messages` and `tools` use the OpenAI chat format that Hugging Face chat templates read: `role`, `content`, optional `reasoning_content`, `tool_calls` (with object `arguments`), `tool_call_id`, and `name`; tools are `{ type: 'function', function: { name, description, parameters } }`. `sampling` holds `temperature` (0 is greedy), `topK`, `topP`, `minP`, `repeatPenalty`, `repeatLastN`, and `seed`. `thinking` is passed to templates as `enable_thinking`. `slot` (default `""`) names the KV cache the request uses: each slot keeps its own cache and reuses the prefix its previous request shares, and loading keeps at most `slots` caches, evicting the least recently used. A model with recurrent layers reuses the prefix up to its newest state checkpoint, so `cachedTokens` can be lower than the shared prefix. `toolGrammar` (default `true`) constrains the text after a `<tool_call>` opener to a JSON call of a declared tool whose arguments match its schema.

## Error codes

| Code | Meaning |
|---|---|
| `BAD_REQUEST` | Malformed line or unknown `op` |
| `NO_MODEL` | `generate` before a successful `load` |
| `UNSUPPORTED_MODEL` | The architecture is not implemented |
| `CONTEXT_WINDOW_EXCEEDED` | The rendered prompt does not fit the loaded window |
| `BAD_DEVICE` | `device` is not `auto`, `cpu`, `cuda`, or `cuda:<n>` |
| `DEVICE_UNAVAILABLE` | `cuda` was requested from a build without CUDA, or the CUDA device could not be opened; `auto` falls back to the CPU instead, also for a GPU older than `minComputeCapability` |
| `ENGINE_ERROR` | Any other failure, such as an unreadable file or a template error |

A `generate` result carries `finishReason`, `promptTokens`, `cachedTokens`, `completionTokens`, `prefillMs`, `decodeMs`, `promptTokensPerSecond` (uncached prompt tokens), and `tokensPerSecond` (generated tokens). `gpuExperts` and `experts` count routed experts on the GPU and in the model; the `vram*` fields are `null` on the CPU. Before placing weights on a GPU, `load` compares that GPU's quantized matrix products with exact host products for the model's weight types. With `device` omitted or `auto`, a GPU that fails the comparison is replaced by the CPU and `warning` says why; with an explicit `cuda` device the load fails with `GPU_KERNEL_MISMATCH`. A forward pass that produces non-finite logits fails the request with `NUMERIC_FAILURE`.

`load` and `generate` send `{ id, type: 'progress', stage, done, ... }` lines before their result. `stage: 'load'` carries `done` and `total` decoder layers loaded, at most ten a second and always the last. `stage: 'prefill'` carries `done` and `total` uncached prompt tokens evaluated, once per evaluation chunk. `stage: 'decode'` carries `done` tokens generated, `tokensPerSecond`, and `contextUsed` positions, at most twice a second. A `generate` result's `finishReason` is `repetition` when the newest 64 generated tokens repeat a cycle of at most 16 tokens; generation stops there. Weight matrices in F16, BF16, or F32 with at least 2^24 elements load as Q8_0. On a GPU, KV caches hold F16 keys and values.
