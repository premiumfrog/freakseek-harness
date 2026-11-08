---
description: "The offline local-model provider for users and maintainers running GGUF models on their own computer through the embedded Rust engine."
kind: "package-reference"
---

# @deepseek-ai/dsh-llm-local-gguf

## Summary

`@deepseek-ai/dsh-llm-local-gguf` serves the `local` provider route from GGUF files on disk. Requests run in the `dsh-gguf-engine` child process built from [`native/gguf-engine`](../../../native/gguf-engine/README.md), which reads model files, renders each model's own chat template, and streams tokens over stdio. The engine opens no sockets and makes no network requests, so a conversation on this route never leaves the computer. The package also serves the `localModels` Remote that the Local models settings page uses to list, load, and unload models.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

Mount it beside `@deepseek-ai/dsh-llm`. The [`dsh-offline`](../../bundle/offline/README.md) bundle does this for the `offline` profile and the desktop app:

```yaml
- name: '@deepseek-ai/dsh-llm'
- name: '@deepseek-ai/dsh-llm-local-gguf'
  config:
    modelDirectories: ['~/models']
    contextLength: 32768
```

Put `.gguf` files in a model directory; the default is `$DSH_HOME/models`, and subdirectories are scanned two levels deep. Each supported file becomes a model whose id is its path relative to the directory without `.gguf`, for example `qwen/qwen2.5-7b-instruct-q4_k_m`. `listModels('local')` rescans the directories and inspects new or changed files; split-file shards after the first and multimodal projector files are skipped.

The first request for a model loads it; a request for another model replaces it, because one engine process holds one model at a time. The `localModels` Remote offers `snapshot()`, `load(modelId)`, and `unload()`; a successful `load` also saves the model as the default for new conversations when `@deepseek-ai/dsh-agent-default-model` is composed.

### Configuration

| Field | Default | Meaning |
|---|---|---|
| `enginePath` | resolved | Engine executable. When omitted: `$DSH_GGUF_ENGINE`, then `<resources>/gguf-engine/` inside the desktop app, then `native/gguf-engine/target/release/` in a source checkout |
| `modelDirectories` | `$DSH_HOME/models` | Directories scanned for GGUF files; `~` expands to the home directory, and an empty list selects the default |
| `contextLength` | `32768` | Context window requested at load, clamped to the model's trained length; the agent's system prompt and tool declarations alone take roughly 10k tokens |
| `cacheSlots` | `4` | KV caches kept at once; each conversation and each auxiliary request kind (such as session titles) uses its own, least recently used evicted first |
| `device` | `auto` | `auto` uses CUDA device 0 when the engine was built with `--features cuda` and a device opens, else the CPU; `cpu`, `cuda`, and `cuda:<n>` force a device |
| `gpuLayers` | all | Trailing transformer layers placed on the GPU; lower it when weights and the active KV cache exceed VRAM |
| `maxOutputTokens` | `4096` | Output cap for requests that name none |
| `threads` | `0` | Inference threads; `0` uses every core |
| `temperature`, `topK`, `topP`, `minP`, `repeatPenalty` | `0.6`, `40`, `0.95`, `0.05`, `1` | Sampling controls; a request's `temperature` overrides the configured one |
| `shutdownGraceMillis` | `3000` | Time the engine gets to exit when the plugin stops |

### Supported models

The engine runs the `llama` (Llama 2 and 3, Mistral, DeepSeek-R1 Llama distills), `qwen2` (Qwen 2 and 2.5, DeepSeek-R1 Qwen distills), and `qwen3` architectures with F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and Q2_K to Q8_K weights. Files with another architecture or I-quant weights are listed on the settings page as not supported, with the reason.

### Failures

Engine failures surface as `LlmError` codes: `CONTEXT_WINDOW_EXCEEDED` when the rendered prompt does not fit the loaded window, `LOCAL_UNKNOWN_MODEL` for an id no directory holds, `LOCAL_UNSUPPORTED_MODEL`, `LOCAL_ENGINE_UNAVAILABLE` when no executable was found, and `LOCAL_ENGINE_EXITED` when the process ended; the next request starts a new process. Cancellation stops generation at the next token and ends the stream as aborted.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

### Source map

| File | Role |
|---|---|
| [`src/index.ts`](src/index.ts) | Plugin entry: registers the `local` route, its configurable-provider entry, and the `localModels` Remote |
| [`src/config.ts`](src/config.ts) | Configuration schema and the explicit engine-path and directory resolution |
| [`src/engine-process.ts`](src/engine-process.ts) | JSON-lines client for one engine process, with out-of-band cancellation |
| [`src/runtime.ts`](src/runtime.ts) | Owner of the process, the directory scan, and the loaded-model state; one queue orders loads and generations |
| [`src/request.ts`](src/request.ts) | Harness history to OpenAI-style template messages |
| [`src/output-parser.ts`](src/output-parser.ts) | Incremental reasoning and tool-call extraction from raw model text |
| [`src/adapter.ts`](src/adapter.ts) | `LlmAdapter` implementation |
| [`src/service.ts`](src/service.ts) | `localModels` Remote |

### Request translation

The leading system messages merge into one `system` message, because most GGUF chat templates read only a leading one. Later system and developer text travels as user turns. Assistant reasoning is passed as `reasoning_content` and tool calls as `tool_calls` with parsed object arguments, the form Hugging Face templates expect. Tool results become `tool` messages named after their call. Images become a text placeholder because the route declares text-only input.

### Output parsing

The engine streams raw text. The parser recognizes `<think>…</think>` reasoning (including output that starts inside a reasoning block the template opened), Hermes and Qwen `<tool_call>{json}</tool_call>` calls, Mistral `[TOOL_CALLS][…]` arrays, and a Llama 3 reply that is one `{"name", "parameters"}` object naming a declared tool. Text that could still become a marker is held until it resolves, so stream boundaries never change the result.

### Prompt cache

The engine keeps one KV cache per slot and reuses the longest token prefix a slot's next prompt shares with its previous one. The adapter keys slots by Session id and request purpose, so a session-title or compaction request does not evict the conversation. Because the harness re-sends the conversation history unchanged, a follow-up turn evaluates only its new tokens; the reused positions are reported as `cacheReadTokens`.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

- [`native/gguf-engine`](../../../native/gguf-engine/README.md) — the inference engine and its stdio protocol.
- [`dsh-offline`](../../bundle/offline/README.md) — the profile bundle that mounts this provider without any remote service.
- [`dsh-client-ui-settings-local-models`](../../client/ui-settings-local-models/README.md) — the settings page over the `localModels` Remote.
- [LLM streaming subsystem](../../../docs/subsystems/llm-streaming.md) — the chunk protocol this adapter emits.

-----

<a id="model-experience"></a>
## Model Experience

### Local model requests

#### What the model sees

The request history and tool declarations rendered through the model file's own `tokenizer.chat_template`, or a ChatML template with Hermes-style tool instructions when the file has none or its template fails to render. Images appear as the text `[image omitted: this local model reads text only]`.

#### Token effect

Every rendered character is counted by the model's own tokenizer; tool declarations add their JSON schemas to the prompt as the template places them. Usage reports uncached prompt tokens, reused prefix tokens, and generated tokens.

#### KV Cache effect

The engine reuses the longest token prefix shared with the previous request of the same conversation and request purpose. Loading another model, a changed system prompt or tool list, or eviction after more than `cacheSlots` other conversations discards the reuse from the first differing token.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- One engine process holds one model; concurrent conversations on different models reload weights on each switch, and concurrent requests run one after another.
- GPU offload requires an engine built with `pnpm run build:gguf-engine:cuda`; the default build runs on the CPU.
- Split GGUF files and I-quant weight types are not supported.
- Templates that rely on Jinja features outside the engine's minijinja dialect fall back to ChatML.
- Tool calling depends on the model following its template's tool format; a malformed call stays visible as text.

<a id="dev-note"></a>
### Dev Note

`tests/loader-composition.spec.ts` boots the route through the Loader and runs the real engine on a synthetic bigram model written by `tests/gguf-fixture.ts`; `pnpm run test` builds the engine first with `pnpm run build:gguf-engine`.
