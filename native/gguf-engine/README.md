---
description: "The offline GGUF inference engine for users and maintainers building, testing, or extending the local model runtime behind the `local` provider route."
kind: "package-reference"
---

# dsh-gguf-engine

## Summary

`dsh-gguf-engine` is a Rust executable that runs GGUF chat models on the CPU. It parses the GGUF header, rebuilds the tokenizer from `tokenizer.ggml.*` metadata, renders the file's own Jinja chat template, keeps weights quantized in memory, and generates tokens with a prefix-reusing KV cache. A host drives it with one JSON object per line on stdin and reads one JSON object per line from stdout ([protocol](PROTOCOL.md)). The engine opens no sockets: its dependency tree contains no networking crate, and its only I/O is the model file and the two pipes. [`@deepseek-ai/dsh-llm-local-gguf`](../../packages/llm/llm-local-gguf/README.md) is its host.

## Table of Contents

- [Build and test](#build-and-test)
- [Supported models](#supported-models)
- [Understand the implementation](#understand-the-implementation)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="build-and-test"></a>
## Build and test

```sh
pnpm run build:gguf-engine          # cargo build --release --locked
cd native/gguf-engine && cargo test --release
```

The release binary lands in `native/gguf-engine/target/release/dsh-gguf-engine` (`.exe` on Windows), where source checkouts find it. Desktop packaging copies it into the application resources as `gguf-engine/dsh-gguf-engine`.

### CUDA

```sh
pnpm run build:gguf-engine:cuda     # cargo build --release --locked --features cuda
```

The `cuda` feature needs the CUDA Toolkit 12.x (`nvcc`) at build time and the NVIDIA driver plus the toolkit's cuBLAS, cuRAND, and runtime libraries at run time. `CUDA_COMPUTE_CAP` sets the lowest compute capability the kernels support; FreakSeek releases use `75` (RTX 20-series), and newer GPUs (RTX 30, 40, and 50-series and workstation cards) compile the kernels' PTX on first use and cache it. Without the variable the build targets the GPU that `nvidia-smi` reports. On Windows, build from an x64 Native Tools prompt of Visual Studio 2022 Build Tools so `nvcc` finds `cl.exe`.

`load` places the last `gpuLayers` transformer layers (default all) and the output head on the GPU; earlier layers and the embedding lookup run on the CPU, so a model larger than VRAM still loads with a lower `gpuLayers`. An offloaded layer's KV cache lives in VRAM while its slot is active and moves to system memory when the slot is parked. With `device: auto` the engine uses CUDA device 0 when the build has CUDA and the device meets that minimum, else the CPU. The `devices` operation reports each GPU's name, compute capability, and free and total memory.

The tests cover tokenizer parity, forward-pass parity, and the protocol:

- `tests/model_reference.rs` writes random `llama`, tied-output `llama`, `qwen2`, and `qwen3` models and compares the engine's logits with candle-transformers' reference implementations for prefill, incremental decoding, and KV-cache prefix reuse.
- `tests/tokenizer_reference.rs` compares tokenization with llama.cpp's `test-tokenizer-0` vocabularies (`llama-spm`, `llama-bpe`, `qwen2`, `phi-3`, `gpt-2`) when `DSH_GGUF_VOCAB_DIR` names a directory holding llama.cpp's `models/ggml-vocab-*.gguf{,.inp,.out}` files; it skips with a message otherwise.
- `tests/protocol.rs` drives the built binary through hello, inspect, load, generate with cache reuse, cancel, unload, and shutdown.

`--threads <n>` limits inference threads; `--version` prints the engine and protocol versions.

<a id="supported-models"></a>
## Supported models

| `general.architecture` | Models | Notes |
|---|---|---|
| `llama`, `mistral` | Llama 2, Llama 3.x, Mistral, DeepSeek-R1 Llama distills | Adjacent-pair RoPE; Llama 3.1 `rope_freqs` scaling |
| `qwen2` | Qwen 2, Qwen 2.5, DeepSeek-R1 Qwen distills | Q/K/V biases; half-split RoPE |
| `qwen3` | Qwen 3 dense | Per-head Q/K RMS norm |

Weight types are those candle's quantized kernels load: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and Q2_K to Q8_K. Tokenizers are SentencePiece score merging (`llama`) and byte-level BPE (`gpt2`) with the pre-tokenizer named by `tokenizer.ggml.pre`.

<a id="understand-the-implementation"></a>
## Understand the implementation

| File | Role |
|---|---|
| [`src/main.rs`](src/main.rs) | JSON-lines protocol: a reader thread answers `hello`, `inspect`, `cancel`, and `shutdown`; a worker thread runs `load`, `unload`, `status`, and `generate` in order |
| [`src/device.rs`](src/device.rs) | Device resolution (`auto`, `cpu`, `cuda:<n>`) and trailing-layer GPU placement |
| [`src/engine.rs`](src/engine.rs) | Header inspection, prompt preparation, prefix reuse, the generation loop, and stop strings |
| [`src/model.rs`](src/model.rs) | Transformer forward pass over candle tensors; growable KV cache with truncation |
| [`src/tokenizer.rs`](src/tokenizer.rs) | GGUF tokenizer: atomic special tokens, heap-ordered SentencePiece merges, ranked BPE, and incremental UTF-8 decoding |
| [`src/template.rs`](src/template.rs) | minijinja rendering of `tokenizer.chat_template` with a ChatML fallback |
| [`src/sampler.rs`](src/sampler.rs) | Repetition penalty, temperature, top-k, top-p, and min-p sampling |

Prefill runs in chunks of 256 tokens with a causal mask, which bounds the attention matrix. Grouped-query attention folds query heads per key/value head instead of copying keys and values. Embedding rows are dequantized on demand, so a large vocabulary does not need a full-precision copy of the embedding table.

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- GPU offload covers CUDA only; candle's Metal backend is not enabled.
- The forward-pass parity tests run on the CPU; CI does not exercise CUDA kernels.
- The KV cache is F32, so long windows on large models need substantial memory.
- Mixture-of-experts, Gemma, Phi, and other architectures are not implemented; `inspect` reports them as unsupported.
- One process holds one model and runs one generation at a time.
