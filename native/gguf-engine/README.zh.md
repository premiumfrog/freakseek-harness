---
description: "离线 GGUF 推理引擎，供构建、测试或扩展 `local` 模型路由背后本地模型运行时的用户和维护者参考。"
kind: "package-reference"
---

# dsh-gguf-engine

[English](README.md) | 中文

## 概述

`dsh-gguf-engine` 是在 CPU 上运行 GGUF 对话模型的 Rust 可执行文件。它解析 GGUF 头部，根据 `tokenizer.ggml.*` 元数据重建分词器，渲染文件自带的 Jinja 对话模板，在内存中保持权重量化，并使用可复用前缀的 KV 缓存生成 token。宿主每行向标准输入写入一个 JSON 对象，并从标准输出每行读取一个 JSON 对象（[协议](PROTOCOL.md)）。引擎不打开任何套接字：依赖树中没有网络库，唯一的 I/O 是模型文件和两条管道。其宿主是 [`@deepseek-ai/dsh-llm-local-gguf`](../../packages/llm/llm-local-gguf/README.zh.md)。

## 目录

- [构建与测试](#build-and-test)
- [支持的模型](#supported-models)
- [了解实现](#understand-the-implementation)
- [已知限制与后续工作](#known-limitations-and-deferred-work)

-----

<a id="build-and-test"></a>
## 构建与测试

```sh
pnpm run build:gguf-engine          # cargo build --release --locked
cd native/gguf-engine && cargo test --release
```

发布版二进制位于 `native/gguf-engine/target/release/dsh-gguf-engine`（Windows 上为 `.exe`），源码检出从这里找到它。桌面打包会把它复制到应用资源中的 `gguf-engine/dsh-gguf-engine`。

测试覆盖分词一致性、前向计算一致性和协议：

- `tests/model_reference.rs` 写出随机的 `llama`、共享输出层的 `llama`、`qwen2` 和 `qwen3` 模型，并在预填充、增量解码和 KV 缓存前缀复用上，将引擎的 logits 与 candle-transformers 的参考实现比较。
- 当 `DSH_GGUF_VOCAB_DIR` 指向包含 llama.cpp `models/ggml-vocab-*.gguf{,.inp,.out}` 文件的目录时，`tests/tokenizer_reference.rs` 会把分词结果与 llama.cpp `test-tokenizer-0` 的词表（`llama-spm`、`llama-bpe`、`qwen2`、`phi-3`、`gpt-2`）比较；否则输出提示并跳过。
- `tests/protocol.rs` 驱动构建出的二进制，依次执行 hello、inspect、load、带缓存复用的 generate、cancel、unload 和 shutdown。

`--threads <n>` 限制推理线程数；`--version` 输出引擎和协议版本。

<a id="supported-models"></a>
## 支持的模型

| `general.architecture` | 模型 | 说明 |
|---|---|---|
| `llama`、`mistral` | Llama 2、Llama 3.x、Mistral、DeepSeek-R1 Llama 蒸馏 | 相邻对 RoPE；Llama 3.1 `rope_freqs` 缩放 |
| `qwen2` | Qwen 2、Qwen 2.5、DeepSeek-R1 Qwen 蒸馏 | Q/K/V 偏置；前后半分 RoPE |
| `qwen3` | Qwen 3 稠密模型 | 逐头 Q/K RMS 归一化 |

权重类型为 candle 量化内核可加载的类型：F32、F16、BF16、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0、Q8_1 以及 Q2_K 到 Q8_K。分词器支持 SentencePiece 分数合并（`llama`）和按 `tokenizer.ggml.pre` 选择预分词器的字节级 BPE（`gpt2`）。

<a id="understand-the-implementation"></a>
## 了解实现

| 文件 | 作用 |
|---|---|
| [`src/main.rs`](src/main.rs) | JSON 行协议：读取线程应答 `hello`、`inspect`、`cancel` 和 `shutdown`；工作线程按顺序执行 `load`、`unload`、`status` 和 `generate` |
| [`src/engine.rs`](src/engine.rs) | 头部检查、提示准备、前缀复用、生成循环和停止串 |
| [`src/model.rs`](src/model.rs) | 基于 candle 张量的 Transformer 前向计算；可增长、可截断的 KV 缓存 |
| [`src/tokenizer.rs`](src/tokenizer.rs) | GGUF 分词器：原子特殊 token、按堆排序的 SentencePiece 合并、按秩 BPE 和增量 UTF-8 解码 |
| [`src/template.rs`](src/template.rs) | 用 minijinja 渲染 `tokenizer.chat_template`，并回退到 ChatML |
| [`src/sampler.rs`](src/sampler.rs) | 重复惩罚、温度、top-k、top-p 和 min-p 采样 |

预填充以每块 256 个 token 并配合因果掩码运行，从而限制注意力矩阵大小。分组查询注意力按键值头折叠查询头，而不是复制键和值。嵌入行按需反量化，因此大词表不需要一份全精度的嵌入表副本。

<a id="known-limitations-and-deferred-work"></a>
## 已知限制与后续工作

- 推理只在 CPU 上运行；未启用 candle 的 CUDA 和 Metal 后端。
- KV 缓存为 F32，大模型的长窗口需要大量内存。
- 未实现混合专家、Gemma、Phi 等架构；`inspect` 会将其报告为不受支持。
- 一个进程只持有一个模型，并且同一时间只执行一次生成。
