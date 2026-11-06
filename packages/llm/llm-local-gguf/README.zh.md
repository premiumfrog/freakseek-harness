---
description: "离线本地模型提供方，供在本机通过内置 Rust 引擎运行 GGUF 模型的用户和维护者参考。"
kind: "package-reference"
---

# @deepseek-ai/dsh-llm-local-gguf

[English](README.md) | 中文

## 概述

`@deepseek-ai/dsh-llm-local-gguf` 使用磁盘上的 GGUF 文件提供 `local` 模型路由。请求在由 [`native/gguf-engine`](../../../native/gguf-engine/README.zh.md) 构建的 `dsh-gguf-engine` 子进程中运行：它读取模型文件，渲染模型自带的对话模板，并通过标准输入输出流式返回 token。引擎不打开任何套接字，也不发起网络请求，因此该路由上的对话不会离开本机。本包还提供 `localModels` Remote，本地模型设置页通过它列出、加载和卸载模型。

## 目录

- [使用本包](#use-this-package)
- [了解实现](#understand-the-implementation)
- [延伸阅读](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

与 `@deepseek-ai/dsh-llm` 一起挂载。[`dsh-offline`](../../bundle/offline/README.zh.md) 套件为 `offline` 配置和桌面应用完成了这一步：

```yaml
- name: '@deepseek-ai/dsh-llm'
- name: '@deepseek-ai/dsh-llm-local-gguf'
  config:
    modelDirectories: ['~/models']
    contextLength: 32768
```

将 `.gguf` 文件放入模型目录；默认目录为 `$DSH_HOME/models`，并向下扫描两级子目录。每个受支持的文件都会成为一个模型，其 id 为相对目录的路径去掉 `.gguf`，例如 `qwen/qwen2.5-7b-instruct-q4_k_m`。`listModels('local')` 会重新扫描目录并检查新增或变化的文件；分片文件的后续分片和多模态投影文件会被跳过。

对某个模型的首次请求会加载它；请求另一个模型会替换当前模型，因为一个引擎进程同一时间只持有一个模型。`localModels` Remote 提供 `snapshot()`、`load(modelId)` 和 `unload()`；组合了 `@deepseek-ai/dsh-agent-default-model` 时，加载成功还会把该模型保存为新对话的默认模型。

### 配置

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enginePath` | 自动解析 | 引擎可执行文件。省略时依次使用 `$DSH_GGUF_ENGINE`、桌面应用内的 `<resources>/gguf-engine/`、源码检出中的 `native/gguf-engine/target/release/` |
| `modelDirectories` | `$DSH_HOME/models` | 扫描 GGUF 文件的目录；`~` 展开为用户主目录，空列表表示使用默认目录 |
| `contextLength` | `32768` | 加载时请求的上下文窗口，不超过模型训练长度；仅智能体的系统提示和工具声明就约占 1 万个 token |
| `cacheSlots` | `4` | 同时保留的 KV 缓存数；每个对话和每类辅助请求（如会话标题）各用一个，最久未使用的先被淘汰 |
| `maxOutputTokens` | `4096` | 请求未指定时的输出上限 |
| `threads` | `0` | 推理线程数；`0` 使用全部核心 |
| `temperature`、`topK`、`topP`、`minP`、`repeatPenalty` | `0.6`、`40`、`0.95`、`0.05`、`1` | 采样参数；请求中的 `temperature` 覆盖配置值 |
| `shutdownGraceMillis` | `3000` | 插件停止时引擎退出的等待时间 |

### 支持的模型

引擎支持 `llama`（Llama 2 和 3、Mistral、DeepSeek-R1 Llama 蒸馏）、`qwen2`（Qwen 2 和 2.5、DeepSeek-R1 Qwen 蒸馏）和 `qwen3` 架构，权重类型为 F32、F16、BF16、Q4_0、Q4_1、Q5_0、Q5_1、Q8_0 以及 Q2_K 到 Q8_K。其他架构或 I-quant 权重的文件会在设置页中标为不受支持，并显示原因。

### 失败

引擎失败以 `LlmError` 代码呈现：渲染后的提示超出已加载窗口时为 `CONTEXT_WINDOW_EXCEEDED`，目录中没有该 id 时为 `LOCAL_UNKNOWN_MODEL`，以及 `LOCAL_UNSUPPORTED_MODEL`；找不到可执行文件时为 `LOCAL_ENGINE_UNAVAILABLE`，进程退出时为 `LOCAL_ENGINE_EXITED`，下一次请求会启动新进程。取消会在下一个 token 处停止生成，并以中止结束流。

-----

<a id="understand-the-implementation"></a>
## 了解实现

<details>
<summary>实现细节 — 点击展开</summary>

### 源码地图

| 文件 | 作用 |
|---|---|
| [`src/index.ts`](src/index.ts) | 插件入口：注册 `local` 路由、可配置提供方条目和 `localModels` Remote |
| [`src/config.ts`](src/config.ts) | 配置模式，以及显式的引擎路径和目录解析 |
| [`src/engine-process.ts`](src/engine-process.ts) | 单个引擎进程的 JSON 行客户端，带带外取消 |
| [`src/runtime.ts`](src/runtime.ts) | 进程、目录扫描和已加载模型状态的所有者；一个队列为加载和生成排序 |
| [`src/request.ts`](src/request.ts) | 将 harness 历史转换为 OpenAI 风格的模板消息 |
| [`src/output-parser.ts`](src/output-parser.ts) | 从原始模型文本中增量提取推理和工具调用 |
| [`src/adapter.ts`](src/adapter.ts) | `LlmAdapter` 实现 |
| [`src/service.ts`](src/service.ts) | `localModels` Remote |

### 请求转换

开头的 system 消息合并为一条 `system` 消息，因为大多数 GGUF 对话模板只读取开头那一条。后续的 system 和 developer 文本以 user 轮次传递。助手推理作为 `reasoning_content` 传递，工具调用作为参数已解析为对象的 `tool_calls` 传递，这是 Hugging Face 模板期望的形式。工具结果成为以对应调用命名的 `tool` 消息。图片变为文本占位符，因为该路由声明只接受文本输入。

### 输出解析

引擎流式输出原始文本。解析器识别 `<think>…</think>` 推理（包括从模板已打开的推理块内部开始的输出）、Hermes 和 Qwen 的 `<tool_call>{json}</tool_call>` 调用、Mistral 的 `[TOOL_CALLS][…]` 数组，以及由单个指向已声明工具的 `{"name", "parameters"}` 对象构成的 Llama 3 回复。可能仍会构成标记的文本会被保留到确定为止，因此流的分段方式不会改变结果。

### 提示缓存

引擎为每个槽位保留一份 KV 缓存，并复用该槽位下一次提示与上一次共享的最长 token 前缀。适配器按会话 id 和请求用途划分槽位，因此会话标题或压缩请求不会挤掉对话缓存。由于 harness 会原样重发对话历史，后续轮次只需计算新增 token；复用的位置以 `cacheReadTokens` 报告。

</details>

-----

<a id="further-exploration"></a>
## 延伸阅读

- [`native/gguf-engine`](../../../native/gguf-engine/README.zh.md) — 推理引擎及其标准输入输出协议。
- [`dsh-offline`](../../bundle/offline/README.zh.md) — 挂载本提供方且不含任何远程服务的配置套件。
- [`dsh-client-ui-settings-local-models`](../../client/ui-settings-local-models/README.zh.md) — 基于 `localModels` Remote 的设置页。
- [LLM 流式子系统](../../../docs/subsystems/llm-streaming.zh.md) — 本适配器输出的分块协议。

-----

<a id="model-experience"></a>
## 模型体验

### 本地模型请求

#### 模型看到的内容

请求历史和工具声明通过模型文件自带的 `tokenizer.chat_template` 渲染；文件没有模板或模板渲染失败时，使用带 Hermes 风格工具说明的 ChatML 模板。图片显示为文本 `[image omitted: this local model reads text only]`。

#### Token 影响

每个渲染字符都由模型自己的分词器计数；工具声明按模板放置的位置把 JSON 模式加入提示。用量报告未缓存的提示 token、复用的前缀 token 和生成的 token。

#### KV 缓存影响

引擎复用与同一对话、同一请求用途的上一次请求共享的最长 token 前缀。加载其他模型、系统提示或工具列表变化，或在超过 `cacheSlots` 个其他对话后被淘汰，都会从第一个不同的 token 起放弃复用。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>

- 一个引擎进程只持有一个模型；不同模型上的并发对话每次切换都会重新加载权重，并发请求依次执行。
- 推理在 CPU 上运行，尚未实现 GPU 卸载。
- 不支持分片 GGUF 文件和 I-quant 权重类型。
- 依赖引擎 minijinja 方言之外 Jinja 特性的模板会回退到 ChatML。
- 工具调用依赖模型遵循其模板的工具格式；格式错误的调用会以文本形式保留。

<a id="dev-note"></a>
### 开发备注

`tests/loader-composition.spec.ts` 通过 Loader 启动该路由，并在 `tests/gguf-fixture.ts` 写出的合成二元模型上运行真实引擎；`pnpm run test` 会先用 `pnpm run build:gguf-engine` 构建引擎。
