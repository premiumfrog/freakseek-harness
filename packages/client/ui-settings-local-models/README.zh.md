---
description: "本地模型设置分区，供在 Web 和桌面客户端中选择、加载和卸载 GGUF 模型的用户参考。"
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-settings-local-models

[English](README.md) | 中文

## 概述

`@deepseek-ai/dsh-client-ui-settings-local-models` 为 Web 和桌面客户端添加 **设置 → 本地模型**。该分区列出 Host 上 [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.zh.md) 提供方找到的每个 GGUF 文件，并显示其架构、大小、量化方式以及工具和推理支持；**加载** 会把模型载入本地引擎并设为新对话的默认模型，**卸载** 会释放其内存。页面还会显示扫描的模型文件夹以及引擎或文件夹问题。

## 目录

- [使用本包](#use-this-package)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

[`dsh-offline`](../../bundle/offline/README.zh.md) 套件挂载这一行。该分区读取 `localModels` Remote，并在 `local` 路由的 `llm/models-updated` 事件时刷新，因此在其他窗口中开始的加载或重新扫描会更新所有打开的页面。没有该提供方的部署会显示不提供本地模型。

-----

<a id="model-experience"></a>
## 模型体验

无，该页面只改变由哪个模型服务新对话，不构造模型输入。

#### KV 缓存影响

加载其他模型会丢弃引擎的已缓存前缀。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>

- 模型文件夹在配置（`modelDirectories`）中编辑，而不是在页面上编辑。

<a id="dev-note"></a>
### 开发备注

无。
