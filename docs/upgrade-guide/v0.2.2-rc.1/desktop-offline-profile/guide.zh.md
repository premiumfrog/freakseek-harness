---
kind: upgrade-guide
description: "桌面应用现在只运行本地 GGUF 模型，不再提供 DeepSeek 或第三方模型提供方、账号登录、遥测和插件安装。"
---

# 桌面应用只使用本地模型

[English](guide.md) | 中文

## 变更

桌面配置在 `dsh-base` 和 `dsh-web-app` 之后叠加新的 `@deepseek-ai/dsh-offline` 套件。它禁用所有会自行联系远程服务的行：DeepSeek 和 pi-ai 模型路由、DeepSeek 请求元数据和会话日志上传、账号登录、遥测和产品分析、插件安装，以及经由 DeepSeek API 的网页搜索。模型来自通过 `@deepseek-ai/dsh-llm-local-gguf` 读取的 GGUF 文件，网页搜索直接访问 DuckDuckGo。欢迎页不再要求登录 DeepSeek 或填写 API 密钥，应用自身的 Electron 会话只能访问回环地址和本地资源。

此前在桌面应用中使用 DeepSeek 或其他托管提供方的用户会受到影响：这些提供方不再出现。之前记录的会话仍可阅读，但继续对话时下一轮会发送给所选的本地模型。`web` 配置和 npm CLI 不变；`dsh offline` 在桌面应用之外运行相同的组合。

## 迁移

1. 将一个或多个 `.gguf` 对话模型复制到 `$DSH_HOME/models`（默认 `~/.dsh/models`）。
2. 打开 **设置 → 本地模型**，在某个模型上选择 **加载**。它会成为新对话的默认模型。
3. 如需继续使用托管提供方，请用 `dsh web` 运行 `web` 配置，而不是桌面应用。

发送一条消息即可确认迁移：在断开网络时回复仍能流式输出，并且 **设置 → 本地模型** 显示该模型已加载。
