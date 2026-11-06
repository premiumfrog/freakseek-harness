---
description: "离线配置套件，供完全使用本地 GGUF 模型、不依赖任何远程服务运行 dsh 和桌面应用的用户参考。"
kind: "package-bundle"
---

# @deepseek-ai/dsh-offline

[English](README.md) | 中文

## 概述

`@deepseek-ai/dsh-offline` 是 `offline` 配置（`dsh-base`、`dsh-web-app`、`dsh-offline`）和桌面应用的最后一层。它移除所有会自行联系远程服务的行，挂载 [`dsh-llm-local-gguf`](../../llm/llm-local-gguf/README.zh.md) 作为唯一的模型路由，并用直接查询 [DuckDuckGo](../../web/web-search-duckduckgo/README.zh.md) 替代经由 DeepSeek API 的网页搜索。剩余的出站流量只由模型发起：网页搜索工具发送搜索词，网页抓取工具获取模型指定的 URL。

## 目录

- [使用本套件](#use-this-bundle)
- [移除的内容](#what-it-removes)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-bundle"></a>
## 使用本套件

```sh
dsh offline                     # the Web app on local models
dsh --profile offline --dump-config
```

将 `.gguf` 文件放入 `$DSH_HOME/models`，打开 **设置 → 本地模型**，然后选择 **加载**。加载的模型会成为新对话的默认模型；输入框的模型选择器会列出所有受支持的文件。`DSH_LOCAL_MODEL_DIRS`（以路径列表分隔符分隔）替换扫描目录，`DSH_GGUF_ENGINE` 指定引擎可执行文件，`DSH_LOCAL_MODEL` 指定加载任何模型前的默认模型 id。

如需同时移除剩余的模型发起流量，请在配置补丁中禁用网页工具：

```yaml
- id: tool-web
  disabled: true
```

<a id="what-it-removes"></a>
## 移除的内容

| 行 | 远程服务 |
|---|---|
| `llm-deepseek`、`llm-deepseek-account`、`llm-pi-ai`、`deepseek-llm-api-extensions`、`plugin-package-inventory-deepseek`、`session-log-deepseek` | DeepSeek 和第三方模型 API、请求元数据、会话日志上传 |
| `deepseek-account`、`account-controller`、`ui-settings-account` | DeepSeek 账号登录 |
| `otel`、`session-telemetry-otel`、`desktop-product-telemetry`、`product-analytics` | 遥测和产品分析 |
| `plugin-manager`、`tool-plugin-manager`、`ui-plugin-manager` | 基于 npm 的插件安装 |
| `web-search-deepseek`、`web-deepseek-api` | 经由 DeepSeek API 的网页搜索 |

桌面应用还会把自身的 Electron 会话限制在回环地址和本地资源，这同样覆盖更新检查和拼写检查词典下载。

<a id="model-experience"></a>
## 模型体验

无，本套件只选择行；本地模型路由和网页工具负责所有模型可见输入。

#### KV 缓存影响

本套件不向请求前缀添加任何内容；适用本地路由的前缀复用。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>

- 用户配置的 MCP 服务器保留各自的网络访问。
- 侧边栏浏览器标签页按用户请求浏览网页；它们使用独立分区，不受桌面离线策略限制。

<a id="dev-note"></a>
### 开发备注

无。
