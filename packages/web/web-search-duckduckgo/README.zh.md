---
description: "DuckDuckGo 网页搜索提供方，供希望在无账号、无密钥、无中间服务的情况下使用模型网页搜索的用户参考。"
kind: "package-reference"
---

# @deepseek-ai/dsh-web-search-duckduckgo

[English](README.md) | 中文

## 概述

`@deepseek-ai/dsh-web-search-duckduckgo` 在 [`dsh-web`](../web/README.zh.md) 接口上注册 `duckduckgo` 搜索提供方。每次搜索只把搜索词提交到 DuckDuckGo 的 HTML 端点，并把自然结果解析为可引用的来源；它不需要账号或密钥，也不发送其他任何内容。[`dsh-offline`](../../bundle/offline/README.zh.md) 套件选用了它。

## 目录

- [使用本包](#use-this-package)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

```yaml
- name: '@deepseek-ai/dsh-web'
  config:
    searchProvider: duckduckgo
- name: '@deepseek-ai/dsh-web-search-duckduckgo'
```

| 字段 | 默认值 | 含义 |
|---|---|---|
| `endpoint` | `https://html.duckduckgo.com/html/` | HTML 搜索端点 |
| `timeoutMs` | `15000` | 单次搜索超时 |
| `userAgent` | `Mozilla/5.0 (compatible; deepseek-harness)` | `User-Agent` 请求头 |

广告和非 HTTP 链接会被丢弃，重复结果会被移除，`maxResults` 限制结果数量。HTTP 失败或网络错误会抛出 `WEB_PROVIDER_ERROR`。

-----

<a id="model-experience"></a>
## 模型体验

### 网页搜索结果

#### 模型看到的内容

`dsh-tool-web` 渲染返回的来源：每条结果的 URL、标题和摘要。本提供方不添加回答文本。

#### Token 影响

结果标题和摘要进入工具结果；其长度取决于搜索词。

#### KV 缓存影响

搜索结果作为工具结果出现在已缓存前缀之后，不会改变更早的请求内容。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>

- 解析依赖 DuckDuckGo 的 HTML 结构；结构变化后，在解析器更新前不会返回结果。
- DuckDuckGo 可能对频繁的自动化查询限流或发起验证。

<a id="dev-note"></a>
### 开发备注

`tests/search.spec.ts` 使用录制的结果页覆盖解析和请求；没有测试会联系 DuckDuckGo。
