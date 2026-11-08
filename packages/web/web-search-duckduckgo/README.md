---
description: "The DuckDuckGo web search provider for users who want model web search without an account, key, or intermediary service."
kind: "package-reference"
---

# @deepseek-ai/dsh-web-search-duckduckgo

## Summary

`@deepseek-ai/dsh-web-search-duckduckgo` registers the `duckduckgo` search provider on the [`dsh-web`](../web/README.md) seam. Each search posts only the query text to DuckDuckGo's HTML endpoint and parses the organic results into citeable sources; it needs no account or key and sends nothing else. The [`dsh-offline`](../../bundle/offline/README.md) bundle selects it.

## Table of Contents

- [Use this package](#use-this-package)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

```yaml
- name: '@deepseek-ai/dsh-web'
  config:
    searchProvider: duckduckgo
- name: '@deepseek-ai/dsh-web-search-duckduckgo'
```

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | `https://html.duckduckgo.com/html/` | HTML search endpoint |
| `timeoutMs` | `15000` | Per-search timeout |
| `userAgent` | `Mozilla/5.0 (compatible; freakseek)` | `User-Agent` header |

Ads and non-HTTP links are dropped, duplicates are removed, and `maxResults` bounds the result. An HTTP failure, a network error, or DuckDuckGo's bot-check page (HTTP 202, served to networks that search too often) raises `WEB_PROVIDER_ERROR`.

-----

<a id="model-experience"></a>
## Model Experience

### Web search results

#### What the model sees

`dsh-tool-web` renders the returned sources: each result's URL, title, and snippet. This provider adds no answer text.

#### Token effect

Result titles and snippets enter the tool result; their length depends on the query.

#### KV Cache effect

Search results arrive as tool results after the cached prefix and do not change earlier request content.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- Parsing depends on DuckDuckGo's HTML markup; a markup change yields no results until the parser is updated.
- DuckDuckGo may rate-limit or challenge frequent automated queries.

<a id="dev-note"></a>
### Dev Note

`tests/search.spec.ts` covers parsing and the request with a recorded results page; no test contacts DuckDuckGo.
