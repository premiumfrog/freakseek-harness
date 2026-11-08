/**
 * Web search provider that queries DuckDuckGo's HTML endpoint directly, with
 * no API key and no intermediary service. Only the query text leaves the
 * machine, and only when the model calls the web search tool.
 *
 * @module dsh-web-search-duckduckgo
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { WebError } from '@deepseek-ai/dsh-web'
import type { WebSearchProvider, WebSearchRequest, WebSearchResult } from '@deepseek-ai/dsh-web'
import { parseResults } from './parse.ts'

export { parseResults, plainText, targetUrl } from './parse.ts'

/** Provider id selected by `web.searchProvider`. */
export const DUCKDUCKGO_PROVIDER_ID = 'duckduckgo'

export const name = 'web-search-duckduckgo'
export const inject = ['web']

/** Plugin configuration. */
export interface Config {
  /** HTML search endpoint. */
  endpoint?: string
  /** Per-search timeout in milliseconds. */
  timeoutMs?: number
  /** `User-Agent` header sent with each search. */
  userAgent?: string
}

/** Configuration after schemastery applied every default. */
type ResolvedConfig = Required<Config>

export const Config: z<Config> = z.object({
  endpoint: z.string().default('https://html.duckduckgo.com/html/'),
  timeoutMs: z.number().step(1).min(1).max(600_000).default(15_000),
  userAgent: z.string().default('Mozilla/5.0 (compatible; freakseek)'),
})

/** The provider over one validated configuration. */
export class DuckDuckGoSearchProvider implements WebSearchProvider {
  readonly id = DUCKDUCKGO_PROVIDER_ID
  readonly title = 'DuckDuckGo'
  readonly description = {
    en: 'Searches DuckDuckGo directly. No account or key; only the search query is sent.',
    zh: '直接搜索 DuckDuckGo，无需账号或密钥，仅发送搜索词。',
  }

  /** @param config - validated configuration with defaults applied. */
  constructor(private readonly config: ResolvedConfig) {}

  available(): boolean {
    return true
  }

  async search(request: WebSearchRequest, signal?: AbortSignal): Promise<WebSearchResult> {
    const timeout = AbortSignal.timeout(this.config.timeoutMs)
    const combined = signal === undefined ? timeout : AbortSignal.any([signal, timeout])
    let response: Response
    try {
      response = await fetch(this.config.endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded', 'user-agent': this.config.userAgent },
        body: new URLSearchParams({ q: request.query }).toString(),
        signal: combined,
      })
    } catch (error) {
      if (signal?.aborted === true) throw error
      throw new WebError(`duckduckgo: search request failed: ${(error as Error).message}`, 'WEB_PROVIDER_ERROR', { cause: error })
    }
    if (!response.ok) throw new WebError(`duckduckgo: search returned HTTP ${response.status}`, 'WEB_PROVIDER_ERROR')
    const html = await response.text()
    // DuckDuckGo answers rate-limited clients with HTTP 202 and a bot-check page instead of results.
    if (response.status === 202 || html.includes('anomaly-modal')) {
      throw new WebError('duckduckgo: DuckDuckGo showed its bot check instead of results (too many searches from this network); try again later', 'WEB_PROVIDER_ERROR')
    }
    const sources = parseResults(html)
    const limited = request.maxResults === undefined ? sources : sources.slice(0, request.maxResults)
    return { sources: limited, truncated: limited.length < sources.length }
  }
}

/**
 * Register the DuckDuckGo search provider.
 * @param ctx - Host context.
 * @param config - validated configuration.
 */
export function apply(ctx: Context, config: Config): void {
  ctx.web.registerSearchProvider(new DuckDuckGoSearchProvider(config as ResolvedConfig))
}
