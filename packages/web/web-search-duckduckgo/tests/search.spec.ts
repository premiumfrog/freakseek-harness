/** DuckDuckGo HTML results become citeable sources; ads and non-HTTP links are dropped. */
import { afterEach, expect, it, vi } from 'vitest'
import { Config, DuckDuckGoSearchProvider, parseResults } from '../src/index.ts'
import type { Config as ConfigShape } from '../src/index.ts'

const PAGE = `
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">Example &amp; <b>A</b></a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">The <b>first</b> result&#x27;s text</a>
</div>
<div class="result result--ad results_links">
  <a class="result__a" href="https://duckduckgo.com/y.js?ad_provider=x">Ad</a>
</div>
<div class="result results_links web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=javascript%3Aalert(1)">Bad</a>
</div>
<div class="result results_links web-result">
  <a class="result__a" href="https://other.example/b">Other</a>
</div>`

afterEach(() => { vi.restoreAllMocks() })

it('extracts organic results in page order', () => {
  expect(parseResults(PAGE)).toEqual([
    { url: 'https://example.com/a?x=1', title: 'Example & A', snippet: "The first result's text" },
    { url: 'https://other.example/b', title: 'Other' },
  ])
})

it('posts only the query and honors maxResults', async () => {
  const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(PAGE))
  const provider = new DuckDuckGoSearchProvider(Config({}) as Required<ConfigShape>)
  const result = await provider.search({ query: 'local llm', maxResults: 1 })
  expect(result).toEqual({ sources: [{ url: 'https://example.com/a?x=1', title: 'Example & A', snippet: "The first result's text" }], truncated: true })
  const [url, init] = fetch.mock.calls[0] as [string, RequestInit]
  expect(url).toBe('https://html.duckduckgo.com/html/')
  expect(init.body).toBe('q=local+llm')
})

it('reports the bot-check page as a provider error instead of an empty result', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('<div class="anomaly-modal__box">Select all squares</div>', { status: 202 }))
  await expect(new DuckDuckGoSearchProvider(Config({}) as Required<ConfigShape>).search({ query: 'x' })).rejects.toThrow(/bot check/u)
})

it('reports HTTP failures as provider errors', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('', { status: 503 }))
  await expect(new DuckDuckGoSearchProvider(Config({}) as Required<ConfigShape>).search({ query: 'x' })).rejects.toMatchObject({ code: 'WEB_PROVIDER_ERROR' })
})
