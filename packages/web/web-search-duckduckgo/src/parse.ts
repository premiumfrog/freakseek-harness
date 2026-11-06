/**
 * Result extraction from DuckDuckGo's HTML search page.
 *
 * @module dsh-web-search-duckduckgo/parse
 */

import type { WebSearchSource } from '@deepseek-ai/dsh-web'

const ENTITIES: Readonly<Record<string, string>> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#x27': "'", '#39': "'" }

/**
 * Strip tags, decode entities, and collapse whitespace.
 * @param html - HTML fragment.
 * @returns plain text.
 */
export function plainText(html: string): string {
  return html
    .replace(/<[^>]*>/gu, '')
    .replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/giu, (whole, name: string) => {
      const known = ENTITIES[name.toLowerCase()]
      if (known !== undefined) return known
      if (name.startsWith('#x') || name.startsWith('#X')) return String.fromCodePoint(Number.parseInt(name.slice(2), 16))
      if (name.startsWith('#')) return String.fromCodePoint(Number.parseInt(name.slice(1), 10))
      return whole
    })
    .replace(/\s+/gu, ' ')
    .trim()
}

/**
 * Resolve DuckDuckGo's redirect link (`//duckduckgo.com/l/?uddg=…`) to its target.
 * @param href - result link as it appears in the page.
 * @returns the HTTP(S) target, or `undefined` for any other link.
 */
export function targetUrl(href: string): string | undefined {
  const decoded = plainText(href)
  let url: URL
  try { url = new URL(decoded, 'https://duckduckgo.com') }
  catch (_malformedHref) { return undefined }
  const target = url.hostname.endsWith('duckduckgo.com') ? url.searchParams.get('uddg') : url.href
  if (target === null) return undefined
  try {
    const parsed = new URL(target)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : undefined
  } catch (_malformedTarget) { return undefined }
}

/**
 * Extract organic results; ads (links through `duckduckgo.com/y.js`) are dropped.
 * @param html - search results page.
 * @returns sources in page order, de-duplicated by URL.
 */
export function parseResults(html: string): WebSearchSource[] {
  const out: WebSearchSource[] = []
  const seen = new Set<string>()
  const blocks = html.split(/<div[^>]+class="[^"]*\bresult\b[^"]*"/u).slice(1)
  for (const block of blocks) {
    if (/result--ad\b/u.test(block)) continue
    const link = /<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/u.exec(block)
      ?? /<a[^>]+href="([^"]+)"[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>([\s\S]*?)<\/a>/u.exec(block)
    if (link === null) continue
    const url = targetUrl(link[1] as string)
    if (url === undefined || seen.has(url)) continue
    seen.add(url)
    const snippet = /class="[^"]*\bresult__snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div|td)>/u.exec(block)
    const title = plainText(link[2] as string)
    const text = snippet === null ? '' : plainText(snippet[1] as string)
    out.push({ url, ...title === '' ? {} : { title }, ...text === '' ? {} : { snippet: text } })
  }
  return out
}
