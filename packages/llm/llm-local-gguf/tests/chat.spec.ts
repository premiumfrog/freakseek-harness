/** Terminal chat helpers: reasoning separation across split tags, and the per-reply figures line. */

import { expect, it } from 'vitest'
import { ReasoningSplitter, replyStats } from '../src/chat.ts'

it('separates reasoning from the answer across deltas that split the tags', () => {
  const splitter = new ReasoningSplitter(true)
  const pieces = [...splitter.push('thinking a'), ...splitter.push('bout it</th'), ...splitter.push('ink>\n\nFour.'), ...splitter.finish()]
  expect(pieces.filter(p => p.reasoning).map(p => p.text).join('')).toBe('thinking about it')
  expect(pieces.filter(p => !p.reasoning).map(p => p.text).join('')).toBe('\n\nFour.')
  const open = new ReasoningSplitter(false)
  const later = [...open.push('<thi'), ...open.push('nk>why</think>ok<'), ...open.finish()]
  expect(later).toEqual([{ reasoning: true, text: 'why' }, { reasoning: false, text: 'ok' }, { reasoning: false, text: '<' }])
})

it('reports speed, prompt reuse, context use, and why a reply stopped early', () => {
  const outcome = {
    finishReason: 'repetition' as const, promptTokens: 2400, cachedTokens: 2000, completionTokens: 64,
    prefillMs: 100, decodeMs: 2000, promptTokensPerSecond: 4000, tokensPerSecond: 32,
  }
  expect(replyStats(outcome, 32768)).toBe(
    '32.0 tok/s · 64 tokens · prompt 2.4k (2.0k cached, 4000 tok/s) · context 2.5k/32.8k · stopped: the reply repeated itself',
  )
})
