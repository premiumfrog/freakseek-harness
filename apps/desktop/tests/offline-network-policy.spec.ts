/** The Desktop session admits loopback and local resources only. */
import { expect, it, vi } from 'vitest'
import { installOfflineNetworkPolicy, isOfflineAllowed } from '../src/offline-network-policy.ts'

it.each([
  'http://127.0.0.1:4317/', 'ws://127.0.0.1:4317/rpc', 'https://localhost/x', 'http://app.localhost/', 'http://[::1]:80/',
  'file:///Applications/x/index.html', 'data:text/plain,hi', 'blob:http://127.0.0.1/abc', 'devtools://devtools/bundled/x',
])('allows %s', (url) => {
  expect(isOfflineAllowed(url)).toBe(true)
})

it.each([
  'https://download.deepseek.com/desktop/latest.yml', 'https://api.deepseek.com/anthropic', 'http://192.168.1.2/',
  'https://127.0.0.1.example.com/', 'ftp://127.0.0.1/', 'not a url',
])('blocks %s', (url) => {
  expect(isOfflineAllowed(url)).toBe(false)
})

it('cancels refused requests through the session filter and removes it on dispose', () => {
  let listener: ((details: { url: string }, callback: (response: { cancel: boolean }) => void) => void) | null = null
  const session = { webRequest: { onBeforeRequest: vi.fn((next: typeof listener) => { listener = next }) } }
  const blocked: string[] = []
  const dispose = installOfflineNetworkPolicy(session as never, (url) => { blocked.push(url) })
  const answers: boolean[] = []
  listener!({ url: 'https://example.com/' }, (r) => { answers.push(r.cancel) })
  listener!({ url: 'http://127.0.0.1:1/' }, (r) => { answers.push(r.cancel) })
  expect(answers).toEqual([true, false])
  expect(blocked).toEqual(['https://example.com/'])
  dispose()
  expect(session.webRequest.onBeforeRequest).toHaveBeenLastCalledWith(null)
})
