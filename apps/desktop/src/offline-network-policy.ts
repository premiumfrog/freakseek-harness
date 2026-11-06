/**
 * Offline network policy for the Desktop application session.
 *
 * The application window, the update client, and spell-check dictionary
 * downloads share Electron's default session. The policy lets that session
 * reach only the loopback Host and local resources, so no Desktop surface
 * contacts a remote server. Model-initiated web search and fetch run in the
 * Host process, and sidebar browser tabs use their own partitions; neither is
 * affected.
 */

import type { Session } from 'electron'

/** URL schemes that never leave the machine. */
const LOCAL_SCHEMES = new Set(['file:', 'data:', 'blob:', 'devtools:', 'chrome:', 'chrome-extension:', 'about:'])

/** Schemes that carry network traffic and are allowed only to loopback hosts. */
const NETWORK_SCHEMES = new Set(['http:', 'https:', 'ws:', 'wss:'])

/**
 * Whether the offline policy admits one request URL.
 * @param raw - request URL.
 * @returns true for local resources and loopback hosts.
 */
export function isOfflineAllowed(raw: string): boolean {
  let url: URL
  try { url = new URL(raw) }
  catch (_unparsableUrl) { return false }
  if (LOCAL_SCHEMES.has(url.protocol)) return true
  if (!NETWORK_SCHEMES.has(url.protocol)) return false
  const host = url.hostname.toLowerCase()
  return host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' || host === '::1' || /^127(?:\.\d{1,3}){3}$/u.test(host)
}

/**
 * Cancel every request of `target` the offline policy does not admit.
 * @param target - session to confine, normally `session.defaultSession`.
 * @param onBlocked - called with each refused URL, for diagnostics.
 * @returns disposer that removes the filter.
 */
export function installOfflineNetworkPolicy(target: Session, onBlocked: (url: string) => void): () => void {
  target.webRequest.onBeforeRequest((details, callback) => {
    const allowed = isOfflineAllowed(details.url)
    if (!allowed) onBlocked(details.url)
    callback({ cancel: !allowed })
  })
  return () => { target.webRequest.onBeforeRequest(null) }
}
