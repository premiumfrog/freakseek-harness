/**
 * The Host's control link to its shell: Node IPC when the Electron shell forked it, or
 * JSON lines over stdio when a native shell (FreakSeek's Tauri shell) spawned it.
 *
 * In stdio mode every control event is one stdout line `@freakseek:<json>`, so ordinary
 * Host logging on stdout stays distinguishable; commands arrive as one JSON object per
 * stdin line, and stdin ending counts as the shell disconnecting.
 *
 * @module dsh-desktop-host/control-channel
 */

import { createInterface } from 'node:readline'
import type { Readable, Writable } from 'node:stream'

/** Prefix of every control event line in stdio mode. */
export const STDIO_EVENT_PREFIX = '@freakseek:'

/** Environment value of `FREAKSEEK_SHELL` that selects stdio mode explicitly. */
export const STDIO_SHELL_MODE = 'stdio'

/** One bidirectional control link between the Host and its shell. */
export interface ControlChannel {
  /** `ipc` for a forked Electron child, `stdio` for a spawned native shell. */
  readonly kind: 'ipc' | 'stdio'
  /** Whether the shell is still attached. */
  readonly connected: boolean
  /**
   * Deliver one control event.
   * @param message - JSON-serializable event with a `type` field.
   * @returns settles once the event was handed to the transport; detached channels resolve without sending.
   */
  send(message: object): Promise<void>
  /**
   * Subscribe to shell commands.
   * @param listener - receives each parsed command; malformed stdio lines are dropped.
   */
  onMessage(listener: (message: unknown) => void): void
  /**
   * Subscribe to the shell going away (IPC disconnect or stdin end).
   * @param listener - called at most once.
   */
  onceDisconnect(listener: () => void): void
  /** Release the link; later sends resolve without effect. */
  disconnect(): void
}

/** Process surface the IPC channel reads; `process` satisfies it. */
export interface IpcProcess {
  readonly connected: boolean
  send?: (message: object, callback: (error: Error | null) => void) => boolean
  on(event: 'message', listener: (message: unknown) => void): unknown
  once(event: 'disconnect', listener: () => void): unknown
  disconnect?: () => void
}

/**
 * Wrap the Node IPC channel of a forked child.
 * @param proc - the current process.
 * @returns a channel over `process.send`.
 */
export function ipcChannel(proc: IpcProcess): ControlChannel {
  return {
    kind: 'ipc',
    get connected() { return proc.connected },
    send: message => new Promise((resolve, reject) => {
      if (!proc.connected || proc.send === undefined) { resolve(); return }
      proc.send(message, (error) => { if (error === null) resolve(); else reject(error) })
    }),
    onMessage: (listener) => { proc.on('message', listener) },
    onceDisconnect: (listener) => { proc.once('disconnect', listener) },
    disconnect: () => { if (proc.connected) proc.disconnect?.() },
  }
}

/**
 * Speak JSON lines over a stdin/stdout pair.
 * @param input - shell-to-Host command stream.
 * @param output - Host-to-shell event stream.
 * @returns a channel that prefixes each event line with {@link STDIO_EVENT_PREFIX}.
 */
export function stdioChannel(input: Readable, output: Writable): ControlChannel {
  let connected = true
  const disconnects: (() => void)[] = []
  const lines = createInterface({ input, crlfDelay: Infinity })
  const detach = (): void => {
    if (!connected) return
    connected = false
    lines.close()
    for (const listener of disconnects.splice(0)) listener()
  }
  lines.once('close', detach)
  return {
    kind: 'stdio',
    get connected() { return connected },
    send: message => new Promise((resolve, reject) => {
      if (!connected) { resolve(); return }
      output.write(`${STDIO_EVENT_PREFIX}${JSON.stringify(message)}\n`, (error) => { if (error == null) resolve(); else reject(error) })
    }),
    onMessage: (listener) => {
      lines.on('line', (line) => {
        const text = line.trim()
        if (text === '') return
        let value: unknown
        try { value = JSON.parse(text) }
        catch (_malformedCommand) {
          // The shell is the only writer; a malformed line is dropped rather than ending the Host.
          return
        }
        listener(value)
      })
    },
    onceDisconnect: (listener) => {
      if (!connected) { listener(); return }
      disconnects.push(listener)
    },
    disconnect: detach,
  }
}

/**
 * Choose the control link for this process.
 * @param proc - the current process (IPC surface plus stdio streams).
 * @param env - launch environment, read for `FREAKSEEK_SHELL`.
 * @returns stdio when `FREAKSEEK_SHELL=stdio` or no IPC channel exists, otherwise IPC.
 */
export function selectControlChannel(
  proc: IpcProcess & { readonly stdin: Readable; readonly stdout: Writable },
  env: Readonly<Record<string, string | undefined>>,
): ControlChannel {
  return env.FREAKSEEK_SHELL === STDIO_SHELL_MODE || !proc.connected ? stdioChannel(proc.stdin, proc.stdout) : ipcChannel(proc)
}
