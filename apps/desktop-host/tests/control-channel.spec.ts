/** The Host's shell link: Node IPC for Electron, JSON lines over stdio for the native shell. */
import { PassThrough } from 'node:stream'
import { describe, expect, it, vi } from 'vitest'
import { ipcChannel, selectControlChannel, stdioChannel, STDIO_EVENT_PREFIX, type IpcProcess } from '../src/control-channel.ts'

function streams() {
  const input = new PassThrough()
  const output = new PassThrough()
  const lines: string[] = []
  output.setEncoding('utf8')
  output.on('data', (chunk: string) => { lines.push(...chunk.split('\n').filter(line => line !== '')) })
  return { input, output, lines }
}

function fakeIpc(connected: boolean): IpcProcess & { sent: object[] } {
  const sent: object[] = []
  return {
    connected,
    sent,
    send: (message, callback) => { sent.push(message); callback(null); return true },
    on: () => undefined,
    once: () => undefined,
    disconnect: vi.fn(),
  }
}

describe('stdio channel', () => {
  it('writes one prefixed JSON line per event', async () => {
    const { input, output, lines } = streams()
    const channel = stdioChannel(input, output)
    await channel.send({ type: 'starting', runId: 'r' })
    await channel.send({ type: 'ready', url: 'http://127.0.0.1:1/?token=t' })
    expect(lines).toEqual([
      `${STDIO_EVENT_PREFIX}{"type":"starting","runId":"r"}`,
      `${STDIO_EVENT_PREFIX}{"type":"ready","url":"http://127.0.0.1:1/?token=t"}`,
    ])
  })

  it('parses command lines, drops malformed ones, and reports stdin end as a disconnect', async () => {
    const { input, output } = streams()
    const channel = stdioChannel(input, output)
    const commands: unknown[] = []
    const disconnected = vi.fn()
    channel.onMessage((message) => { commands.push(message) })
    channel.onceDisconnect(disconnected)
    input.write('{"type":"quit-inspection","requestId":3}\nnot json\n\n{"type":"shutdown"}\n')
    input.end()
    await vi.waitFor(() => { expect(disconnected).toHaveBeenCalledOnce() })
    expect(commands).toEqual([{ type: 'quit-inspection', requestId: 3 }, { type: 'shutdown' }])
    expect(channel.connected).toBe(false)
    await channel.send({ type: 'shutdown-complete' })
  })

  it('sends nothing after an explicit disconnect', async () => {
    const { input, output, lines } = streams()
    const channel = stdioChannel(input, output)
    channel.disconnect()
    await channel.send({ type: 'ready', url: 'x' })
    expect(lines).toEqual([])
  })
})

describe('channel selection', () => {
  it('uses IPC for a forked child and stdio when asked or when no IPC exists', () => {
    const { input, output } = streams()
    expect(selectControlChannel({ ...fakeIpc(true), stdin: input, stdout: output }, {}).kind).toBe('ipc')
    expect(selectControlChannel({ ...fakeIpc(true), stdin: input, stdout: output }, { FREAKSEEK_SHELL: 'stdio' }).kind).toBe('stdio')
    expect(selectControlChannel({ ...fakeIpc(false), stdin: input, stdout: output }, {}).kind).toBe('stdio')
  })

  it('keeps IPC delivery through process.send', async () => {
    const proc = fakeIpc(true)
    await ipcChannel(proc).send({ type: 'ready', url: 'u' })
    expect(proc.sent).toEqual([{ type: 'ready', url: 'u' }])
    await ipcChannel(fakeIpc(false)).send({ type: 'ignored' })
  })
})
