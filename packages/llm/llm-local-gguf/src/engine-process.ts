/**
 * Client for one `dsh-gguf-engine` child process speaking JSON lines over stdio.
 *
 * The engine opens no sockets; this client is its only channel. Requests are
 * serialized by the engine's worker thread, while `cancel` is answered out of
 * band so a running generation can stop at its next token.
 *
 * @module dsh-llm-local-gguf/engine-process
 */

import { spawn } from 'node:child_process'
import type { ChildProcessWithoutNullStreams } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { EngineDelta, EngineFailure, EngineGenerateRequest, EngineLoaded, EngineModelFacts, EngineOutcome, EngineStart } from './types.ts'

/** Wire protocol revision this client speaks. */
export const ENGINE_PROTOCOL_VERSION = 1

/** Engine-reported failure, keeping the engine's stable code. */
export class EngineError extends Error {
  override name = 'EngineError'
  /**
   * @param code - stable engine failure code such as `CONTEXT_WINDOW_EXCEEDED`.
   * @param message - engine diagnostic.
   */
  constructor(readonly code: string, message: string) {
    super(message)
  }
}

interface Pending {
  resolve(value: unknown): void
  reject(error: Error): void
  onEvent: ((line: EngineLine) => void) | undefined
}

/** One stdout line of the engine protocol. */
type EngineLine =
  | ({ id: string; type: 'start' } & EngineStart)
  | ({ id: string; type: 'delta' } & EngineDelta)
  | { id: string | null; type: 'result'; ok: true; value: unknown }
  | { id: string | null; type: 'result'; ok: false; error: EngineFailure }

/** Launch facts for the engine binary. */
export interface EngineLaunch {
  /** Absolute engine executable path. */
  readonly executable: string
  /** Inference threads; omitted lets the engine use every available core. */
  readonly threads?: number
}

/** Streaming callbacks of one generation. */
export interface GenerateHandlers {
  onStart(start: EngineStart): void
  onDelta(text: string): void
}

/**
 * One running engine process. Construction spawns it; `close()` ends it. A
 * process that exits unexpectedly rejects every pending request with
 * `ENGINE_EXITED`, and the owner creates a new client for later work.
 */
export class EngineProcess {
  private readonly child: ChildProcessWithoutNullStreams
  private readonly pending = new Map<string, Pending>()
  private nextId = 0
  private exited: EngineError | undefined
  private stderrTail = ''
  /** Settles once the process has exited. */
  readonly exit: Promise<void>

  /**
   * @param launch - executable and thread count.
   */
  constructor(launch: EngineLaunch) {
    const args = launch.threads === undefined ? [] : ['--threads', String(launch.threads)]
    this.child = spawn(launch.executable, args, { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true })
    const lines = createInterface({ input: this.child.stdout })
    lines.on('line', (line) => { this.dispatch(line) })
    this.child.stderr.setEncoding('utf8')
    this.child.stderr.on('data', (chunk: string) => { this.stderrTail = (this.stderrTail + chunk).slice(-2000) })
    this.exit = new Promise<void>((resolve) => {
      const fail = (detail: string) => {
        this.exited ??= new EngineError('ENGINE_EXITED', `local model engine exited (${detail})${this.stderrTail === '' ? '' : `: ${this.stderrTail.trim()}`}`)
        for (const pending of this.pending.values()) pending.reject(this.exited)
        this.pending.clear()
        resolve()
      }
      this.child.once('error', (error) => { fail(error.message) })
      this.child.once('exit', (code, signal) => { fail(signal === null ? `code ${String(code)}` : `signal ${signal}`) })
    })
    // A write after exit raises EPIPE on the stream; `exit` already reports the failure.
    this.child.stdin.on('error', (_closedPipe) => {})
  }

  /** Whether the process is still running. */
  get alive(): boolean {
    return this.exited === undefined
  }

  private dispatch(raw: string): void {
    let line: EngineLine
    try { line = JSON.parse(raw) as EngineLine }
    catch (_notJson) { return }
    if (line.id === null) return
    const pending = this.pending.get(line.id)
    if (pending === undefined) return
    if (line.type !== 'result') {
      pending.onEvent?.(line)
      return
    }
    this.pending.delete(line.id)
    if (line.ok) pending.resolve(line.value)
    else pending.reject(new EngineError(line.error.code, line.error.message))
  }

  private request<T>(command: Record<string, unknown>, onEvent?: (line: EngineLine) => void): { id: string; result: Promise<T> } {
    const id = String(++this.nextId)
    if (this.exited !== undefined) return { id, result: Promise.reject(this.exited) }
    const result = new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: (value) => { resolve(value as T) }, reject, onEvent })
    })
    this.child.stdin.write(`${JSON.stringify({ id, ...command })}\n`)
    return { id, result }
  }

  /**
   * Check the engine's protocol revision.
   * @returns resolves when the engine speaks {@link ENGINE_PROTOCOL_VERSION}.
   */
  async handshake(): Promise<void> {
    const hello = await this.request<{ protocol: number }>({ op: 'hello' }).result
    if (hello.protocol !== ENGINE_PROTOCOL_VERSION) {
      throw new EngineError('ENGINE_PROTOCOL', `local model engine speaks protocol ${hello.protocol}; this harness requires ${ENGINE_PROTOCOL_VERSION}`)
    }
  }

  /**
   * Read a GGUF header without loading weights.
   * @param path - absolute GGUF path.
   * @returns header facts.
   */
  inspect(path: string): Promise<EngineModelFacts> {
    return this.request<EngineModelFacts>({ op: 'inspect', path }).result
  }

  /**
   * Load a model, replacing any loaded one.
   * @param path - absolute GGUF path.
   * @param contextLength - requested context window in tokens.
   * @param slots - KV cache slots the engine keeps.
   * @returns loaded model facts.
   */
  load(path: string, contextLength: number, slots: number): Promise<EngineLoaded> {
    return this.request<EngineLoaded>({ op: 'load', path, contextLength, slots }).result
  }

  /**
   * Release the loaded model's memory.
   * @returns resolves once released.
   */
  async unload(): Promise<void> {
    await this.request({ op: 'unload' }).result
  }

  /**
   * Generate one reply. Aborting `signal` cancels at the next token and the
   * promise resolves with `finishReason: 'cancelled'`.
   * @param request - messages, tools, and sampling controls.
   * @param handlers - start and delta callbacks.
   * @param signal - cancellation.
   * @returns generation outcome.
   */
  async generate(request: EngineGenerateRequest, handlers: GenerateHandlers, signal?: AbortSignal): Promise<EngineOutcome> {
    const { id, result } = this.request<EngineOutcome>({ op: 'generate', ...request }, (line) => {
      if (line.type === 'start') handlers.onStart(line)
      else if (line.type === 'delta') handlers.onDelta(line.text)
    })
    const cancel = () => { if (this.alive) this.child.stdin.write(`${JSON.stringify({ id: `c${id}`, op: 'cancel', target: id })}\n`) }
    if (signal?.aborted === true) cancel()
    signal?.addEventListener('abort', cancel, { once: true })
    try { return await result }
    finally { signal?.removeEventListener('abort', cancel) }
  }

  /**
   * End the process: request shutdown, then kill it if it has not exited after `graceMillis`.
   * @param graceMillis - wait before killing.
   * @returns resolves after exit.
   */
  async close(graceMillis: number): Promise<void> {
    if (this.alive) this.child.stdin.end(`${JSON.stringify({ id: 'shutdown', op: 'shutdown' })}\n`)
    const timer = setTimeout(() => { this.child.kill('SIGKILL') }, graceMillis)
    try { await this.exit }
    finally { clearTimeout(timer) }
  }
}
