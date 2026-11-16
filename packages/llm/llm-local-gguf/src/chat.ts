/**
 * Interactive terminal chat with one local model: streaming replies, reasoning shown dimmed, live load progress,
 * per-reply statistics, and slash commands. It talks to the engine directly, without tools or the agent loop.
 *
 * @module dsh-llm-local-gguf/chat
 */

import { createInterface } from 'node:readline/promises'
import type { Readable, Writable } from 'node:stream'
import { Config, runtimeSettingsOf, samplingOf } from './config.ts'
import { DEFAULT_MODEL_ID, LocalModelRuntime } from './runtime.ts'
import type { EngineMessage, EngineOutcome, LocalModelsState } from './types.ts'

/** Inputs of one chat session. */
export interface LocalChatOptions {
  readonly input: Readable
  readonly output: Writable
  readonly env: Readonly<Record<string, string | undefined>>
  /** Catalog id to load; omitted uses `DSH_LOCAL_MODEL`, then the model loaded last, then the only supported model. */
  readonly modelId?: string
  /** Color output with ANSI escapes. */
  readonly color: boolean
}

/** Slash commands and what they do, printed by `/help`. */
export const CHAT_HELP = `Commands:
  /help              show this list
  /clear             start a new conversation
  /system <text>     set the system message and start a new conversation
  /think on|off|default  switch reasoning for models that support it
  /stats             show the loaded model and the last reply's figures
  /exit              quit (also Ctrl+D)
`

/** Token count in k units. */
function k(count: number): string {
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count)
}

/**
 * One line of figures for a finished reply.
 * @param outcome - engine outcome.
 * @param contextLength - loaded context window.
 * @returns the line without a newline.
 */
export function replyStats(outcome: EngineOutcome, contextLength: number): string {
  const parts = [
    `${outcome.tokensPerSecond.toFixed(1)} tok/s`,
    `${String(outcome.completionTokens)} tokens`,
    `prompt ${k(outcome.promptTokens)} (${k(outcome.cachedTokens)} cached, ${outcome.promptTokensPerSecond.toFixed(0)} tok/s)`,
    `context ${k(outcome.promptTokens + outcome.completionTokens)}/${k(contextLength)}`,
  ]
  if (outcome.finishReason === 'repetition') parts.push('stopped: the reply repeated itself')
  if (outcome.finishReason === 'length') parts.push('stopped at the output limit')
  return parts.join(' · ')
}

/** Splits streamed text into reasoning and answer, for a template that opens or emits `<think>` blocks. */
export class ReasoningSplitter {
  private inside: boolean
  private pending = ''

  /** @param opensReasoning - the prompt ended inside a reasoning block. */
  constructor(opensReasoning: boolean) {
    this.inside = opensReasoning
  }

  /**
   * Feed one delta.
   * @param text - streamed text.
   * @returns pieces in order, each marked as reasoning or answer.
   */
  push(text: string): { readonly reasoning: boolean; readonly text: string }[] {
    const out: { reasoning: boolean; text: string }[] = []
    let rest = this.pending + text
    this.pending = ''
    for (;;) {
      const tag = this.inside ? '</think>' : '<think>'
      const at = rest.indexOf(tag)
      if (at < 0) {
        // Hold back a tail that could be the start of the tag.
        let hold = 0
        for (let n = Math.min(tag.length - 1, rest.length); n > 0; n--) {
          if (tag.startsWith(rest.slice(rest.length - n))) { hold = n; break }
        }
        if (rest.length > hold) out.push({ reasoning: this.inside, text: rest.slice(0, rest.length - hold) })
        this.pending = rest.slice(rest.length - hold)
        return out
      }
      if (at > 0) out.push({ reasoning: this.inside, text: rest.slice(0, at) })
      rest = rest.slice(at + tag.length)
      this.inside = !this.inside
    }
  }

  /**
   * Flush held text at the end of a reply.
   * @returns the remaining piece, if any.
   */
  finish(): { readonly reasoning: boolean; readonly text: string }[] {
    const rest = this.pending
    this.pending = ''
    return rest === '' ? [] : [{ reasoning: this.inside, text: rest }]
  }
}

/**
 * Run an interactive chat until the user quits or input ends.
 * @param options - streams, environment, and model choice.
 * @returns the process exit code: 0 after a normal exit, 1 when no model could load.
 */
export async function runLocalChat(options: LocalChatOptions): Promise<number> {
  const { input, output, env } = options
  const dim = (text: string) => (options.color ? `\u001B[2m${text}\u001B[22m` : text)
  const write = (text: string) => { output.write(text) }
  const config = Config({})
  const settings = runtimeSettingsOf(config, env, undefined)
  const engine = settings.engine()
  if ('reason' in engine) {
    write(`freakseek chat: ${engine.reason}\n`)
    return 1
  }
  // Written by the progress callback.
  const progress = { shown: false }
  const runtime = new LocalModelRuntime(settings, () => {}, (state: LocalModelsState) => {
    if (state.kind !== 'loading' || state.layers === undefined) return
    const done = state.layersLoaded ?? 0
    write(`\rLoading ${String(Math.round(done / Math.max(1, state.layers) * 100))}% (${String(done)}/${String(state.layers)} layers)`)
    progress.shown = true
  })
  const rl = createInterface({ input, output, terminal: options.color })
  // Created at once, the iterator buffers lines typed or piped while the model loads or a reply streams.
  const lines = rl[Symbol.asyncIterator]()
  try {
    await runtime.refresh()
    const modelId = options.modelId ?? env.DSH_LOCAL_MODEL ?? (await runtime.lastLoadedModel()) ?? DEFAULT_MODEL_ID
    let contextLength: number
    try { contextLength = await runtime.load(modelId) }
    catch (error) {
      write(`${progress.shown ? '\n' : ''}freakseek chat: ${(error as Error).message}\n`)
      return 1
    }
    const entry = runtime.find(modelId)
    const state = runtime.snapshot().state
    const where = state.kind === 'loaded' ? state.device : 'cpu'
    write(`${progress.shown ? '\n' : ''}${entry?.name ?? modelId} · ${where} · context ${k(contextLength)}. Type /help for commands.\n`)
    let system = ''
    let thinking: boolean | undefined = runtime.settingsFor(modelId)?.thinking
    let history: EngineMessage[] = []
    let last: EngineOutcome | undefined
    let conversation = 0
    for (;;) {
      write('> ')
      const next = await lines.next()
      if (next.done === true) break
      const text = next.value.trim()
      if (text === '') continue
      if (text === '/exit' || text === '/quit') break
      if (text === '/help') { write(CHAT_HELP); continue }
      if (text === '/clear') { history = []; conversation++; write(dim('New conversation.\n')); continue }
      if (text.startsWith('/system')) {
        system = text.slice('/system'.length).trim()
        history = []
        conversation++
        write(dim(system === '' ? 'System message cleared; new conversation.\n' : 'System message set; new conversation.\n'))
        continue
      }
      if (text.startsWith('/think')) {
        const value = text.slice('/think'.length).trim()
        thinking = value === 'on' ? true : value === 'off' ? false : undefined
        write(dim(`Reasoning: ${thinking === undefined ? 'model default' : thinking ? 'on' : 'off'}.\n`))
        continue
      }
      if (text === '/stats') {
        write(dim(`${entry?.name ?? modelId} · ${where} · context ${k(contextLength)}\n`))
        if (last !== undefined) write(dim(`${replyStats(last, contextLength)}\n`))
        continue
      }
      if (text.startsWith('/')) { write(`Unknown command ${text.split(' ')[0] ?? ''}; type /help.\n`); continue }
      history.push({ role: 'user', content: text })
      const messages: EngineMessage[] = system === '' ? history : [{ role: 'system', content: system }, ...history]
      const saved = runtime.settingsFor(modelId)
      let splitter = new ReasoningSplitter(false)
      let answer = ''
      const show = (pieces: readonly { reasoning: boolean; text: string }[]) => {
        for (const piece of pieces) {
          write(piece.reasoning ? dim(piece.text) : piece.text)
          if (!piece.reasoning) answer += piece.text
        }
      }
      const controller = new AbortController()
      const interrupt = () => { controller.abort() }
      process.once('SIGINT', interrupt)
      try {
        last = await runtime.generate(modelId, {
          messages,
          tools: [],
          maxTokens: config.maxOutputTokens.get(),
          stop: [],
          sampling: {
            ...samplingOf(config, entry),
            ...saved?.temperature === undefined ? {} : { temperature: saved.temperature },
            ...saved?.topP === undefined ? {} : { topP: saved.topP },
            ...saved?.topK === undefined ? {} : { topK: saved.topK },
          },
          ...thinking === undefined ? {} : { thinking },
          slot: `chat-${String(conversation)}`,
          toolGrammar: false,
        }, {
          onStart: (start) => { splitter = new ReasoningSplitter(start.opensReasoning) },
          onDelta: (delta) => { show(splitter.push(delta)) },
        }, controller.signal)
        show(splitter.finish())
        history.push({ role: 'assistant', content: answer.trim() })
        write(`\n${dim(`[${replyStats(last, contextLength)}]`)}\n`)
      } catch (error) {
        history.pop()
        write(`\nfreakseek chat: ${(error as Error).message}\n`)
      } finally {
        process.removeListener('SIGINT', interrupt)
      }
    }
    return 0
  } finally {
    rl.close()
    await runtime.dispose()
  }
}
