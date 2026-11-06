/**
 * Incremental conversion of raw local-model text into harness stream chunks.
 *
 * Recognizes `<think>…</think>` reasoning, Hermes/Qwen `<tool_call>{json}</tool_call>`
 * calls, Mistral `[TOOL_CALLS][…]` arrays, and a Llama-style reply that is a
 * single `{"name", "parameters"}` object naming a declared tool. Text that
 * could still become a marker is held until it resolves.
 *
 * @module dsh-llm-local-gguf/output-parser
 */

import { brandString } from '@deepseek-ai/dsh-brand'
import type { ContentBlockType, StreamChunk, ToolCallId } from '@deepseek-ai/dsh-llm'

const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'
const TOOL_OPEN = '<tool_call>'
const TOOL_CLOSE = '</tool_call>'
const MISTRAL_TOOLS = '[TOOL_CALLS]'

type Mode = 'start' | 'reasoning' | 'text' | 'tool' | 'mistral' | 'json'

/** Parser construction options. */
export interface OutputParserOptions {
  /** The prompt already opened a reasoning block. */
  readonly opensReasoning: boolean
  /** Declared tool names; bare-JSON calls must name one of them. */
  readonly toolNames: ReadonlySet<string>
  /** Source of tool call ids. */
  readonly newCallId: () => string
}

/** Length of the longest suffix of `text` that is a proper prefix of one of `markers`. */
function partialMarker(text: string, markers: readonly string[]): number {
  let best = 0
  for (const marker of markers) {
    for (let n = Math.min(marker.length - 1, text.length); n > best; n--) {
      if (marker.startsWith(text.slice(text.length - n))) { best = n; break }
    }
  }
  return best
}

/** Parsed call: name plus raw JSON arguments. */
interface ParsedCall { readonly name: string; readonly arguments: string }

function callOf(value: unknown): ParsedCall | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const record = value as Record<string, unknown>
  const name = record.name
  if (typeof name !== 'string' || name === '') return undefined
  const args = record.arguments ?? record.parameters ?? {}
  return { name, arguments: typeof args === 'string' ? args : JSON.stringify(args) }
}

function parseJson(text: string): unknown {
  try {
    const value: unknown = JSON.parse(text)
    return value
  }
  catch (_notJson) { return undefined }
}

/** Streaming parser for one generation. */
export class OutputParser {
  private mode: Mode
  private buffer = ''
  private index = -1
  private open: ContentBlockType | undefined
  private openText = ''
  private calls = 0

  /** @param options - reasoning start state, declared tools, and id source. */
  constructor(private readonly options: OutputParserOptions) {
    this.mode = options.opensReasoning ? 'reasoning' : 'start'
  }

  /** Number of tool calls emitted so far. */
  get toolCalls(): number {
    return this.calls
  }

  /**
   * Consume one text delta.
   * @param text - raw engine output.
   * @returns chunks that became certain.
   */
  push(text: string): StreamChunk[] {
    this.buffer += text
    const out: StreamChunk[] = []
    this.drain(out, false)
    return out
  }

  /**
   * Flush everything at end of generation.
   * @returns the remaining chunks, closing any open block.
   */
  finish(): StreamChunk[] {
    const out: StreamChunk[] = []
    this.drain(out, true)
    this.close(out)
    return out
  }

  private drain(out: StreamChunk[], final: boolean): void {
    for (;;) {
      const before = this.mode
      const buffered = this.buffer.length
      this.step(out, final)
      if (this.mode === before && this.buffer.length === buffered) return
    }
  }

  private step(out: StreamChunk[], final: boolean): void {
    switch (this.mode) {
      case 'start': {
        const trimmed = this.buffer.trimStart()
        if (trimmed === '') {
          if (final) this.buffer = ''
          return
        }
        if (trimmed.startsWith(THINK_OPEN)) {
          this.buffer = trimmed.slice(THINK_OPEN.length)
          this.mode = 'reasoning'
          return
        }
        if (!final && THINK_OPEN.startsWith(trimmed)) return
        this.buffer = trimmed
        this.mode = trimmed.startsWith('{') && this.options.toolNames.size > 0 ? 'json' : 'text'
        return
      }
      case 'reasoning': {
        const end = this.buffer.indexOf(THINK_CLOSE)
        if (end >= 0) {
          this.emit(out, 'reasoning', this.buffer.slice(0, end))
          this.close(out)
          this.buffer = this.buffer.slice(end + THINK_CLOSE.length).trimStart()
          this.mode = 'text'
          return
        }
        const hold = final ? 0 : partialMarker(this.buffer, [THINK_CLOSE])
        this.emit(out, 'reasoning', this.buffer.slice(0, this.buffer.length - hold))
        this.buffer = this.buffer.slice(this.buffer.length - hold)
        return
      }
      case 'text': {
        const markers = [TOOL_OPEN, MISTRAL_TOOLS]
        const hits = markers.map(m => [m, this.buffer.indexOf(m)] as const).filter(([, at]) => at >= 0)
        if (hits.length > 0) {
          const [marker, at] = hits.reduce((a, b) => (b[1] < a[1] ? b : a))
          this.emit(out, 'text', this.buffer.slice(0, at))
          this.buffer = this.buffer.slice(at + marker.length)
          this.mode = marker === TOOL_OPEN ? 'tool' : 'mistral'
          return
        }
        const hold = final ? 0 : partialMarker(this.buffer, markers)
        this.emit(out, 'text', this.buffer.slice(0, this.buffer.length - hold))
        this.buffer = this.buffer.slice(this.buffer.length - hold)
        return
      }
      case 'tool': {
        const end = this.buffer.indexOf(TOOL_CLOSE)
        if (end < 0 && !final) return
        const body = end < 0 ? this.buffer : this.buffer.slice(0, end)
        this.buffer = end < 0 ? '' : this.buffer.slice(end + TOOL_CLOSE.length).trimStart()
        const call = callOf(parseJson(body.trim()))
        if (call === undefined) this.emit(out, 'text', `${TOOL_OPEN}${body}${end < 0 ? '' : TOOL_CLOSE}`)
        else this.emitCall(out, call)
        this.mode = 'text'
        return
      }
      case 'mistral':
      case 'json': {
        if (!final) return
        const body = this.buffer.trim()
        this.buffer = ''
        const parsed = parseJson(body)
        const calls = (Array.isArray(parsed) ? parsed : [parsed]).map(callOf)
        const valid = calls.length > 0 && calls.every((c): c is ParsedCall => c !== undefined
          && (this.mode === 'mistral' || this.options.toolNames.has(c.name)))
        if (valid) for (const call of calls) this.emitCall(out, call)
        else this.emit(out, 'text', this.mode === 'mistral' ? `${MISTRAL_TOOLS}${body}` : body)
        this.mode = 'text'
        return
      }
    }
  }

  private emit(out: StreamChunk[], type: 'text' | 'reasoning', text: string): void {
    if (text === '') return
    if (this.open !== type) {
      this.close(out)
      // A block never starts with the whitespace separating it from the previous one.
      text = text.trimStart()
      if (text === '') return
      this.index++
      this.open = type
      this.openText = ''
      out.push({ type: 'block-start', index: this.index, blockType: type })
    }
    this.openText += text
    out.push(type === 'text'
      ? { type: 'text-delta', index: this.index, text }
      : { type: 'reasoning-delta', index: this.index, text })
  }

  private close(out: StreamChunk[]): void {
    if (this.open === undefined) return
    const text = this.openText.trimEnd()
    out.push({
      type: 'block-end',
      index: this.index,
      block: this.open === 'reasoning' ? { type: 'reasoning', text } : { type: 'text', text },
    })
    this.open = undefined
    this.openText = ''
  }

  private emitCall(out: StreamChunk[], call: ParsedCall): void {
    this.close(out)
    this.index++
    this.calls++
    const id = brandString<ToolCallId>(this.options.newCallId())
    out.push({ type: 'block-start', index: this.index, blockType: 'tool-call' })
    out.push({ type: 'tool-call-delta', index: this.index, id, name: call.name, argumentsDelta: call.arguments })
    out.push({ type: 'block-end', index: this.index, block: { type: 'tool-call', id, name: call.name, arguments: call.arguments } })
  }
}
