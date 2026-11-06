/**
 * The `local` provider route: harness requests run on an in-process-owned
 * GGUF engine with no network access.
 *
 * @module dsh-llm-local-gguf/adapter
 */

import { randomUUID } from 'node:crypto'
import { CONTEXT_WINDOW_EXCEEDED_CODE, LlmAdapter, LlmError } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, LlmModelInfo, LlmProviderInfo, LlmResolvedModelInfo, StreamChunk } from '@deepseek-ai/dsh-llm'
import { EngineError } from './engine-process.ts'
import { OutputParser } from './output-parser.ts'
import { engineMessages, engineTools } from './request.ts'
import type { LocalModelRuntime } from './runtime.ts'
import type { EngineSampling, EngineStart, LocalModelEntry } from './types.ts'

/** Provider route id served by this adapter. */
export const LOCAL_PROVIDER = 'local'

/** Live adapter inputs. */
export interface LocalAdapterSettings {
  sampling(): EngineSampling
  maxOutputTokens(): number
}

/** Map engine failure codes to harness failure codes. */
function llmErrorOf(error: unknown): LlmError {
  if (error instanceof LlmError) return error
  if (error instanceof EngineError) {
    const code = error.code === 'CONTEXT_WINDOW_EXCEEDED' ? CONTEXT_WINDOW_EXCEEDED_CODE : `LOCAL_${error.code}`
    return new LlmError(`local model: ${error.message}`, code, { cause: error })
  }
  return new LlmError(`local model: ${(error as Error).message}`, 'LOCAL_ENGINE_ERROR', { cause: error })
}

function describe(entry: LocalModelEntry): string {
  const parts = [entry.architecture]
  if (entry.parameterCount > 0) parts.push(`${(entry.parameterCount / 1e9).toFixed(entry.parameterCount < 1e10 ? 1 : 0)}B parameters`)
  if (entry.quantization !== undefined) parts.push(entry.quantization)
  parts.push(`${(entry.fileBytes / 2 ** 30).toFixed(1)} GiB on disk`)
  return parts.join(' · ')
}

/**
 * KV cache slot of one request: its Session and auxiliary purpose.
 * @param options - assembled request.
 * @returns slot key.
 */
function slotOf(options: GenerateOptions<string>): string {
  // Producer packages declare purposes, so the forwarding form reads them as open strings.
  return `${options.sessionId ?? ''}:${options.purpose ?? ''}`
}

/** Adapter for the `local` route. */
export class LocalGgufAdapter extends LlmAdapter {
  /**
   * @param runtime - engine and catalog owner.
   * @param settings - live sampling and output settings.
   */
  constructor(private readonly runtime: LocalModelRuntime, private readonly settings: LocalAdapterSettings) {
    super()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Local models' }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const entries = await this.runtime.refresh()
    return entries.filter(e => e.supported).map(e => ({
      provider, id: e.id, name: e.name, description: describe(e), inputModalities: ['text'],
    }))
  }

  override async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    let entry = this.runtime.models.find(e => e.id === model)
    if (entry === undefined) {
      await this.runtime.refresh()
      entry = this.runtime.models.find(e => e.id === model)
    }
    const contextWindow = this.runtime.contextWindow(model)
    return {
      provider,
      id: model,
      name: entry?.name ?? model,
      inputModalities: ['text'],
      ...contextWindow === undefined ? {} : { context: { contextWindow } },
      defaultMaxTokens: this.settings.maxOutputTokens(),
    }
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const tools = engineTools(options)
    const queue: StreamChunk[] = []
    let wake: (() => void) | undefined
    let parser: OutputParser | undefined
    let start: EngineStart | undefined
    const notify = () => { wake?.(); wake = undefined }
    const toolNames = new Set(tools.map(t => t.function.name))
    const generation = this.runtime.generate(options.model, {
      messages: engineMessages(options),
      tools,
      maxTokens: options.maxTokens ?? this.settings.maxOutputTokens(),
      stop: options.stop ?? [],
      // One slot per conversation and request kind, so a title or compaction request does not evict the conversation prefix.
      slot: slotOf(options),
      sampling: { ...this.settings.sampling(), ...options.temperature === undefined ? {} : { temperature: options.temperature } },
    }, {
      onStart: (s) => {
        start = s
        parser = new OutputParser({ opensReasoning: s.opensReasoning, toolNames, newCallId: () => `call_${randomUUID().replaceAll('-', '').slice(0, 24)}` })
      },
      onDelta: (text) => {
        queue.push(...(parser as OutputParser).push(text))
        notify()
      },
    }, options.signal)
    // Written by the generation callbacks while this generator drains the queue.
    const settlement: { done: boolean; result?: Awaited<typeof generation>; failure?: { error: unknown } } = { done: false }
    generation.then(
      (result) => { settlement.result = result },
      (error: unknown) => { settlement.failure = { error } },
    ).finally(() => { settlement.done = true; notify() })
    for (;;) {
      while (queue.length > 0) yield queue.shift() as StreamChunk
      if (settlement.done) break
      await new Promise<void>((resolve) => { wake = resolve })
    }
    if (settlement.failure !== undefined) throw llmErrorOf(settlement.failure.error)
    const result = settlement.result as Awaited<typeof generation>
    if (result.finishReason === 'cancelled') {
      throw new LlmError('local model: generation cancelled', 'ABORTED')
    }
    if (parser !== undefined) yield* parser.finish()
    const cached = start?.cachedTokens ?? result.cachedTokens
    yield {
      type: 'usage',
      usage: {
        inputTokens: result.promptTokens - cached,
        cacheReadTokens: cached,
        outputTokens: result.completionTokens,
        totalTokens: result.promptTokens + result.completionTokens,
      },
    }
    const toolCalls = parser?.toolCalls ?? 0
    yield {
      type: 'finish',
      reason: toolCalls > 0 ? { kind: 'tool-calls' } : result.finishReason === 'length' ? { kind: 'max-tokens' } : { kind: 'stop' },
    }
  }
}
