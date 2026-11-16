/**
 * Wire and service vocabulary of the local GGUF provider.
 *
 * @module dsh-llm-local-gguf/types
 */

import type {} from '@deepseek-ai/cordis'

declare module '@deepseek-ai/cordis' {
  interface Events {
    /**
     * The local engine's state changed without a catalog change: load progress, generation progress, or a finished
     * generation's statistics. Sent at most four times a second; catalog changes still arrive as `llm/models-updated`.
     * @mode emit
     * @param state - the new state.
     */
    'local-models/state'(state: LocalModelsState): void
  }
}

/** Header facts the engine reads from one GGUF file. */
export interface EngineModelFacts {
  readonly path: string
  /** `general.name`, when the file declares one. */
  readonly name: string | null
  /** `general.architecture`. */
  readonly architecture: string
  /** Whether the engine can run this architecture. */
  readonly supported: boolean
  readonly unsupportedReason: string | null
  /** Trained context length; `null` for unsupported architectures. */
  readonly contextLength: number | null
  readonly parameterCount: number
  /** `general.file_type` quantization code. */
  readonly fileType: number | null
  readonly fileBytes: number
  readonly hasChatTemplate: boolean
  /** The chat template references tool declarations. */
  readonly templateTools: boolean
  /** The chat template references `<think>` reasoning blocks. */
  readonly templateReasoning: boolean
  /** Decoder layers; `null` for an unsupported file. */
  readonly layers: number | null
  readonly attentionLayers: number | null
  readonly experts: number | null
  readonly expertsUsed: number | null
  readonly kvBytesPerToken: number | null
  readonly stateBytes: number | null
  readonly hasMtp: boolean
  /** `[type, bytes]` pairs, largest first. */
  readonly weightTypes: readonly (readonly [string, number])[]
}

/** Compute placement requested at load. */
export interface EnginePlacement {
  /** `auto`, `cpu`, `cuda`, or `cuda:<ordinal>`. */
  readonly device: string
  /** Trailing layers placed on the GPU; omitted places every layer there. */
  readonly gpuLayers?: number
}

/** Result of loading a model. */
export interface EngineLoaded {
  readonly facts: EngineModelFacts
  /** Effective context window after clamping to the trained length. */
  readonly contextLength: number
  readonly vocab: number
  /** Device running the offloaded layers: `cpu`, or `cuda:<ordinal>`. */
  readonly device: string
  /** Layers running on that device; 0 on the CPU. */
  readonly gpuLayers: number
  /** Routed experts on that device; 0 for a model without routed experts. */
  readonly gpuExperts: number
  /** Routed experts in the model; 0 for a model without them. */
  readonly experts: number
  /** Memory of the GPU in use; `null` on the CPU. */
  readonly vramTotalBytes: number | null
  /** Free memory of that GPU after loading; `null` on the CPU. */
  readonly vramFreeBytes: number | null
  /** Milliseconds the load took. */
  readonly loadMs: number
  /** Why the engine loaded on the CPU although a GPU was available. */
  readonly warning?: string
}

/** Engine failure body. */
export interface EngineFailure {
  readonly code: string
  readonly message: string
}

/** Prompt preparation facts sent before the first delta. */
export interface EngineStart {
  readonly promptTokens: number
  /** Prompt positions reused from the previous request's KV cache. */
  readonly cachedTokens: number
  /** The rendered prompt ends inside an opened reasoning block. */
  readonly opensReasoning: boolean
}

/** One text increment. */
export interface EngineDelta {
  readonly text: string
}

/** Progress of a load or a generation. */
export type EngineProgress =
  | { readonly stage: 'load'; readonly done: number; readonly total: number }
  | { readonly stage: 'prefill'; readonly done: number; readonly total: number }
  | { readonly stage: 'decode'; readonly done: number; readonly tokensPerSecond: number; readonly contextUsed: number }

/** Why the engine stopped; `repetition` means the output settled into a short repeated cycle. */
export type EngineFinishReason = 'stop' | 'length' | 'cancelled' | 'repetition'

/** Terminal result of a generation. */
export interface EngineOutcome {
  readonly finishReason: EngineFinishReason
  readonly promptTokens: number
  readonly cachedTokens: number
  readonly completionTokens: number
  /** Milliseconds spent evaluating the uncached prompt tokens. */
  readonly prefillMs: number
  /** Milliseconds spent generating. */
  readonly decodeMs: number
  /** Uncached prompt tokens evaluated per second; 0 when none were evaluated. */
  readonly promptTokensPerSecond: number
  /** Generated tokens per second; 0 when none were generated. */
  readonly tokensPerSecond: number
}

/** Placement and timing of the loaded model, for configuration surfaces. */
export interface LocalModelLoadStats {
  readonly gpuExperts: number
  readonly experts: number
  readonly vramTotalBytes: number | null
  readonly vramFreeBytes: number | null
  readonly loadMs: number
  readonly warning?: string
  /** The most recent finished generation; absent until one finishes. */
  readonly lastRun?: EngineOutcome
  /** The generation in progress; absent while idle. */
  readonly activity?: EngineProgress
}

/** OpenAI-style message the engine passes to the model's chat template. */
export interface EngineMessage {
  readonly role: 'system' | 'user' | 'assistant' | 'tool'
  readonly content: string
  readonly reasoning_content?: string
  readonly tool_calls?: readonly EngineToolCall[]
  readonly tool_call_id?: string
  readonly name?: string
}

/** One historical tool call; `arguments` is an object, as Hugging Face chat templates expect. */
export interface EngineToolCall {
  readonly id: string
  readonly type: 'function'
  readonly function: { readonly name: string; readonly arguments: unknown }
}

/** OpenAI-style tool declaration. */
export interface EngineTool {
  readonly type: 'function'
  readonly function: { readonly name: string; readonly description: string; readonly parameters: Record<string, unknown> }
}

/** Sampling controls on the wire. */
export interface EngineSampling {
  readonly temperature: number
  readonly topK: number
  readonly topP: number
  readonly minP: number
  readonly repeatPenalty: number
  readonly repeatLastN: number
  readonly seed?: number
}

/** One generation request on the wire. */
export interface EngineGenerateRequest {
  readonly messages: readonly EngineMessage[]
  readonly tools: readonly EngineTool[]
  readonly maxTokens: number
  readonly stop: readonly string[]
  readonly sampling: EngineSampling
  /** Reasoning switch passed to templates as `enable_thinking`. */
  readonly thinking?: boolean
  /** KV cache slot; requests in one slot reuse each other's prompt prefix. */
  readonly slot: string
  /** Constrain each `<tool_call>` body to a declared tool and its parameter schema. */
  readonly toolGrammar: boolean
}

/** Settings saved for one model file, applied whenever it loads. Omitted fields use the plugin configuration. */
export interface LocalModelSettings {
  /** Show the load options before loading this model, even when settings are saved. */
  readonly askOnLoad: boolean
  /** Context window in tokens; clamped to the trained length. */
  readonly contextLength?: number
  /** Trailing layers placed on the GPU; omitted lets the engine place every layer or fill VRAM with experts. */
  readonly gpuLayers?: number
  readonly temperature?: number
  readonly topP?: number
  readonly topK?: number
  /** Reasoning switch passed to the chat template. */
  readonly thinking?: boolean
}

/** Load request from a configuration surface: settings for this load, optionally saved for later loads. */
export interface LocalModelLoadOptions {
  readonly settings?: LocalModelSettings
  /** Save `settings` for this model file. */
  readonly remember?: boolean
}

/** Model structure from the GGUF header, for model cards and memory estimates. */
export interface LocalModelStructure {
  readonly layers: number
  readonly attentionLayers: number
  readonly experts: number
  readonly expertsUsed: number
  /** KV cache bytes per context position with F16 keys and values. */
  readonly kvBytesPerToken: number
  /** Fixed recurrent-state bytes per sequence. */
  readonly stateBytes: number
  /** The file carries multi-token-prediction layers, which are not run. */
  readonly hasMtp: boolean
  /** Weight bytes by type, largest first. */
  readonly weightTypes: readonly (readonly [string, number])[]
}

/** One model file offered by the catalog. */
export interface LocalModelEntry {
  /** Stable model id used in requests: the file path relative to its model directory, without `.gguf`. */
  readonly id: string
  /** Display name. */
  readonly name: string
  readonly path: string
  readonly directory: string
  readonly fileBytes: number
  readonly architecture: string
  readonly supported: boolean
  readonly unsupportedReason?: string
  readonly trainedContextLength?: number
  readonly parameterCount: number
  readonly quantization?: string
  readonly tools: boolean
  readonly reasoning: boolean
  /** Collection holding the file (`''` for the models root); absent for imported files and explicit directories. */
  readonly collection?: string
  /** Header structure; absent for files the engine cannot read. */
  readonly structure?: LocalModelStructure
  /** Saved settings; absent when none are saved. */
  readonly settings?: LocalModelSettings
}

/** Engine and loaded-model state reported to configuration surfaces. */
export type LocalModelsState =
  | { readonly kind: 'engine-unavailable'; readonly reason: string }
  | { readonly kind: 'idle' }
  | { readonly kind: 'loading'; readonly modelId: string; readonly layersLoaded?: number; readonly layers?: number }
  | { readonly kind: 'loaded'; readonly modelId: string; readonly contextLength: number; readonly device: string; readonly gpuLayers: number; readonly stats: LocalModelLoadStats }
  | { readonly kind: 'failed'; readonly modelId: string; readonly reason: string }

/** One CUDA device as the engine's `devices` operation reports it. */
export interface EngineDevice {
  readonly ordinal: number
  readonly name: string
  /** `major.minor`, for example `8.6`. */
  readonly computeCapability: string
  readonly totalBytes: number
  readonly freeBytes: number
  /** The device meets the kernels' minimum compute capability. */
  readonly supported: boolean
}

/** Result of the engine's `devices` operation. */
export interface EngineDevices {
  readonly cuda: boolean
  /** Lowest supported compute capability as `major * 10 + minor`. */
  readonly minComputeCapability: number
  readonly devices: readonly EngineDevice[]
}

/** An engine executable the provider may start, with the accelerator its build targets. */
export interface EngineCandidate {
  readonly path: string
  /** `cuda` and `cpu` builds come from an engine directory; `unknown` is a single configured or found executable. */
  readonly accelerator: 'cuda' | 'cpu' | 'unknown'
}

/** The running engine: which executable, what it computes on, and why a CUDA engine was passed over. */
export interface LocalEngineInfo {
  readonly path: string
  readonly accelerator: 'cuda' | 'cpu'
  /** CUDA devices the engine reports; empty for the CPU. */
  readonly devices: readonly EngineDevice[]
  /** Why the CUDA engine is not in use, when an engine directory offered one. */
  readonly fallbackReason?: string
}

/** One model collection: a folder under the models root, or the root itself (`id` `''`). */
export interface LocalModelCollection {
  readonly id: string
  readonly modelCount: number
}

/** Collections under the models root and the active one. */
export interface LocalModelCollections {
  readonly root: string
  readonly active: string
  readonly items: readonly LocalModelCollection[]
}

/** Catalog snapshot returned to configuration surfaces. */
export interface LocalModelsSnapshot {
  readonly state: LocalModelsState
  /** Model directories scanned, in configuration order. */
  readonly directories: readonly string[]
  /** The active collection's models, or every configured directory's when collections are off. */
  readonly models: readonly LocalModelEntry[]
  /** Directories that could not be read, with the reason. */
  readonly problems: readonly { readonly directory: string; readonly reason: string }[]
  /** The engine started for this provider; absent before the first start. */
  readonly engine?: LocalEngineInfo
  /** Model collections; absent when explicit `modelDirectories` configuration replaces them. */
  readonly collections?: LocalModelCollections
}

/** Outcome of loading a GGUF file chosen on disk. */
export type LocalModelsFileResult =
  | { readonly ok: true; readonly modelId: string; readonly snapshot: LocalModelsSnapshot }
  | { readonly ok: false; readonly reason: string; readonly snapshot: LocalModelsSnapshot }

/** Outcome of a load or unload request from a configuration surface. */
export type LocalModelsActionResult =
  | { readonly ok: true; readonly snapshot: LocalModelsSnapshot }
  | { readonly ok: false; readonly reason: string; readonly snapshot: LocalModelsSnapshot }
