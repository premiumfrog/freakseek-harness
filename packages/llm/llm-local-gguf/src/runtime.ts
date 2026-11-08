/**
 * Owner of the engine process, the model catalog, and the loaded-model state.
 *
 * Every engine operation runs through one queue so a model switch never
 * interleaves with a generation. The process starts on first use and is
 * replaced after an unexpected exit.
 *
 * @module dsh-llm-local-gguf/runtime
 */

import { readdir, stat } from 'node:fs/promises'
import { basename, dirname, join, relative, sep } from 'node:path'
import { EngineError, EngineProcess } from './engine-process.ts'
import { addImport, importedModelId, readImports, validateModelFile } from './imports.ts'
import type { GenerateHandlers } from './engine-process.ts'
import type { EngineGenerateRequest, EngineModelFacts, EnginePlacement, EngineOutcome, LocalModelEntry, LocalModelsSnapshot, LocalModelsState } from './types.ts'

/** `general.file_type` codes (llama.cpp `llama_ftype`) to display names. */
const FILE_TYPES: Readonly<Record<number, string>> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M',
  13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 32: 'BF16',
}

/** Shards after the first of a split GGUF, and multimodal projectors, are not chat models. */
const NOT_A_MODEL = /(?:-0000[2-9]|-000[1-9]\d)-of-\d{5}\.gguf$|mmproj/iu

/** Runtime inputs read at each operation, so settings changes apply without a restart. */
export interface RuntimeSettings {
  /** Engine executable, or the reason none is available. */
  engine(): { readonly path: string } | { readonly reason: string }
  threads(): number | undefined
  directories(): readonly string[]
  contextLength(): number
  /** KV cache slots kept per loaded model. */
  cacheSlots(): number
  /** JSON file listing GGUF files loaded from outside the model directories. */
  importsFile(): string
  /** Compute device and GPU layer count requested at load. */
  placement(): EnginePlacement
  shutdownGraceMillis(): number
}

interface CachedFacts {
  readonly size: number
  readonly mtimeMs: number
  readonly facts: EngineModelFacts | { readonly error: string }
}

/** Display name from header and file facts. */
function displayName(facts: EngineModelFacts, file: string, quantization: string | undefined): string {
  const name = facts.name ?? basename(file, '.gguf')
  return quantization === undefined || name.toUpperCase().includes(quantization) ? name : `${name} (${quantization})`
}

/** Model id that selects the only supported model in the catalog, for launches that name none. */
export const DEFAULT_MODEL_ID = 'default'

/** Owner of local model state; one per plugin instance. */
export class LocalModelRuntime {
  private process: EngineProcess | undefined
  private queue: Promise<unknown> = Promise.resolve()
  private readonly facts = new Map<string, CachedFacts>()
  private entries: LocalModelEntry[] = []
  private problems: { directory: string; reason: string }[] = []
  private loaded: { readonly modelId: string; readonly path: string; readonly contextLength: number } | undefined
  private state: LocalModelsState = { kind: 'idle' }
  private disposed = false
  /** Orders read-modify-write updates of the imports file. */
  private importing: Promise<unknown> = Promise.resolve()

  /**
   * @param settings - live settings readers.
   * @param onChange - called after the catalog or state changes.
   */
  constructor(private readonly settings: RuntimeSettings, private readonly onChange: () => void) {}

  /** Run `task` after every earlier engine operation settles. */
  private serial<T>(task: () => Promise<T>): Promise<T> {
    const run = this.queue.then(task, task)
    this.queue = run.catch((_reportedToCaller: unknown) => {})
    return run
  }

  private setState(state: LocalModelsState): void {
    this.state = state
    this.onChange()
  }

  private spawning: Promise<EngineProcess> | undefined

  /** The running engine, spawning one when none is alive; concurrent callers share one spawn. */
  private engine(): Promise<EngineProcess> {
    if (this.disposed) return Promise.reject(new EngineError('ENGINE_DISPOSED', 'local model provider is shutting down'))
    if (this.process?.alive === true) return Promise.resolve(this.process)
    this.spawning ??= this.spawn().finally(() => { this.spawning = undefined })
    return this.spawning
  }

  private async spawn(): Promise<EngineProcess> {
    const resolved = this.settings.engine()
    if (!('path' in resolved)) throw new EngineError('ENGINE_UNAVAILABLE', resolved.reason)
    const threads = this.settings.threads()
    const proc = new EngineProcess({ executable: resolved.path, ...threads === undefined ? {} : { threads } })
    this.process = proc
    this.loaded = undefined
    void proc.exit.then(() => {
      if (this.process !== proc || this.disposed) return
      this.loaded = undefined
      if (this.state.kind === 'loaded' || this.state.kind === 'loading') this.setState({ kind: 'idle' })
    })
    try { await proc.handshake() }
    catch (error) {
      await proc.close(0)
      throw error
    }
    return proc
  }

  /**
   * Current snapshot for configuration surfaces.
   * @returns state, directories, and the last scanned catalog.
   */
  snapshot(): LocalModelsSnapshot {
    const engine = this.settings.engine()
    const state: LocalModelsState = 'reason' in engine ? { kind: 'engine-unavailable', reason: engine.reason } : this.state
    return { state, directories: this.settings.directories(), models: this.entries, problems: this.problems }
  }

  /** Last scanned catalog. */
  get models(): readonly LocalModelEntry[] {
    return this.entries
  }

  /**
   * Effective context window of a model if loaded now: the configured length clamped to the trained one.
   * @param modelId - catalog id.
   * @returns the window in tokens, or `undefined` for an unknown model.
   */
  contextWindow(modelId: string): number | undefined {
    if (this.loaded?.modelId === modelId) return this.loaded.contextLength
    const entry = this.find(modelId)
    if (entry === undefined) return undefined
    return Math.min(this.settings.contextLength(), entry.trainedContextLength ?? this.settings.contextLength())
  }

  /**
   * Rescan every model directory, inspecting new or changed files.
   * @returns the refreshed catalog.
   */
  async refresh(): Promise<readonly LocalModelEntry[]> {
    const entries: LocalModelEntry[] = []
    const problems: { directory: string; reason: string }[] = []
    const files: { directory: string; path: string; imported?: true }[] = []
    for (const directory of this.settings.directories()) {
      try {
        for (const path of await ggufFiles(directory, 2)) files.push({ directory, path })
      } catch (error) {
        const code = (error as NodeJS.ErrnoException).code
        // A default directory that does not exist yet simply holds no models.
        if (code !== 'ENOENT') problems.push({ directory, reason: (error as Error).message })
      }
    }
    const importsFile = this.settings.importsFile()
    try {
      for (const path of await readImports(importsFile)) {
        // An import that a scanned directory already lists keeps its directory id.
        if (files.some(f => f.path === path)) continue
        try { await stat(path) }
        catch (error) {
          problems.push({ directory: path, reason: (error as Error).message })
          continue
        }
        files.push({ directory: dirname(path), path, imported: true })
      }
    } catch (error) {
      problems.push({ directory: importsFile, reason: (error as Error).message })
    }
    for (const { directory, path, imported } of files) {
      const facts = await this.inspect(path)
      const id = imported === true ? importedModelId(path) : relative(directory, path).split(sep).join('/').replace(/\.gguf$/iu, '')
      if (entries.some(e => e.id === id)) continue
      if ('error' in facts) {
        const size = this.facts.get(path)?.size ?? 0
        entries.push({
          id, name: basename(path, '.gguf'), path, directory, fileBytes: size, architecture: 'unknown', supported: false,
          unsupportedReason: facts.error, parameterCount: 0, tools: false, reasoning: false,
        })
        continue
      }
      const quantization = facts.fileType === null ? undefined : FILE_TYPES[facts.fileType]
      entries.push({
        id,
        name: displayName(facts, path, quantization),
        path,
        directory,
        fileBytes: facts.fileBytes,
        architecture: facts.architecture,
        supported: facts.supported,
        ...facts.unsupportedReason === null ? {} : { unsupportedReason: facts.unsupportedReason },
        ...facts.contextLength === null ? {} : { trainedContextLength: facts.contextLength },
        parameterCount: facts.parameterCount,
        ...quantization === undefined ? {} : { quantization },
        tools: facts.templateTools,
        reasoning: facts.templateReasoning,
      })
    }
    // Announce only real catalog changes: listeners re-read the catalog, which rescans again.
    const changed = JSON.stringify([entries, problems]) !== JSON.stringify([this.entries, this.problems])
    this.entries = entries
    this.problems = problems
    if (changed) this.onChange()
    return entries
  }

  private async inspect(path: string): Promise<EngineModelFacts | { readonly error: string }> {
    const info = await stat(path)
    const cached = this.facts.get(path)
    if (cached !== undefined && cached.size === info.size && cached.mtimeMs === info.mtimeMs) return cached.facts
    let facts: CachedFacts['facts']
    try { facts = await (await this.engine()).inspect(path) }
    catch (error) {
      if (error instanceof EngineError && error.code !== 'ENGINE_ERROR') throw error
      facts = { error: (error as Error).message }
    }
    this.facts.set(path, { size: info.size, mtimeMs: info.mtimeMs, facts })
    return facts
  }

  /**
   * Add a GGUF file from anywhere on disk to the catalog, remember it, and load it.
   * @param path - absolute `.gguf` path received from a configuration surface.
   * @returns the model's catalog id.
   * @throws EngineError `INVALID_MODEL_FILE` for a path that is not a readable `.gguf` file, or any load failure.
   */
  async loadFile(path: unknown): Promise<string> {
    const file = await validateModelFile(path)
    const added = this.importing.then(() => addImport(this.settings.importsFile(), file))
    this.importing = added.catch((_reportedToCaller: unknown) => {})
    await added
    const entry = (await this.refresh()).find(e => e.path === file)
    if (entry === undefined) throw new EngineError('INVALID_MODEL_FILE', `${file} did not enter the local model catalog`)
    await this.load(entry.id)
    return entry.id
  }

  /**
   * Load a catalog model unless it is already loaded.
   * @param modelId - catalog id.
   * @returns the effective context window.
   */
  load(modelId: string): Promise<number> {
    return this.serial(() => this.loadNow(modelId))
  }

  /** Catalog entry for an id; {@link DEFAULT_MODEL_ID} names the only supported model. */
  private find(modelId: string): LocalModelEntry | undefined {
    if (modelId !== DEFAULT_MODEL_ID) return this.entries.find(e => e.id === modelId)
    const supported = this.entries.filter(e => e.supported)
    return supported.length === 1 ? supported[0] : undefined
  }

  private async loadNow(modelId: string): Promise<number> {
    let entry = this.find(modelId)
    if (entry === undefined) {
      await this.refresh()
      entry = this.find(modelId)
    }
    if (entry === undefined && modelId === DEFAULT_MODEL_ID) {
      const ids = this.entries.filter(e => e.supported).map(e => e.id)
      throw new EngineError('UNKNOWN_MODEL', ids.length === 0
        ? `no supported local model in ${this.settings.directories().join(', ')}`
        : `several local models are available; choose one (for example with DSH_LOCAL_MODEL): ${ids.join(', ')}`)
    }
    if (entry === undefined) throw new EngineError('UNKNOWN_MODEL', `no local model "${modelId}" in ${this.settings.directories().join(', ')}`)
    if (!entry.supported) throw new EngineError('UNSUPPORTED_MODEL', `${entry.name}: ${entry.unsupportedReason ?? 'unsupported model'}`)
    const engine = await this.engine()
    const contextLength = this.settings.contextLength()
    const effective = Math.min(contextLength, entry.trainedContextLength ?? contextLength)
    if (this.loaded?.path === entry.path && this.loaded.contextLength === effective) {
      return this.loaded.contextLength
    }
    this.loaded = undefined
    this.setState({ kind: 'loading', modelId })
    try {
      const result = await engine.load(entry.path, contextLength, this.settings.cacheSlots(), this.settings.placement())
      this.loaded = { modelId, path: entry.path, contextLength: result.contextLength }
      this.setState({ kind: 'loaded', modelId, contextLength: result.contextLength, device: result.device, gpuLayers: result.gpuLayers })
      return result.contextLength
    } catch (error) {
      this.setState({ kind: 'failed', modelId, reason: (error as Error).message })
      throw error
    }
  }

  /**
   * Release the loaded model.
   * @returns resolves once memory is released.
   */
  unload(): Promise<void> {
    return this.serial(async () => {
      if (this.process?.alive === true) await this.process.unload()
      this.loaded = undefined
      this.setState({ kind: 'idle' })
    })
  }

  /**
   * Load `modelId` if needed, then generate.
   * @param modelId - catalog id.
   * @param request - wire request.
   * @param handlers - streaming callbacks.
   * @param signal - cancellation.
   * @returns generation outcome.
   */
  generate(modelId: string, request: EngineGenerateRequest, handlers: GenerateHandlers, signal?: AbortSignal): Promise<EngineOutcome> {
    return this.serial(async () => {
      await this.loadNow(modelId)
      const engine = await this.engine()
      return engine.generate(request, handlers, signal)
    })
  }

  /**
   * Stop the engine process.
   * @returns resolves once it exited.
   */
  async dispose(): Promise<void> {
    this.disposed = true
    await this.process?.close(this.settings.shutdownGraceMillis())
    this.process = undefined
  }
}

/**
 * List `.gguf` files under `directory`, descending at most `depth` levels.
 * @param directory - root to scan.
 * @param depth - remaining subdirectory levels.
 * @returns absolute paths sorted by name.
 */
async function ggufFiles(directory: string, depth: number): Promise<string[]> {
  const out: string[] = []
  const entries = await readdir(directory, { withFileTypes: true })
  entries.sort((a, b) => a.name.localeCompare(b.name))
  for (const entry of entries) {
    const path = join(directory, entry.name)
    if (entry.isDirectory() && depth > 0 && !entry.name.startsWith('.')) {
      try { out.push(...await ggufFiles(path, depth - 1)) }
      catch (_unreadableSubdirectory) {
        // An unreadable nested folder hides only its own files.
        continue
      }
    } else if (entry.isFile() && /\.gguf$/iu.test(entry.name) && !NOT_A_MODEL.test(entry.name)) {
      out.push(path)
    }
  }
  return out
}
