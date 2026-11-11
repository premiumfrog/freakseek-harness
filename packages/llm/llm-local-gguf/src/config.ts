/**
 * Configuration of the local GGUF provider and its explicit resolution step.
 *
 * @module dsh-llm-local-gguf/config
 */

import { existsSync } from 'node:fs'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Volatile } from '@deepseek-ai/cordis'
import { expandHomePath, resolveDshHome } from '@deepseek-ai/dsh-home-paths'
import z from '@deepseek-ai/schemastery'
import type { EngineCandidate, EnginePlacement, EngineSampling } from './types.ts'

/** Environment variable naming the engine executable. */
export const ENGINE_ENV = 'DSH_GGUF_ENGINE'

/** Environment variable naming a directory with `cuda/` and `cpu/` engine builds. */
export const ENGINE_DIR_ENV = 'DSH_GGUF_ENGINE_DIR'

/** Environment variable naming the models root whose subfolders are collections. */
export const MODELS_ROOT_ENV = 'DSH_LOCAL_MODELS_ROOT'

/** Plugin configuration. */
export interface Config {
  /** Engine executable; omitted resolves through {@link resolveEngineCandidates}. */
  enginePath?: string
  /** Directory holding `cuda/` and `cpu/` engine builds; `device: auto` prefers the CUDA one and falls back to the CPU one. */
  engineDirectory?: string
  /** Models root whose subfolders are collections; omitted resolves to `$DSH_HOME/models`. Unused when `modelDirectories` is set. */
  modelsRoot?: string
  /** Directories scanned for `.gguf` files, `~`-expanded; a nonempty list replaces collections under {@link Config.modelsRoot}. */
  modelDirectories: Volatile<string[] | undefined>
  /** Context window requested at load, clamped by the engine to the trained length. */
  contextLength: Volatile<number>
  /** KV cache slots kept at once; each conversation and auxiliary request kind uses its own. */
  cacheSlots: number
  /** Compute device: `auto` uses CUDA device 0 when the engine was built with CUDA and one is present, else the CPU. */
  device: string
  /** Trailing layers placed on the GPU; omitted places every layer there. Lower it when weights and KV cache exceed VRAM. */
  gpuLayers?: number
  /** Default per-reply output cap. */
  maxOutputTokens: Volatile<number>
  /** Inference threads; 0 uses every core. */
  threads: number
  /** Sampling temperature; 0 selects greedy decoding. A request's own temperature overrides it. */
  temperature: Volatile<number>
  /** Candidates kept by top-k filtering; 0 keeps the whole vocabulary. */
  topK: Volatile<number>
  /** Cumulative probability kept by nucleus (top-p) filtering. */
  topP: Volatile<number>
  /** Candidates below this fraction of the most likely token's probability are dropped. */
  minP: Volatile<number>
  /** Penalty divisor for tokens among the last 64 generated; 1 disables it. */
  repeatPenalty: Volatile<number>
  /** Milliseconds the engine gets to exit on shutdown before it is killed. */
  shutdownGraceMillis: number
  /** JSON list of GGUF files loaded from outside the model directories; omitted resolves to `$DSH_HOME/local-models/imported.json`. */
  importedModelsFile?: string
}

export const Config = z.object({
  enginePath: z.string(),
  engineDirectory: z.string(),
  modelsRoot: z.string(),
  modelDirectories: z.array(z.string()).volatile(),
  contextLength: z.number().step(1).min(256).max(1_048_576).default(32768).volatile(),
  cacheSlots: z.number().step(1).min(1).max(64).default(4),
  device: z.string().pattern(/^(?:auto|cpu|cuda(?::\d+)?)$/).default('auto'),
  gpuLayers: z.natural(),
  maxOutputTokens: z.number().step(1).min(1).max(1_048_576).default(4096).volatile(),
  threads: z.number().step(1).min(0).max(1024).default(0),
  temperature: z.number().min(0).max(5).default(0.6).volatile(),
  topK: z.number().step(1).min(0).max(100_000).default(40).volatile(),
  topP: z.number().min(0).max(1).default(0.95).volatile(),
  minP: z.number().min(0).max(1).default(0.05).volatile(),
  repeatPenalty: z.number().min(0.5).max(3).default(1).volatile(),
  shutdownGraceMillis: z.number().step(1).min(0).max(60_000).default(3000),
  importedModelsFile: z.string(),
})

/**
 * Engine executable file name.
 * @param platform - target platform.
 * @returns the file name for that platform.
 */
export function engineFileName(platform: NodeJS.Platform = process.platform): string {
  return platform === 'win32' ? 'dsh-gguf-engine.exe' : 'dsh-gguf-engine'
}

/**
 * Resolve the engine executables to try, in order: the configured path or `$DSH_GGUF_ENGINE` alone; else an engine
 * directory (`engineDirectory` or `$DSH_GGUF_ENGINE_DIR`) offering its `cuda/` then `cpu/` build; else the single
 * executable {@link resolveEnginePath} finds.
 * @param configured - `enginePath` and `engineDirectory` from configuration.
 * @param env - launch environment.
 * @param resourcesPath - Electron `process.resourcesPath`, when running in the desktop app.
 * @returns the candidates, or a reason none exists.
 */
export function resolveEngineCandidates(
  configured: { readonly enginePath?: string | undefined; readonly engineDirectory?: string | undefined },
  env: Readonly<Record<string, string | undefined>>,
  resourcesPath: string | undefined,
): { readonly candidates: readonly EngineCandidate[] } | { readonly reason: string } {
  const explicit = configured.enginePath ?? env[ENGINE_ENV]
  const directory = configured.engineDirectory ?? env[ENGINE_DIR_ENV]
  if ((explicit === undefined || explicit === '') && directory !== undefined && directory !== '') {
    const root = resolve(expandHomePath(directory))
    const candidates = (['cuda', 'cpu'] as const)
      .map(accelerator => ({ path: join(root, accelerator, engineFileName()), accelerator }))
      .filter(candidate => existsSync(candidate.path))
    return candidates.length > 0 ? { candidates } : { reason: `no local model engine in ${root}/cuda or ${root}/cpu` }
  }
  const single = resolveEnginePath(explicit, env, resourcesPath)
  return 'reason' in single ? single : { candidates: [{ path: single.path, accelerator: 'unknown' }] }
}

/**
 * Resolve one engine executable: the configured path, then `$DSH_GGUF_ENGINE`,
 * then a desktop resources copy, then a repository release build.
 * @param configured - `enginePath` from configuration.
 * @param env - launch environment.
 * @param resourcesPath - Electron `process.resourcesPath`, when running in the desktop app.
 * @returns the absolute executable path, or a reason no executable was found.
 */
export function resolveEnginePath(
  configured: string | undefined,
  env: Readonly<Record<string, string | undefined>>,
  resourcesPath: string | undefined,
): { readonly path: string } | { readonly reason: string } {
  const explicit = configured ?? env[ENGINE_ENV]
  if (explicit !== undefined && explicit !== '') {
    const path = resolve(expandHomePath(explicit))
    return existsSync(path) ? { path } : { reason: `local model engine not found at ${path}` }
  }
  const candidates: string[] = []
  if (resourcesPath !== undefined) candidates.push(join(resourcesPath, 'gguf-engine', engineFileName()))
  // Source checkouts: native/gguf-engine/target/release beside packages/.
  let dir = dirname(fileURLToPath(import.meta.url))
  for (let depth = 0; depth < 6; depth++) {
    candidates.push(join(dir, 'native', 'gguf-engine', 'target', 'release', engineFileName()))
    dir = dirname(dir)
  }
  const found = candidates.find(existsSync)
  return found === undefined
    ? { reason: `local model engine not found; build it with \`cargo build --release\` in native/gguf-engine or set ${ENGINE_ENV}` }
    : { path: found }
}

/** Where the catalog comes from: collections under one root, or explicitly configured directories. */
export type ModelSource =
  | { readonly kind: 'collections'; readonly root: string }
  | { readonly kind: 'directories'; readonly directories: readonly string[] }

/**
 * Resolve the model source: a nonempty `modelDirectories` list keeps those directories; otherwise collections live
 * under `modelsRoot`, then `$DSH_LOCAL_MODELS_ROOT`, then `$DSH_HOME/models`.
 * @param configured - `modelDirectories` and `modelsRoot` from configuration.
 * @param env - launch environment.
 * @returns the source.
 */
export function resolveModelSource(
  configured: { readonly modelDirectories?: readonly string[] | undefined; readonly modelsRoot?: string | undefined },
  env: Readonly<Record<string, string | undefined>>,
): ModelSource {
  if (configured.modelDirectories !== undefined && configured.modelDirectories.length > 0) {
    return { kind: 'directories', directories: resolveModelDirectories(configured.modelDirectories, env) }
  }
  const root = configured.modelsRoot ?? env[MODELS_ROOT_ENV]
  return {
    kind: 'collections',
    root: root === undefined || root === '' ? join(resolveDshHome(undefined, env), 'models') : resolve(expandHomePath(root)),
  }
}

/**
 * Resolve the scanned model directories.
 * @param configured - `modelDirectories` from configuration.
 * @param env - launch environment, read for `DSH_HOME`.
 * @returns absolute, de-duplicated directories in configuration order, or `$DSH_HOME/models` when none are configured.
 */
export function resolveModelDirectories(
  configured: readonly string[] | undefined,
  env: Readonly<Record<string, string | undefined>>,
): string[] {
  // The schema materializes an omitted list as `[]`; an empty list therefore selects the default directory.
  const dirs = configured === undefined || configured.length === 0 ? [join(resolveDshHome(undefined, env), 'models')] : configured
  const out: string[] = []
  for (const dir of dirs) {
    const expanded = expandHomePath(dir)
    const absolute = isAbsolute(expanded) ? expanded : resolve(expanded)
    if (!out.includes(absolute)) out.push(absolute)
  }
  return out
}

/**
 * Read the live sampling controls.
 * @param config - live configuration.
 * @returns wire sampling controls.
 */
export function samplingOf(config: Config): EngineSampling {
  return {
    temperature: config.temperature.get(),
    topK: config.topK.get(),
    topP: config.topP.get(),
    minP: config.minP.get(),
    repeatPenalty: config.repeatPenalty.get(),
    repeatLastN: 64,
  }
}

/**
 * Read the load-time compute placement.
 * @param config - validated configuration.
 * @returns wire placement; `gpuLayers` is omitted when every layer should be offloaded.
 */
export function placementOf(config: Config): EnginePlacement {
  return config.gpuLayers === undefined ? { device: config.device } : { device: config.device, gpuLayers: config.gpuLayers }
}

/**
 * Resolve the local models state file (the active collection), kept beside the imported-models list.
 * @param importsFile - resolved imported-models list.
 * @returns the absolute file path.
 */
export function resolveStateFile(importsFile: string): string {
  return join(dirname(importsFile), 'state.json')
}

/**
 * Resolve the imported-models list.
 * @param configured - `importedModelsFile` from configuration.
 * @param env - launch environment, read for `DSH_HOME`.
 * @returns the absolute file path.
 */
export function resolveImportedModelsFile(configured: string | undefined, env: Readonly<Record<string, string | undefined>>): string {
  if (configured !== undefined && configured !== '') return resolve(expandHomePath(configured))
  return join(resolveDshHome(undefined, env), 'local-models', 'imported.json')
}
