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
import type { EngineSampling } from './types.ts'

/** Environment variable naming the engine executable. */
export const ENGINE_ENV = 'DSH_GGUF_ENGINE'

/** Plugin configuration. */
export interface Config {
  /** Engine executable; omitted resolves through {@link resolveEnginePath}. */
  enginePath?: string
  /** Directories scanned for `.gguf` files, `~`-expanded; omitted or empty scans `$DSH_HOME/models`. */
  modelDirectories: Volatile<string[] | undefined>
  /** Context window requested at load, clamped by the engine to the trained length. */
  contextLength: Volatile<number>
  /** KV cache slots kept at once; each conversation and auxiliary request kind uses its own. */
  cacheSlots: number
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
}

export const Config = z.object({
  enginePath: z.string(),
  modelDirectories: z.array(z.string()).volatile(),
  contextLength: z.number().step(1).min(256).max(1_048_576).default(32768).volatile(),
  cacheSlots: z.number().step(1).min(1).max(64).default(4),
  maxOutputTokens: z.number().step(1).min(1).max(1_048_576).default(4096).volatile(),
  threads: z.number().step(1).min(0).max(1024).default(0),
  temperature: z.number().min(0).max(5).default(0.6).volatile(),
  topK: z.number().step(1).min(0).max(100_000).default(40).volatile(),
  topP: z.number().min(0).max(1).default(0.95).volatile(),
  minP: z.number().min(0).max(1).default(0.05).volatile(),
  repeatPenalty: z.number().min(0.5).max(3).default(1).volatile(),
  shutdownGraceMillis: z.number().step(1).min(0).max(60_000).default(3000),
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
 * Resolve the engine executable: the configured path, then `$DSH_GGUF_ENGINE`,
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
