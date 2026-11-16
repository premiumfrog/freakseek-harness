/**
 * Per-model load and sampling settings, and the last loaded model, saved in one JSON file.
 *
 * @module dsh-llm-local-gguf/model-settings
 */

import { readFile } from 'node:fs/promises'
import { EngineError } from './engine-process.ts'
import { writeJsonAtomic } from './imports.ts'
import type { LocalModelSettings } from './types.ts'

const VERSION = 1

/** Saved document: settings by absolute model path, and the path loaded last. */
export interface ModelSettingsDocument {
  readonly models: Readonly<Record<string, LocalModelSettings>>
  readonly lastLoaded?: string
}

function finite(value: unknown, min: number, max: number): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max ? value : undefined
}

/**
 * Validate settings received from a configuration surface or read from disk; invalid fields are dropped.
 * @param value - untrusted settings object.
 * @returns the valid fields; `askOnLoad` defaults to true.
 */
export function parseModelSettings(value: unknown): LocalModelSettings {
  const v = typeof value === 'object' && value !== null ? value as Record<string, unknown> : {}
  const contextLength = finite(v.contextLength, 256, 1_048_576)
  const gpuLayers = finite(v.gpuLayers, 0, 10_000)
  const temperature = finite(v.temperature, 0, 5)
  const topP = finite(v.topP, 0, 1)
  const topK = finite(v.topK, 0, 100_000)
  return {
    askOnLoad: typeof v.askOnLoad === 'boolean' ? v.askOnLoad : true,
    ...contextLength === undefined ? {} : { contextLength: Math.round(contextLength) },
    ...gpuLayers === undefined ? {} : { gpuLayers: Math.round(gpuLayers) },
    ...temperature === undefined ? {} : { temperature },
    ...topP === undefined ? {} : { topP },
    ...topK === undefined ? {} : { topK: Math.round(topK) },
    ...typeof v.thinking === 'boolean' ? { thinking: v.thinking } : {},
  }
}

/**
 * Read the settings file.
 * @param file - settings file.
 * @returns the document; empty when the file does not exist.
 * @throws EngineError `STATE_INVALID` when the file is unreadable or not a settings document.
 */
export async function readModelSettings(file: string): Promise<ModelSettingsDocument> {
  let text: string
  try { text = await readFile(file, 'utf8') }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { models: {} }
    throw new EngineError('STATE_INVALID', `cannot read model settings ${file}: ${(error as Error).message}`)
  }
  let value: unknown
  try { value = JSON.parse(text) }
  catch (_malformedJson) { throw new EngineError('STATE_INVALID', `model settings ${file} is not valid JSON`) }
  if (typeof value !== 'object' || value === null || !('version' in value) || value.version !== VERSION) {
    throw new EngineError('STATE_INVALID', `model settings ${file} must be {"version":1,"models":{...}}`)
  }
  const doc = value as { models?: unknown; lastLoaded?: unknown }
  const models: Record<string, LocalModelSettings> = {}
  if (typeof doc.models === 'object' && doc.models !== null) {
    for (const [path, settings] of Object.entries(doc.models)) models[path] = parseModelSettings(settings)
  }
  return { models, ...typeof doc.lastLoaded === 'string' ? { lastLoaded: doc.lastLoaded } : {} }
}

/**
 * Save the settings file; the write replaces it atomically.
 * @param file - settings file.
 * @param doc - document to save.
 */
export async function writeModelSettings(file: string, doc: ModelSettingsDocument): Promise<void> {
  await writeJsonAtomic(file, { version: VERSION, ...doc })
}
