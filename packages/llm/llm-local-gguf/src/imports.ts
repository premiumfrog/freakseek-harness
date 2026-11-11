/**
 * The persisted list of GGUF files a user loaded from outside the model
 * directories, so they stay in the catalog across restarts.
 *
 * @module dsh-llm-local-gguf/imports
 */

import { mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, normalize } from 'node:path'
import { EngineError } from './engine-process.ts'

/** Catalog id prefix of an imported file; directory ids are relative paths and never take it. */
export const IMPORTED_ID_PREFIX = 'file:'

/** File-format revision of the imports file. */
const IMPORTS_VERSION = 1

/**
 * Catalog id of an imported file: the prefix and its absolute path without `.gguf`, with forward slashes.
 * @param path - absolute GGUF path.
 * @returns the catalog id.
 */
export function importedModelId(path: string): string {
  return IMPORTED_ID_PREFIX + path.split('\\').join('/').replace(/\.gguf$/iu, '')
}

/**
 * Read the imported paths.
 * @param file - imports file.
 * @returns the paths in import order; a missing file holds none.
 * @throws EngineError `IMPORTS_INVALID` when the file is unreadable or not a version-1 path list.
 */
export async function readImports(file: string): Promise<string[]> {
  let text: string
  try { text = await readFile(file, 'utf8') }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw new EngineError('IMPORTS_INVALID', `cannot read imported models list ${file}: ${(error as Error).message}`)
  }
  let value: unknown
  try { value = JSON.parse(text) }
  catch (_malformedJson) {
    throw new EngineError('IMPORTS_INVALID', `imported models list ${file} is not valid JSON`)
  }
  const files = typeof value === 'object' && value !== null && 'version' in value && value.version === IMPORTS_VERSION
    && 'files' in value && Array.isArray(value.files) ? value.files : undefined
  if (files === undefined || !files.every((path): path is string => typeof path === 'string' && isAbsolute(path))) {
    throw new EngineError('IMPORTS_INVALID', `imported models list ${file} must be {"version":1,"files":[absolute paths]}`)
  }
  return files
}

/**
 * Validate a GGUF path received over the wire.
 * @param path - candidate path.
 * @returns the normalized absolute path.
 * @throws EngineError `INVALID_MODEL_FILE` unless it names an existing regular `.gguf` file.
 */
export async function validateModelFile(path: unknown): Promise<string> {
  if (typeof path !== 'string' || !isAbsolute(path) || !/\.gguf$/iu.test(path)) {
    throw new EngineError('INVALID_MODEL_FILE', 'choose an absolute path to a .gguf file')
  }
  const normalized = normalize(path)
  let regular = false
  try { regular = (await stat(normalized)).isFile() }
  catch (error) { throw new EngineError('INVALID_MODEL_FILE', `cannot read ${normalized}: ${(error as Error).message}`) }
  if (!regular) throw new EngineError('INVALID_MODEL_FILE', `${normalized} is not a file`)
  return normalized
}

/**
 * Append a path to the imports file unless it is already listed; the write replaces the file atomically.
 * @param file - imports file.
 * @param path - validated absolute GGUF path.
 */
export async function addImport(file: string, path: string): Promise<void> {
  const files = await readImports(file)
  if (files.includes(path)) return
  await writeJsonAtomic(file, { version: IMPORTS_VERSION, files: [...files, path] })
}

/**
 * Write a JSON document by replacing the file atomically, creating its folder when missing.
 * @param file - target file.
 * @param value - JSON-serializable document.
 */
export async function writeJsonAtomic(file: string, value: unknown): Promise<void> {
  await mkdir(dirname(file), { recursive: true })
  const temporary = `${file}.${String(process.pid)}.tmp`
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`)
  await rename(temporary, file)
}
