/**
 * Model collections: the named subfolders of the models root, the persisted
 * active collection, and copying a chosen GGUF file into one.
 *
 * @module dsh-llm-local-gguf/collections
 */

import { copyFile, mkdir, open, readdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { basename, extname, join } from 'node:path'
import { EngineError } from './engine-process.ts'
import { writeJsonAtomic } from './imports.ts'

/** Id of the models root itself, which holds loose files as the default collection. */
export const DEFAULT_COLLECTION_ID = ''

/** File-format revision of the state file. */
const STATE_VERSION = 1

/** Windows device names, which cannot name a folder with or without an extension. */
const RESERVED = /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$/iu

/**
 * Validate a collection name received over the wire.
 * @param name - candidate folder name.
 * @returns the trimmed name.
 * @throws EngineError `INVALID_COLLECTION` unless it is a portable single folder name.
 */
export function validateCollectionName(name: unknown): string {
  const trimmed = typeof name === 'string' ? name.trim() : ''
  const control = Array.from({ length: trimmed.length }, (_, index) => trimmed.charCodeAt(index)).some(code => code < 32)
  if (trimmed.length === 0 || trimmed.length > 64 || /[<>:"/\\|?*]/u.test(trimmed) || control
    || trimmed.startsWith('.') || trimmed.endsWith('.') || RESERVED.test(trimmed)) {
    throw new EngineError('INVALID_COLLECTION', 'use 1–64 characters without < > : " / \\ | ? *, a leading or trailing dot, or a Windows device name')
  }
  return trimmed
}

/**
 * Named collections under the models root, sorted by name; the root's own files are the default collection.
 * @param root - models root.
 * @returns collection ids (folder names); a missing root holds none.
 */
export async function collectionIds(root: string): Promise<string[]> {
  let entries
  try { entries = await readdir(root, { withFileTypes: true }) }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw error
  }
  return entries.filter(e => e.isDirectory() && !e.name.startsWith('.')).map(e => e.name).sort((a, b) => a.localeCompare(b))
}

/**
 * Read the active collection id.
 * @param file - state file.
 * @returns the saved id, or the default collection when none is saved.
 * @throws EngineError `STATE_INVALID` when the file is unreadable or malformed.
 */
export async function readActiveCollection(file: string): Promise<string> {
  let text: string
  try { text = await readFile(file, 'utf8') }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return DEFAULT_COLLECTION_ID
    throw new EngineError('STATE_INVALID', `cannot read local models state ${file}: ${(error as Error).message}`)
  }
  let value: unknown
  try { value = JSON.parse(text) }
  catch (_malformedJson) { throw new EngineError('STATE_INVALID', `local models state ${file} is not valid JSON`) }
  const active = typeof value === 'object' && value !== null && 'version' in value && value.version === STATE_VERSION
    && 'activeCollection' in value ? value.activeCollection : undefined
  if (typeof active !== 'string') {
    throw new EngineError('STATE_INVALID', `local models state ${file} must be {"version":1,"activeCollection":"<name>"}`)
  }
  return active
}

/**
 * Save the active collection id; the write replaces the file atomically.
 * @param file - state file.
 * @param id - collection id.
 */
export async function writeActiveCollection(file: string, id: string): Promise<void> {
  await writeJsonAtomic(file, { version: STATE_VERSION, activeCollection: id })
}

/**
 * Copy a GGUF file into a collection folder. A file there with the same name, size, and first and last MiB is reused;
 * another file of the same name gets a numbered name. The copy lands under a temporary name and is renamed once complete, so a scan never
 * lists a partial file.
 * @param source - validated absolute GGUF path.
 * @param directory - collection folder; created when missing.
 * @returns the path of the copy.
 */
export async function copyIntoCollection(source: string, directory: string): Promise<string> {
  await mkdir(directory, { recursive: true })
  const size = (await stat(source)).size
  const stem = basename(source, extname(source))
  for (let n = 1; ; n++) {
    const target = join(directory, n === 1 ? `${stem}.gguf` : `${stem} (${String(n)}).gguf`)
    if (target === source) return target
    let existing: number | undefined
    try { existing = (await stat(target)).size }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
    }
    if (existing === size && await sameEnds(source, target, size)) return target
    if (existing !== undefined) continue
    const partial = `${target}.${String(process.pid)}.part`
    try {
      await copyFile(source, partial)
      await rename(partial, target)
    } catch (error) {
      await rm(partial, { force: true })
      throw new EngineError('IMPORT_FAILED', `cannot copy ${source} into ${directory}: ${(error as Error).message}`)
    }
    return target
  }
}

/** Bytes compared at each end of two equal-size files, enough to tell model files apart without reading gigabytes. */
const END_BYTES = 1 << 20

/**
 * Whether two files of `size` bytes have the same first and last {@link END_BYTES}.
 * @param a - first file.
 * @param b - second file.
 * @param size - their common size.
 * @returns true when both ends match.
 */
async function sameEnds(a: string, b: string, size: number): Promise<boolean> {
  const length = Math.min(END_BYTES, size)
  const read = async (path: string, position: number): Promise<Buffer> => {
    const handle = await open(path, 'r')
    try {
      const buffer = Buffer.alloc(length)
      await handle.read(buffer, 0, length, position)
      return buffer
    } finally { await handle.close() }
  }
  for (const position of [0, size - length]) {
    if (!(await read(a, position)).equals(await read(b, position))) return false
  }
  return true
}
