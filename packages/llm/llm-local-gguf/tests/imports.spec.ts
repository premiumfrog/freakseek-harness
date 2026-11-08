/** The imported-models list is untrusted durable input and wire input. */
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { addImport, importedModelId, readImports, validateModelFile } from '../src/imports.ts'
import { resolveImportedModelsFile } from '../src/config.ts'

const roots: string[] = []
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }) })
async function root(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-gguf-imports-'))
  roots.push(dir)
  return dir
}

it('names imported files by absolute path with forward slashes and no extension', () => {
  expect(importedModelId('/models/a.GGUF')).toBe('file:/models/a')
  expect(importedModelId('C:\\Models\\q.gguf')).toBe('file:C:/Models/q')
})

it('reads a missing list as empty and rejects malformed or non-version-1 lists', async () => {
  const dir = await root()
  const file = join(dir, 'imported.json')
  expect(await readImports(file)).toEqual([])
  for (const body of ['not json', '[]', '{"version":2,"files":[]}', '{"version":1,"files":["relative.gguf"]}', '{"version":1,"files":[1]}']) {
    await writeFile(file, body)
    await expect(readImports(file)).rejects.toMatchObject({ code: 'IMPORTS_INVALID' })
  }
})

it('appends each path once and creates the list directory', async () => {
  const dir = await root()
  const file = join(dir, 'nested', 'imported.json')
  await addImport(file, '/models/a.gguf')
  await addImport(file, '/models/b.gguf')
  await addImport(file, '/models/a.gguf')
  expect(await readImports(file)).toEqual(['/models/a.gguf', '/models/b.gguf'])
})

it('accepts only absolute paths to existing regular .gguf files', async () => {
  const dir = await root()
  const model = join(dir, 'm.gguf')
  await writeFile(model, 'GGUF')
  await mkdir(join(dir, 'folder.gguf'))
  expect(await validateModelFile(model)).toBe(model)
  for (const path of [42, 'm.gguf', join(dir, 'missing.gguf'), join(dir, 'folder.gguf'), join(dir, 'notes.txt')]) {
    await expect(validateModelFile(path)).rejects.toMatchObject({ code: 'INVALID_MODEL_FILE' })
  }
})

it('resolves the list under $DSH_HOME unless configured', () => {
  expect(resolveImportedModelsFile(undefined, { DSH_HOME: '/home/u/.dsh' })).toBe(join('/home/u/.dsh', 'local-models', 'imported.json'))
  expect(resolveImportedModelsFile('/data/imports.json', {})).toBe('/data/imports.json')
})
