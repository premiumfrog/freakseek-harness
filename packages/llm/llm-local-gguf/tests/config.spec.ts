/** Engine and model-directory resolution. */
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { readActiveCollection, writeActiveCollection } from '../src/collections.ts'
import {
  Config, engineFileName, placementOf, resolveEngineCandidates, resolveEnginePath, resolveModelDirectories, resolveModelSource,
} from '../src/config.ts'

let root: string | undefined
afterEach(async () => { if (root !== undefined) await rm(root, { recursive: true, force: true }); root = undefined })

it('scans $DSH_HOME/models when no directory is configured, including the schema-materialized empty list', () => {
  expect(resolveModelDirectories(undefined, { DSH_HOME: '/h' })).toEqual([join('/h', 'models')])
  expect(resolveModelDirectories([], { DSH_HOME: '/h' })).toEqual([join('/h', 'models')])
  expect(resolveModelDirectories(['/a', '/a', '/b'], {})).toEqual(['/a', '/b'])
})

it('prefers the configured engine, then $DSH_GGUF_ENGINE, and reports a missing file', async () => {
  root = await mkdtemp(join(tmpdir(), 'dsh-engine-path-'))
  const engine = join(root, 'engine')
  await writeFile(engine, '')
  expect(resolveEnginePath(engine, {}, undefined)).toEqual({ path: engine })
  expect(resolveEnginePath(undefined, { DSH_GGUF_ENGINE: engine }, undefined)).toEqual({ path: engine })
  expect(resolveEnginePath(join(root, 'missing'), {}, undefined)).toEqual({ reason: `local model engine not found at ${join(root, 'missing')}` })
})

it('defaults to automatic device selection with every layer offloaded and rejects unknown devices', () => {
  const config = Config({})
  expect(placementOf(config)).toEqual({ device: 'auto' })
  expect(placementOf(Config({ device: 'cuda:1', gpuLayers: 20 }))).toEqual({ device: 'cuda:1', gpuLayers: 20 })
  expect(() => Config({ device: 'gpu' })).toThrow()
})

it('offers the CUDA engine before the CPU engine from an engine directory, and a single configured engine alone', async () => {
  root = await mkdtemp(join(tmpdir(), 'dsh-engine-dir-'))
  for (const accelerator of ['cuda', 'cpu']) {
    await mkdir(join(root, accelerator))
    await writeFile(join(root, accelerator, engineFileName()), '')
  }
  expect(resolveEngineCandidates({}, { DSH_GGUF_ENGINE_DIR: root }, undefined)).toEqual({ candidates: [
    { path: join(root, 'cuda', engineFileName()), accelerator: 'cuda' },
    { path: join(root, 'cpu', engineFileName()), accelerator: 'cpu' },
  ] })
  const single = join(root, 'cpu', engineFileName())
  expect(resolveEngineCandidates({ enginePath: single, engineDirectory: root }, {}, undefined)).toEqual({ candidates: [{ path: single, accelerator: 'unknown' }] })
  expect(resolveEngineCandidates({ engineDirectory: join(root, 'missing') }, {}, undefined)).toMatchObject({ reason: expect.stringContaining('no local model engine') as unknown })
})

it('uses collections under the models root unless model directories are configured', () => {
  expect(resolveModelSource({}, { DSH_HOME: '/h' })).toEqual({ kind: 'collections', root: join('/h', 'models') })
  expect(resolveModelSource({ modelDirectories: [] }, { DSH_LOCAL_MODELS_ROOT: '/m' })).toEqual({ kind: 'collections', root: '/m' })
  expect(resolveModelSource({ modelDirectories: ['/a'], modelsRoot: '/m' }, {})).toEqual({ kind: 'directories', directories: ['/a'] })
})

it('reads the saved active collection and rejects a malformed state file', async () => {
  root = await mkdtemp(join(tmpdir(), 'dsh-collection-state-'))
  const file = join(root, 'state.json')
  expect(await readActiveCollection(file)).toBe('')
  await writeActiveCollection(file, 'coding')
  expect(await readActiveCollection(file)).toBe('coding')
  await writeFile(file, '{"version":2}')
  await expect(readActiveCollection(file)).rejects.toMatchObject({ code: 'STATE_INVALID' })
})
