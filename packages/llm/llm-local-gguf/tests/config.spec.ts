/** Engine and model-directory resolution. */
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { resolveEnginePath, resolveModelDirectories } from '../src/config.ts'

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
