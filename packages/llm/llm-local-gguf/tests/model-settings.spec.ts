/** Per-model settings: validation, the saved document, and its failure reports. */

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, it } from 'vitest'
import { parseModelSettings, readModelSettings, writeModelSettings } from '../src/model-settings.ts'

const dirs: string[] = []
afterEach(async () => { for (const dir of dirs.splice(0)) await rm(dir, { recursive: true, force: true }) })

async function scratch(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-model-settings-'))
  dirs.push(dir)
  return join(dir, 'model-settings.json')
}

it('keeps valid fields, drops invalid ones, and asks on load by default', () => {
  expect(parseModelSettings({ contextLength: 65536.4, gpuLayers: 20, temperature: 0.6, topP: 2, topK: -1, thinking: false, extra: 1 }))
    .toEqual({ askOnLoad: true, contextLength: 65536, gpuLayers: 20, temperature: 0.6, thinking: false })
  expect(parseModelSettings({ askOnLoad: false, contextLength: 100 })).toEqual({ askOnLoad: false })
  expect(parseModelSettings(null)).toEqual({ askOnLoad: true })
})

it('round-trips the document and reads a missing file as empty', async () => {
  const file = await scratch()
  expect(await readModelSettings(file)).toEqual({ models: {} })
  await writeModelSettings(file, { models: { '/m/a.gguf': { askOnLoad: false, contextLength: 8192 } }, lastLoaded: '/m/a.gguf' })
  expect(JSON.parse(await readFile(file, 'utf8'))).toMatchObject({ version: 1, lastLoaded: '/m/a.gguf' })
  expect(await readModelSettings(file)).toEqual({ models: { '/m/a.gguf': { askOnLoad: false, contextLength: 8192 } }, lastLoaded: '/m/a.gguf' })
})

it('reports malformed files', async () => {
  const file = await scratch()
  await writeFile(file, '{')
  await expect(readModelSettings(file)).rejects.toThrow('is not valid JSON')
  await writeFile(file, '{"version":2}')
  await expect(readModelSettings(file)).rejects.toThrow('must be {"version":1')
})
