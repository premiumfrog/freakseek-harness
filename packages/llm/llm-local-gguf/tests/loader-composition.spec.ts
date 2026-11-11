/** A Loader-booted `local` route lists GGUF files and streams replies from the real engine process. */

import { existsSync } from 'node:fs'
import { chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { afterEach, beforeAll, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import type { ModuleLoaderV2 } from '@deepseek-ai/cordis-plugin-loader'
import Include from '@deepseek-ai/cordis-plugin-include'
import LlmRuntime, { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { Branded } from '@deepseek-ai/dsh-brand'
import * as LocalGguf from '../src/index.ts'
import { engineFileName } from '../src/config.ts'
import { writeBigramModel } from './gguf-fixture.ts'

const ENGINE = fileURLToPath(new URL(`../../../../native/gguf-engine/target/release/${engineFileName()}`, import.meta.url))

let root: string | undefined
let context: Context | undefined

beforeAll(() => {
  // `pnpm run test` builds the engine first (`build:gguf-engine`).
  if (!existsSync(ENGINE)) throw new Error(`missing ${ENGINE}; run pnpm run build:gguf-engine`)
})

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
})

function moduleLoader(modules: ReadonlyMap<string, unknown>): ModuleLoaderV2 {
  return {
    version: 'v2',
    import: (specifier: string) => {
      if (!modules.has(specifier)) return Promise.reject(new Error(`unexpected Loader import: ${specifier}`))
      return Promise.resolve(modules.get(specifier))
    },
    loadCache: new Map(),
    register(): never { throw new Error('unexpected module hook registration') },
    getOrCreateModuleJob(): never { throw new Error('unexpected module job creation') },
    resolveSync(): never { throw new Error('unexpected synchronous module resolution') },
    load(): never { throw new Error('unexpected module load') },
  }
}

/** Boot variants: collections under a models root instead of explicit directories, and engine-config lines. */
interface BootOptions {
  readonly collections?: boolean
  readonly engine?: readonly string[]
  /** Reuse this root instead of a fresh one, as a restart does. */
  readonly root?: string
}

async function boot(options: BootOptions = {}): Promise<{ ctx: Context; models: string }> {
  root = options.root ?? await mkdtemp(join(tmpdir(), 'dsh-local-gguf-'))
  const models = join(root, 'models')
  if (options.root === undefined) {
    await mkdir(join(models, 'family'), { recursive: true })
    await writeBigramModel(join(models, 'family', 'bigram-hi.gguf'), 'Hi!')
    await writeFile(join(models, 'not-a-model.gguf'), 'garbage')
  }
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    '- id: llm',
    "  name: '@deepseek-ai/dsh-llm'",
    '- id: llm-local-gguf',
    "  name: '@deepseek-ai/dsh-llm-local-gguf'",
    '  config:',
    ...options.engine ?? [`    enginePath: ${JSON.stringify(ENGINE)}`],
    options.collections === true ? `    modelsRoot: ${JSON.stringify(models)}` : `    modelDirectories: [${JSON.stringify(models)}]`,
    `    importedModelsFile: ${JSON.stringify(join(root, 'imported.json'))}`,
    '    contextLength: 512',
    '    temperature: 0',
    '    threads: 2',
    '',
  ].join('\n'))
  const modules = new Map<string, unknown>([
    ['@deepseek-ai/dsh-llm', LlmRuntime],
    ['@deepseek-ai/dsh-llm-local-gguf', LocalGguf],
  ])
  for (const name of modules.keys()) {
    const dir = join(root, 'node_modules', ...name.split('/'))
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'package.json'), `${JSON.stringify({ name, version: '0.0.0', type: 'module' })}\n`)
  }
  const ctx = new Context()
  context = ctx
  ctx.baseUrl = pathToFileURL(root).href + '/'
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  ctx.loader.internal = moduleLoader(modules)
  await ctx.loader.create({ name: 'cordis:include', config: { path: pathToFileURL(configPath).href } })
  await ctx.loader.await()
  return { ctx, models }
}

async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const out: StreamChunk[] = []
  for await (const chunk of stream) out.push(chunk)
  return out
}

const request = (text: string) => ({
  provider: 'local',
  model: 'family/bigram-hi',
  messages: [createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] })],
})

it('lists supported GGUF files and streams a reply with prompt-cache usage', async () => {
  const { ctx } = await boot()
  expect(ctx.llm.listProviders().map(p => p.id)).toContain('local')
  const updates: string[] = []
  ctx.on('llm/models-updated', (provider) => { updates.push(provider) })
  const listed = await ctx.llm.listModels('local')
  // A rescan that finds the same files announces nothing, so catalog readers do not loop.
  await ctx.llm.listModels('local')
  expect(updates).toEqual(['local'])
  expect(listed.map(m => [m.id, m.name])).toEqual([['family/bigram-hi', 'Bigram fixture (F32)']])

  const first = await collect(ctx.llm.stream(request('hello')))
  const assembler = new BlockAssembler()
  for (const chunk of first) assembler.push(chunk)
  expect(assembler.blocks()).toEqual([{ type: 'text', text: 'Hi!' }])
  expect(first.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })
  const usage = first.find(c => c.type === 'usage')
  expect(usage).toMatchObject({ usage: { cacheReadTokens: 0, outputTokens: 4 } })

  // The identical prompt reuses every cached position but the last.
  const second = await collect(ctx.llm.stream(request('hello')))
  const again = second.find(c => c.type === 'usage')
  expect(again).toMatchObject({ usage: { inputTokens: 1 } })

  const snapshot = await ctx.localModels.snapshot()
  expect(snapshot.state).toEqual({ kind: 'loaded', modelId: 'family/bigram-hi', contextLength: 512, device: 'cpu', gpuLayers: 0 })
  expect(snapshot.models.find(m => m.id === 'not-a-model')).toMatchObject({ supported: false })
  expect((await ctx.localModels.unload()).snapshot.state).toEqual({ kind: 'idle' })
})

it('keeps each conversation prefix cached across interleaved conversations', async () => {
  const { ctx } = await boot()
  const inSession = (id: string, text: string) => ({ ...request(text), sessionId: brandString<Branded<'SessionId'>>(id) })
  const usageOf = async (options: ReturnType<typeof inSession>) => (await collect(ctx.llm.stream(options))).find(c => c.type === 'usage')
  await usageOf(inSession('session-a', 'a long first conversation'))
  expect(await usageOf(inSession('session-b', 'another conversation'))).toMatchObject({ usage: { cacheReadTokens: 0 } })
  expect(await usageOf(inSession('session-a', 'a long first conversation'))).toMatchObject({ usage: { inputTokens: 1 } })
})

it('reports an unknown model as a typed failure', async () => {
  const { ctx } = await boot()
  const chunks = await collect(ctx.llm.stream({ ...request('x'), model: 'missing' }))
  expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'error', failure: { code: 'LOCAL_UNKNOWN_MODEL' } } })
})

it('answers the default model id with the only supported model and names the choices when there are several', async () => {
  const { ctx, models } = await boot()
  const only = await collect(ctx.llm.stream({ ...request('hello'), model: 'default' }))
  expect(only.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })
  await writeBigramModel(join(models, 'second.gguf'), 'Yo!')
  const several = await collect(ctx.llm.stream({ ...request('hello'), model: 'default' }))
  expect(several.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'error', failure: { code: 'LOCAL_UNKNOWN_MODEL' } } })
  expect(JSON.stringify(several.at(-1))).toContain('family/bigram-hi, second')
})

it('loads a GGUF file from outside the model directories and keeps it in the catalog after a restart', async () => {
  const { ctx } = await boot()
  const outside = join(root!, 'elsewhere', 'picked.gguf')
  await mkdir(dirname(outside), { recursive: true })
  await writeBigramModel(outside, 'Picked!')
  const picked = await ctx.localModels.loadFile(outside)
  const id = `file:${outside.split('\\').join('/').replace(/\.gguf$/u, '')}`
  expect(picked).toMatchObject({ ok: true, modelId: id, snapshot: { state: { kind: 'loaded', modelId: id } } })
  const reply = await collect(ctx.llm.stream({ ...request('hello'), model: id }))
  const assembler = new BlockAssembler()
  for (const chunk of reply) assembler.push(chunk)
  expect(assembler.blocks()).toEqual([{ type: 'text', text: 'Picked!' }])
  // Importing the same file again keeps one entry.
  expect(await ctx.localModels.loadFile(outside)).toMatchObject({ ok: true, modelId: id })
  expect(JSON.parse(await readFile(join(root!, 'imported.json'), 'utf8'))).toEqual({ version: 1, files: [outside] })

  await context?.fiber.dispose()
  const first = root!
  const imports = await readFile(join(first, 'imported.json'), 'utf8')
  try {
    const { ctx: restarted } = await boot()
    await writeFile(join(root!, 'imported.json'), imports)
    expect((await restarted.llm.listModels('local')).map(m => m.id)).toContain(id)
  } finally {
    await rm(first, { recursive: true, force: true })
  }
})

it('refuses a path that is not an absolute, existing .gguf file without importing it', async () => {
  const { ctx, models } = await boot()
  for (const path of ['relative.gguf', join(models, 'missing.gguf'), join(models, 'family'), join(root!, 'cordis.yml')]) {
    expect(await ctx.localModels.loadFile(path)).toMatchObject({ ok: false })
  }
  await expect(readFile(join(root!, 'imported.json'), 'utf8')).rejects.toMatchObject({ code: 'ENOENT' })
})

it('stops the engine process when the plugin is disposed', async () => {
  const { ctx } = await boot()
  const service = ctx.localModels
  await service.load('family/bigram-hi')
  await context?.fiber.dispose()
  context = undefined
  expect(await service.load('family/bigram-hi')).toMatchObject({ ok: false, reason: 'local model provider is shutting down' })
})

it('lists the active collection, switches and creates collections, and resolves models across them', async () => {
  const { ctx, models } = await boot({ collections: true })
  const first = await ctx.localModels.snapshot()
  expect(first.collections).toEqual({ root: models, active: '', items: [{ id: '', modelCount: 1 }, { id: 'family', modelCount: 1 }] })
  expect(first.models.map(m => m.id)).toEqual(['not-a-model'])

  const family = await ctx.localModels.setActiveCollection('family')
  expect(family).toMatchObject({ ok: true, snapshot: { collections: { active: 'family' }, directories: [join(models, 'family')] } })
  expect(family.snapshot.models.map(m => [m.id, m.collection])).toEqual([['family/bigram-hi', 'family']])
  expect(JSON.parse(await readFile(join(root!, 'state.json'), 'utf8'))).toEqual({ version: 1, activeCollection: 'family' })

  const coding = await ctx.localModels.createCollection('  coding ')
  expect(coding).toMatchObject({ ok: true, snapshot: { collections: { active: 'coding' }, models: [] } })
  for (const name of ['', 'a/b', '..', 'CON', 'x'.repeat(65), 'trailing.']) {
    expect(await ctx.localModels.createCollection(name)).toMatchObject({ ok: false })
  }
  expect(await ctx.localModels.setActiveCollection('missing')).toMatchObject({ ok: false })
  // A model outside the active collection still answers by id, as a conversation started earlier needs.
  expect(await ctx.localModels.load('family/bigram-hi')).toMatchObject({ ok: true, snapshot: { state: { kind: 'loaded' } } })
})

it('copies an imported GGUF file into the active collection, loads it, and keeps the collection after a restart', async () => {
  const { ctx, models } = await boot({ collections: true })
  await ctx.localModels.createCollection('coding')
  const outside = join(root!, 'elsewhere', 'picked.gguf')
  await mkdir(dirname(outside), { recursive: true })
  await writeBigramModel(outside, 'Picked!')
  const imported = await ctx.localModels.importFile(outside)
  expect(imported).toMatchObject({ ok: true, modelId: 'coding/picked', snapshot: { state: { kind: 'loaded', modelId: 'coding/picked' } } })
  expect(existsSync(join(models, 'coding', 'picked.gguf'))).toBe(true)
  // The same file again reuses the copy; a different file of that name gets a numbered copy.
  expect(await ctx.localModels.importFile(outside)).toMatchObject({ ok: true, modelId: 'coding/picked' })
  await writeBigramModel(outside, 'Changed reply')
  expect(await ctx.localModels.importFile(outside)).toMatchObject({ ok: true, modelId: 'coding/picked (2)' })
  expect(await ctx.localModels.importFile(join(root!, 'cordis.yml'))).toMatchObject({ ok: false })

  const kept = root!
  await context?.fiber.dispose()
  context = undefined
  const { ctx: restarted } = await boot({ collections: true, root: kept })
  expect((await restarted.localModels.snapshot()).collections?.active).toBe('coding')
})

it.skipIf(process.platform === 'win32')('falls back to the CPU engine when the CUDA engine cannot start', async () => {
  const engines = await mkdtemp(join(tmpdir(), 'dsh-local-gguf-engines-'))
  try {
    await mkdir(join(engines, 'cuda'), { recursive: true })
    await mkdir(join(engines, 'cpu'), { recursive: true })
    // A CUDA build on a PC without the NVIDIA driver exits before it can answer.
    await writeFile(join(engines, 'cuda', engineFileName()), '#!/bin/sh\necho "nvcuda.dll was not found" >&2\nexit 1\n')
    await chmod(join(engines, 'cuda', engineFileName()), 0o755)
    await copyFile(ENGINE, join(engines, 'cpu', engineFileName()))
    await chmod(join(engines, 'cpu', engineFileName()), 0o755)
    const { ctx } = await boot({ engine: [`    engineDirectory: ${JSON.stringify(engines)}`] })
    const loaded = await ctx.localModels.load('family/bigram-hi')
    expect(loaded).toMatchObject({ ok: true, snapshot: { engine: { path: join(engines, 'cpu', engineFileName()), accelerator: 'cpu', devices: [] } } })
    expect(loaded.snapshot.engine?.fallbackReason).toContain('the CUDA engine did not start')
  } finally {
    await rm(engines, { recursive: true, force: true })
  }
})
