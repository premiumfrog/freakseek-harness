/** A Loader-booted `local` route lists GGUF files and streams replies from the real engine process. */

import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
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

async function boot(): Promise<{ ctx: Context; models: string }> {
  root = await mkdtemp(join(tmpdir(), 'dsh-local-gguf-'))
  const models = join(root, 'models')
  await mkdir(join(models, 'family'), { recursive: true })
  await writeBigramModel(join(models, 'family', 'bigram-hi.gguf'), 'Hi!')
  await writeFile(join(models, 'not-a-model.gguf'), 'garbage')
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    '- id: llm',
    "  name: '@deepseek-ai/dsh-llm'",
    '- id: llm-local-gguf',
    "  name: '@deepseek-ai/dsh-llm-local-gguf'",
    '  config:',
    `    enginePath: ${JSON.stringify(ENGINE)}`,
    `    modelDirectories: [${JSON.stringify(models)}]`,
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

it('stops the engine process when the plugin is disposed', async () => {
  const { ctx } = await boot()
  const service = ctx.localModels
  await service.load('family/bigram-hi')
  await context?.fiber.dispose()
  context = undefined
  expect(await service.load('family/bigram-hi')).toMatchObject({ ok: false, reason: 'local model provider is shutting down' })
})
