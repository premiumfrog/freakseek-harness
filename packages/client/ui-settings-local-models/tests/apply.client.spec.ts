/** The Local models section: registration, refresh triggers, and load and unload outcomes. */

import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, vi } from 'vitest'
import { resolveSlotLabel } from '@deepseek-ai/dsh-client-ui-slots'
import { SlotRegistry } from '@deepseek-ai/dsh-client-ui-renderer/client'
import { RemoteError, TestRemote } from '@deepseek-ai/dsh-client-test-runtime'
import { LocaleRuntime } from '@deepseek-ai/dsh-client-locale/client'
import { apply, inject, NS } from '../src/client/index.ts'
import type { LocalModelsSectionInjected } from '../src/client/index.ts'
import { apply as hostApply } from '../src/index.ts'

const model = {
  id: 'qwen/q', name: 'Qwen', path: '/m/qwen/q.gguf', directory: '/m', fileBytes: 2 ** 30, architecture: 'qwen2',
  supported: true, parameterCount: 7e9, tools: true, reasoning: false,
}
const idle = { state: { kind: 'idle' }, directories: ['/m'], models: [model], problems: [] }
const loaded = { ...idle, state: { kind: 'loaded', modelId: 'qwen/q', contextLength: 8192 } }

async function bench() {
  const ctx = new Context()
  await ctx.plugin(SlotRegistry).await()
  const locale = new LocaleRuntime(ctx)
  locale.setLocale('en')
  ctx.provide('locale', locale)
  const snapshot = vi.fn(() => Promise.resolve({ ok: true, value: idle }))
  const load = vi.fn(() => Promise.resolve({ ok: true, value: { ok: true, snapshot: loaded } }))
  const unload = vi.fn(() => Promise.resolve({ ok: false, error: new RemoteError('gateway/internal', 'host gone', {}) }))
  const remote = new TestRemote(ctx, { localModels: { snapshot, load, unload } })
  const slots = ctx.get('slots') as SlotRegistry
  slots.register({ name: 'root', children: { 'settings.section': { kind: 'list', scope: 'root' } } } as never, () => null)
  await ctx.plugin({ inject: [...inject], apply }).await()
  await vi.waitFor(() => { expect(slots.entries('settings.section')).toHaveLength(1) })
  const entry = slots.entries('settings.section')[0]!
  const face = (entry.inject as () => LocalModelsSectionInjected)()
  return { ctx, entry, face, snapshot, load, unload, remote }
}

describe('ui-settings-local-models apply', () => {
  it('keeps the host Loader entry inert and declares its services', () => {
    expect(hostApply).not.toThrow()
    expect(inject).toEqual(['slots', 'locale', 'remote', 'remote.localModels'])
  })

  it('registers the section and reads the catalog', async () => {
    const { entry, face, snapshot } = await bench()
    expect(entry.options).toMatchObject({ id: 'local-models', order: 5 })
    expect(resolveSlotLabel(entry.options.label)).toBe('Local models')
    expect(entry.locale).toBe(NS)
    await vi.waitFor(() => { expect(face.hooks.models.getSnapshot().snapshot).toEqual(idle) })
    expect(snapshot).toHaveBeenCalledTimes(1)
  })

  it('refreshes only for the local route', async () => {
    const { face, snapshot, remote } = await bench()
    await vi.waitFor(() => { expect(face.hooks.models.getSnapshot().snapshot).toBeDefined() })
    remote.emit('llm/models-updated', ['deepseek-official'])
    remote.emit('llm/models-updated', ['local'])
    await vi.waitFor(() => { expect(snapshot).toHaveBeenCalledTimes(2) })
  })

  it('shows the loaded state after load and the reason after a failed unload', async () => {
    const { face, load } = await bench()
    face.load('qwen/q')
    await vi.waitFor(() => { expect(face.hooks.models.getSnapshot().snapshot?.state).toEqual(loaded.state) })
    expect(load).toHaveBeenCalledWith('qwen/q')
    expect(face.hooks.models.getSnapshot().pending).toBeUndefined()
    face.unload()
    await vi.waitFor(() => { expect(face.hooks.models.getSnapshot().error).toBe('host gone') })
  })

  it('reports a Host without local models', async () => {
    const ctx = new Context()
    await ctx.plugin(SlotRegistry).await()
    const locale = new LocaleRuntime(ctx)
    ctx.provide('locale', locale)
    new TestRemote(ctx, { localModels: { snapshot: () => Promise.resolve({ ok: false, error: new RemoteError('gateway/internal', 'absent', {}) }) } })
    const slots = ctx.get('slots') as SlotRegistry
    slots.register({ name: 'root', children: { 'settings.section': { kind: 'list', scope: 'root' } } } as never, () => null)
    await ctx.plugin({ inject: [...inject], apply }).await()
    const face = (slots.entries('settings.section')[0]!.inject as () => LocalModelsSectionInjected)()
    await vi.waitFor(() => { expect(face.hooks.models.getSnapshot().unavailable).toBe(true) })
  })
})
