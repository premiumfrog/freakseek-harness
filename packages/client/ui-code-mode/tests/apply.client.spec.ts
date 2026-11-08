/** Code mode: header registration, entering with files and terminal panes, reuse of open tabs, and leaving. */

import { Context } from '@deepseek-ai/cordis'
import { describe, expect, it, vi } from 'vitest'
import { createSnapshotStore } from '@deepseek-ai/dsh-client-store'
import { SlotRegistry } from '@deepseek-ai/dsh-client-ui-renderer/client'
import { LocaleRuntime } from '@deepseek-ai/dsh-client-locale/client'
import type { SidebarRightOpenTab, TabId } from '@deepseek-ai/dsh-client-ui-sidebar-right/client'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import { apply, CODE_MODE_COMMAND, inject, NS, type CodeModeButtonInjected } from '../src/client/index.ts'
import { apply as hostApply } from '../src/index.ts'

const session = 'session-1' as SessionId

function fakeSidebar(open: readonly SidebarRightOpenTab[] = []) {
  let expanded = false
  const openTab = vi.fn((kind: string) => { if (kind === 'files') expanded = true })
  return {
    openTab,
    focus: vi.fn(),
    isExpanded: () => expanded,
    toggleExpanded: vi.fn(() => { expanded = !expanded }),
    openTabs: createSnapshotStore<readonly SidebarRightOpenTab[]>(open),
    commandTarget: vi.fn(() => ({ sessionId: session })),
    isTargetCurrent: vi.fn(() => true),
  }
}

async function bench(options: { open?: readonly SidebarRightOpenTab[]; kinds?: readonly string[] } = {}) {
  const ctx = new Context()
  await ctx.plugin(SlotRegistry).await()
  const locale = new LocaleRuntime(ctx)
  locale.setLocale('en')
  ctx.provide('locale', locale)
  const sidebar = fakeSidebar(options.open)
  ctx.provide('sidebarRight', sidebar)
  const kinds = options.kinds ?? ['files', 'terminal']
  ctx.provide('sidebarRightTabs', { get: (kind: string) => kinds.includes(kind) ? { kind } : undefined })
  const register = vi.fn(() => () => {})
  ctx.provide('shortcuts', { register, catalog: createSnapshotStore([]) })
  const slots = ctx.get('slots') as SlotRegistry
  slots.register({
    name: 'root', children: { 'conversation.session.header.utilities': { kind: 'list', scope: 'session' } },
  } as never, () => null)
  await ctx.plugin({ inject: [...inject], apply }).await()
  await vi.waitFor(() => { expect(slots.entries('conversation.session.header.utilities')).toHaveLength(1) })
  const entry = slots.entries('conversation.session.header.utilities')[0]!
  const injectFace = entry.inject
  if (injectFace === undefined) throw new Error('code mode registered no inject face')
  const face = injectFace() as CodeModeButtonInjected
  return { ctx, entry, face, sidebar, register }
}

describe('ui-code-mode apply', () => {
  it('keeps the host Loader entry inert and declares its services', () => {
    expect(hostApply).not.toThrow()
    expect(inject).toEqual(['slots', 'locale', 'sidebarRight', 'sidebarRightTabs', 'shortcuts'])
  })

  it('registers the header toggle and its shortcut', async () => {
    const { entry, register } = await bench()
    expect(entry.options).toMatchObject({ id: 'code-mode', order: -6 })
    expect(entry.locale).toBe(NS)
    expect(register).toHaveBeenCalledWith(expect.objectContaining({ id: CODE_MODE_COMMAND }))
  })

  it('enters with the files page and a terminal in a new pane, then leaves by collapsing', async () => {
    const { face, sidebar } = await bench()
    face.toggle(session)
    expect(sidebar.openTab.mock.calls).toEqual([['files'], ['terminal', { preferNewPane: true }]])
    expect(face.hooks.active.getSnapshot().has(session)).toBe(true)
    face.toggle(session)
    expect(sidebar.toggleExpanded).toHaveBeenCalledTimes(1)
    expect(sidebar.isExpanded()).toBe(false)
    expect(face.hooks.active.getSnapshot().has(session)).toBe(false)
  })

  it('focuses an open terminal instead of opening another', async () => {
    const terminal = { sessionId: session, tabId: 'tab-7' as TabId, kind: 'terminal', contentId: 't' }
    const { face, sidebar } = await bench({ open: [terminal] })
    face.toggle(session)
    expect(sidebar.openTab.mock.calls).toEqual([['files']])
    expect(sidebar.focus).toHaveBeenCalledWith('tab-7')
  })

  it('re-enters when the column was collapsed elsewhere and expands without a files provider', async () => {
    const { face, sidebar } = await bench({ kinds: [] })
    face.toggle(session)
    expect(sidebar.openTab).not.toHaveBeenCalled()
    expect(sidebar.isExpanded()).toBe(true)
    sidebar.toggleExpanded()
    face.toggle(session)
    expect(sidebar.isExpanded()).toBe(true)
    expect(face.hooks.active.getSnapshot().has(session)).toBe(true)
  })
})
