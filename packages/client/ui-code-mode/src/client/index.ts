/**
 * @deepseek-ai/dsh-client-ui-code-mode — a Session header toggle that turns
 * the conversation into a coding layout: the right Sidebar expanded with the
 * workspace files page and a terminal pane beside the transcript.
 *
 * @module dsh-client-ui-code-mode
 */
import type { Context as ClientContext } from '@deepseek-ai/cordis'
import type { ShortcutCommandId } from '@deepseek-ai/dsh-client-shortcuts/client'
import type {} from '@deepseek-ai/dsh-client-locale/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type {} from '@deepseek-ai/dsh-client-ui-sidebar-right/client'
import { CODE_MODE_COMMAND, CodeModeButton, type CodeModeButtonInjected } from './CodeModeButton.tsx'
import { CodeModeController } from './controller.ts'
import { de, en, es, fr, ja, NS, pt, ru, zh } from './locales.ts'

export { CODE_MODE_COMMAND, CodeModeButton } from './CodeModeButton.tsx'
export type { CodeModeButtonInjected, CodeModeButtonProps } from './CodeModeButton.tsx'
export { CodeModeController, FILES_KIND, TERMINAL_KIND } from './controller.ts'
export type { CodeModeKindAvailable, CodeModeSidebar } from './controller.ts'
export { NS } from './locales.ts'
export type { CodeModeLocaleKey } from './locales.ts'

/** Required browser services: slots, copy, the right Sidebar and its tab types, and shortcuts. */
export const inject = ['slots', 'locale', 'sidebarRight', 'sidebarRightTabs', 'shortcuts']

/**
 * Register the header toggle, its shortcut, and the copy.
 * @param ctx - client root context.
 */
export function apply(ctx: ClientContext): void {
  const t = ctx.locale.bind(NS)
  const controller = new CodeModeController(ctx.sidebarRight, kind => ctx.sidebarRightTabs.get(kind) !== undefined)
  ctx.effect(() => ctx.locale.register(NS, { zh, en, ja, es, fr, de, ru, pt }), 'ui-code-mode: dictionaries')
  ctx.effect(() => ctx.shortcuts.register({
    id: CODE_MODE_COMMAND as ShortcutCommandId, label: () => t('shortcut'), aliases: ['code mode', 'code'],
    defaults: {
      'desktop:macos': { code: 'KeyE', modifiers: ['primary', 'shift'] },
      'desktop:windows': { code: 'KeyE', modifiers: ['primary', 'shift'] },
      'desktop:linux': { code: 'KeyE', modifiers: ['primary', 'shift'] },
    },
    regions: ['page', 'editable', 'terminal'], modals: [],
    resolve: () => {
      const target = ctx.sidebarRight.commandTarget(null)
      if (target === undefined) return { status: 'blocked', reason: t('noSession') }
      return { status: 'handled', run: () => { if (ctx.sidebarRight.isTargetCurrent(target)) controller.toggle(target.sessionId) } }
    },
  }), 'ui-code-mode: shortcut')
  const injected: CodeModeButtonInjected = {
    hooks: { active: controller.active, shortcuts: ctx.shortcuts.catalog },
    toggle: (sessionId) => { controller.toggle(sessionId) },
  }
  ctx.effect(() => ctx.slots.inject('conversation.session.header.utilities', () => ctx.slots.register({
    name: 'conversation.session.header.utilities',
    id: 'code-mode',
    // Left of the Schedule catalog (order -5) and the overflow menu (order 0).
    order: -6,
    locale: NS,
    inject: () => injected,
  }, CodeModeButton)), 'ui-code-mode: header toggle')
}
