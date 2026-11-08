/**
 * The Session header's code mode toggle: a pressed button while the Session
 * is in code mode, with the shortcut in its tooltip.
 */
import type { ReactNode } from 'react'
import clsx from 'clsx'
import { Button, IconCodeOutlineRegular, Tooltip } from '@deepseek-ai/dsh-client-ui-primitives'
import type { HostObservable, InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { ShortcutCatalogEntry } from '@deepseek-ai/dsh-client-shortcuts/client'
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import type { SessionId } from '@deepseek-ai/dsh-session/types'
import type { NS } from './locales.ts'
import css from './CodeModeButton.module.css'

/** Injected state and command of the toggle. */
// A type alias, not an interface, so it satisfies the slot runtime's record-typed inject face.
export type CodeModeButtonInjected = {
  readonly hooks: {
    /** Sessions in code mode. */
    readonly active: HostObservable<ReadonlySet<SessionId>>
    /** Effective shortcut bindings, read for the code mode command. */
    readonly shortcuts: HostObservable<readonly ShortcutCatalogEntry[]>
  }
  /**
   * Enter or leave code mode.
   * @param sessionId - the Session whose header holds the toggle.
   */
  readonly toggle: (sessionId: SessionId) => void
}

/** Full props: the header utilities seat, the injected face, and copy. */
export type CodeModeButtonProps =
  & PropsRuntime<'conversation.session.header.utilities'>
  & PropsLocale<typeof NS>
  & InjectFace<CodeModeButtonInjected>

/** Shortcut command id of the toggle. */
export const CODE_MODE_COMMAND = 'workspace.codeMode'

/**
 * Render the toggle.
 * @param props - seat, injected face, and copy.
 * @returns the toggle button.
 */
export function CodeModeButton({ sessionId, useActive, useShortcuts, toggle, t }: CodeModeButtonProps): ReactNode {
  const pressed = useActive(active => active.has(sessionId))
  const shortcut = useShortcuts(entries => entries.find(entry => entry.id === CODE_MODE_COMMAND))
  const label = pressed ? t('leave') : t('enter')
  return (
    <Tooltip label={label} shortcutKeys={shortcut?.keys} side="bottom" delayMs={500}>
      <Button
        size="sm"
        className={clsx(css.button, pressed && css.pressed)}
        aria-pressed={pressed}
        aria-label={label}
        aria-keyshortcuts={shortcut?.aria}
        data-code-mode-toggle
        icon={<IconCodeOutlineRegular />}
        onClick={() => { toggle(sessionId) }}
      >
        {t('label')}
      </Button>
    </Tooltip>
  )
}
