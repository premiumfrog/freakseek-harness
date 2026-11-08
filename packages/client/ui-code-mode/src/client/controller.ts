/**
 * Code mode for one Session: the right Sidebar expanded with the workspace
 * files page and a terminal pane beside the conversation.
 *
 * @module dsh-client-ui-code-mode/controller
 */

import { createSnapshotStore, type ObservableSnapshot } from '@deepseek-ai/dsh-client-store'
import type { ISidebarRight, SidebarRightOpenTab } from '@deepseek-ai/dsh-client-ui-sidebar-right/client'
import type { SessionId } from '@deepseek-ai/dsh-session/types'

/** Tab kind of the workspace files page (`@deepseek-ai/dsh-client-ui-sidebar-files`). */
export const FILES_KIND = 'files'
/** Tab kind of an interactive terminal (`@deepseek-ai/dsh-client-ui-sidebar-terminal`). */
export const TERMINAL_KIND = 'terminal'

/** The right-Sidebar operations code mode drives. */
export type CodeModeSidebar = Pick<ISidebarRight, 'openTab' | 'focus' | 'isExpanded' | 'toggleExpanded'> & {
  /** Open tab metadata across Sessions, read to reuse an existing terminal. */
  readonly openTabs: ObservableSnapshot<readonly SidebarRightOpenTab[]>
}

/** Whether a tab kind has a registered type; code mode skips panes nobody provides. */
export type CodeModeKindAvailable = (kind: string) => boolean

/** Session-keyed code mode state and its enter and leave operations. */
export class CodeModeController {
  private readonly store = createSnapshotStore<ReadonlySet<SessionId>>(new Set())
  /** Sessions currently in code mode. */
  readonly active: ObservableSnapshot<ReadonlySet<SessionId>> = this.store

  /**
   * @param sidebar - the on-screen Session's right Sidebar.
   * @param available - whether a tab kind is registered.
   */
  constructor(private readonly sidebar: CodeModeSidebar, private readonly available: CodeModeKindAvailable) {}

  /**
   * Enter code mode when the Session is out of it or its column was collapsed since; otherwise leave it.
   * @param sessionId - the on-screen Session.
   */
  toggle(sessionId: SessionId): void {
    if (this.store.getSnapshot().has(sessionId) && this.sidebar.isExpanded()) this.leave(sessionId)
    else this.enter(sessionId)
  }

  /**
   * Expand the column with the files page and a terminal pane, revealing open ones instead of duplicating them.
   * @param sessionId - the on-screen Session.
   */
  enter(sessionId: SessionId): void {
    // Opening a docked page expands the column in the same step.
    if (this.available(FILES_KIND)) this.sidebar.openTab(FILES_KIND)
    else if (!this.sidebar.isExpanded()) this.sidebar.toggleExpanded()
    if (this.available(TERMINAL_KIND)) {
      const terminal = this.sidebar.openTabs.getSnapshot().find(tab => tab.sessionId === sessionId && tab.kind === TERMINAL_KIND)
      // Terminal tabs allow several copies, so an open one is focused rather than opened again.
      if (terminal === undefined) this.sidebar.openTab(TERMINAL_KIND, { preferNewPane: true })
      else this.sidebar.focus(terminal.tabId)
    }
    this.store.set(new Set([...this.store.getSnapshot(), sessionId]))
  }

  /**
   * Collapse the column and mark the Session out of code mode; open tabs stay for the next entry.
   * @param sessionId - the on-screen Session.
   */
  leave(sessionId: SessionId): void {
    if (this.sidebar.isExpanded()) this.sidebar.toggleExpanded()
    this.store.set(new Set([...this.store.getSnapshot()].filter(id => id !== sessionId)))
  }
}
