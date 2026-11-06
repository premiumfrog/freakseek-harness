/**
 * Local models page state: the Host catalog snapshot plus the pending action.
 *
 * @module dsh-client-ui-settings-local-models/controller
 */

import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type { LocalModelsActionResult, LocalModelsSnapshot } from '@deepseek-ai/dsh-api-remotes/client'
import { createSnapshotStore } from '@deepseek-ai/dsh-client-store'
import type { SnapshotStore } from '@deepseek-ai/dsh-client-store'

/** Page state. */
export interface LocalModelsPageState {
  /** Latest Host snapshot; absent before the first read completes. */
  snapshot?: LocalModelsSnapshot
  /** The read itself failed, for example because the Host serves no local models. */
  unavailable: boolean
  /** Model id whose load is in flight, or `unload`. */
  pending?: string
  /** Last action failure, shown until the next action. */
  error?: string
}

/** Owner of the page store; one per plugin activation. */
export class LocalModelsController {
  /** Store bound to the section's `models` hook. */
  readonly store: SnapshotStore<LocalModelsPageState> = createSnapshotStore<LocalModelsPageState>({ unavailable: false })
  private sequence = 0
  private disposed = false

  /** @param ctx - client context carrying `remote`. */
  constructor(private readonly ctx: Context) {}

  /** Re-read the Host snapshot; later reads supersede earlier ones. */
  refresh(): void {
    void this.read()
  }

  private async read(): Promise<void> {
    const sequence = ++this.sequence
    const response = await this.ctx.remote.localModels.snapshot()
    if (this.disposed || sequence !== this.sequence) return
    if (response.ok) this.store.update((draft) => { draft.snapshot = response.value; draft.unavailable = false })
    else this.store.update((draft) => { draft.unavailable = true })
  }

  private async act(pending: string, run: () => ReturnType<typeof this.ctx.remote.localModels.load>): Promise<void> {
    this.store.update((draft) => { draft.pending = pending; delete draft.error })
    const response = await run()
    if (this.disposed) return
    this.sequence++
    this.store.update((draft) => {
      delete draft.pending
      if (!response.ok) { draft.error = response.error.message; return }
      const result: LocalModelsActionResult = response.value
      draft.snapshot = result.snapshot
      if (!result.ok) draft.error = result.reason
    })
  }

  /**
   * Load a model and make it the default for new conversations.
   * @param modelId - catalog id.
   * @returns resolves once the Host answered.
   */
  load(modelId: string): Promise<void> {
    return this.act(modelId, () => this.ctx.remote.localModels.load(modelId))
  }

  /**
   * Release the loaded model.
   * @returns resolves once the Host answered.
   */
  unload(): Promise<void> {
    return this.act('unload', () => this.ctx.remote.localModels.unload())
  }

  /** Ignore answers arriving after deactivation. */
  dispose(): void {
    this.disposed = true
  }
}
