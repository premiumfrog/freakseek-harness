/**
 * Local models page state: the Host catalog snapshot plus the pending action.
 *
 * @module dsh-client-ui-settings-local-models/controller
 */

import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type { LocalModelsActionResult, LocalModelsFileResult, LocalModelsSnapshot } from '@deepseek-ai/dsh-api-remotes/client'
import { createSnapshotStore } from '@deepseek-ai/dsh-client-store'
import type { SnapshotStore } from '@deepseek-ai/dsh-client-store'

/** Page state. */
export interface LocalModelsPageState {
  /** Latest Host snapshot; absent before the first read completes. */
  snapshot?: LocalModelsSnapshot
  /** The read itself failed, for example because the Host serves no local models. */
  unavailable: boolean
  /** Model id whose load is in flight, `unload`, or `file` while a chosen file loads. */
  pending?: string
  /** Last action failure, shown until the next action. */
  error?: string
}

/** The desktop's native GGUF file dialog: the chosen absolute path, or `null` when cancelled. */
export interface ModelFilePicker {
  pick(): Promise<string | null>
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

  private async act<R extends LocalModelsActionResult | LocalModelsFileResult>(
    pending: string,
    run: () => Promise<{ ok: true; value: R } | { ok: false; error: { message: string } }>,
  ): Promise<R | undefined> {
    this.store.update((draft) => { draft.pending = pending; delete draft.error })
    const response = await run()
    if (this.disposed) return undefined
    this.sequence++
    this.store.update((draft) => {
      delete draft.pending
      if (!response.ok) { draft.error = response.error.message; return }
      draft.snapshot = response.value.snapshot
      if (!response.value.ok) draft.error = response.value.reason
    })
    return response.ok && response.value.ok ? response.value : undefined
  }

  /**
   * Load a model and make it the default for new conversations.
   * @param modelId - catalog id.
   * @returns resolves once the Host answered.
   */
  async load(modelId: string): Promise<void> {
    await this.act(modelId, () => this.ctx.remote.localModels.load(modelId))
  }

  /**
   * Ask the picker for a GGUF file, load it, and remember it; it becomes the default for new conversations.
   * @param picker - the desktop file dialog.
   * @returns the loaded model id, or `undefined` when cancelled or failed (the failure is on the store).
   */
  async loadFile(picker: ModelFilePicker): Promise<string | undefined> {
    if (this.store.getSnapshot().pending !== undefined) return undefined
    let path: string | null
    try { path = await picker.pick() }
    catch (error) {
      this.store.update((draft) => { draft.error = (error as Error).message })
      return undefined
    }
    if (path === null || this.disposed) return undefined
    const result = await this.act('file', () => this.ctx.remote.localModels.loadFile(path))
    return result?.ok === true ? result.modelId : undefined
  }

  /**
   * Record a failure from a follow-up step, such as switching a conversation to the loaded model.
   * @param message - failure text.
   */
  fail(message: string): void {
    if (!this.disposed) this.store.update((draft) => { draft.error = message })
  }

  /**
   * Release the loaded model.
   * @returns resolves once the Host answered.
   */
  async unload(): Promise<void> {
    await this.act('unload', () => this.ctx.remote.localModels.unload())
  }

  /** Ignore answers arriving after deactivation. */
  dispose(): void {
    this.disposed = true
  }
}
