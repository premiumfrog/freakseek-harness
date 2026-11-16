/**
 * Local models page state: the Host catalog snapshot plus the pending action.
 *
 * @module dsh-client-ui-settings-local-models/controller
 */

import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type {
  LocalModelSettings, LocalModelsActionResult, LocalModelsFileResult, LocalModelsSnapshot, LocalModelsState,
} from '@deepseek-ai/dsh-api-remotes/client'
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
  /** Chosen and dropped files are copied into the active collection before loading; off loads them in place. */
  copyIntoCollection: boolean
  /** Load options awaiting confirmation: the model and the surface showing the dialog. */
  dialog?: { readonly modelId: string; readonly owner: string }
}

/** Session id as the conversation's model selection takes it. */
export type SeatSessionId = Parameters<Context['remote']['session']['selectModel']>[0]['sessionId']

/** A mounted composer model seat: its Session and whether it is on screen now. */
interface SeatRegistration {
  readonly sessionId: SeatSessionId
  readonly visible: () => boolean
}

/** The desktop's native GGUF file dialog: the chosen absolute path, or `null` when cancelled. */
export interface ModelFilePicker {
  pick(): Promise<string | null>
}

/** Owner of the page store; one per plugin activation. */
export class LocalModelsController {
  /** Store bound to the section's `models` hook. */
  readonly store: SnapshotStore<LocalModelsPageState> = createSnapshotStore<LocalModelsPageState>({
    unavailable: false, copyIntoCollection: true,
  })
  private sequence = 0
  private disposed = false
  private readonly seats = new Set<SeatRegistration>()
  /** Settles the request that opened the load dialog with the loaded model id, or `undefined` when cancelled. */
  private settleDialog: ((modelId: string | undefined) => void) | undefined

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
   * Apply a state notice from the Host: load progress, generation progress, or a finished generation's statistics.
   * @param state - the Host's new engine state.
   */
  applyState(state: LocalModelsState): void {
    if (this.disposed) return
    this.store.update((draft) => { if (draft.snapshot !== undefined) draft.snapshot = { ...draft.snapshot, state } })
  }

  /**
   * Load a model and make it the default for new conversations. A model without saved settings, or one set to ask on
   * every load, opens the load options dialog on `owner` first.
   * @param modelId - catalog id.
   * @param owner - surface that shows the dialog.
   * @returns the loaded model id, or `undefined` when cancelled or failed.
   */
  async load(modelId: string, owner = 'section'): Promise<string | undefined> {
    const model = this.store.getSnapshot().snapshot?.models.find(m => m.id === modelId)
    if (model?.settings !== undefined && !model.settings.askOnLoad) {
      const result = await this.act(modelId, () => this.ctx.remote.localModels.load(modelId))
      return result === undefined ? undefined : modelId
    }
    this.settleDialog?.(undefined)
    this.store.update((draft) => { draft.dialog = { modelId, owner }; delete draft.error })
    return new Promise((resolve) => { this.settleDialog = resolve })
  }

  /**
   * Load the dialog's model with the chosen settings.
   * @param settings - settings for this load.
   * @param remember - save them for later loads of this model file.
   * @returns resolves once the Host answered.
   */
  async confirmLoad(settings: LocalModelSettings, remember: boolean): Promise<void> {
    const dialog = this.store.getSnapshot().dialog
    if (dialog === undefined) return
    const settle = this.settleDialog
    this.settleDialog = undefined
    this.store.update((draft) => { delete draft.dialog })
    const result = await this.act(dialog.modelId, () => this.ctx.remote.localModels.load(dialog.modelId, { settings, remember }))
    settle?.(result === undefined ? undefined : dialog.modelId)
  }

  /** Close the load dialog without loading. */
  cancelLoad(): void {
    const settle = this.settleDialog
    this.settleDialog = undefined
    this.store.update((draft) => { delete draft.dialog })
    settle?.(undefined)
  }

  /**
   * Save settings for a model without loading it.
   * @param modelId - catalog id.
   * @param settings - settings to save.
   * @returns resolves once the Host answered.
   */
  async saveSettings(modelId: string, settings: LocalModelSettings): Promise<void> {
    await this.act('settings', () => this.ctx.remote.localModels.saveSettings(modelId, settings))
  }

  /**
   * Ask the picker for a GGUF file and open it: copied into the active collection, or in place when copying is off.
   * It becomes the default for new conversations.
   * @param picker - the desktop file dialog.
   * @param owner - surface that shows the load options dialog.
   * @returns the loaded model id, or `undefined` when cancelled or failed (the failure is on the store).
   */
  async loadFile(picker: ModelFilePicker, owner = 'section'): Promise<string | undefined> {
    if (this.store.getSnapshot().pending !== undefined) return undefined
    let path: string | null
    try { path = await picker.pick() }
    catch (error) {
      this.store.update((draft) => { draft.error = (error as Error).message })
      return undefined
    }
    if (path === null || this.disposed) return undefined
    return this.openPath(path, owner)
  }

  /**
   * Open a GGUF file path, such as one dropped on the window, the way {@link loadFile} opens a chosen one.
   * @param path - absolute `.gguf` path.
   * @param owner - surface that shows the load options dialog.
   * @returns the loaded model id, or `undefined` when cancelled, failed, or another action is in flight.
   */
  async openPath(path: string, owner = 'section'): Promise<string | undefined> {
    if (this.store.getSnapshot().pending !== undefined) return undefined
    const copy = this.store.getSnapshot().copyIntoCollection && this.store.getSnapshot().snapshot?.collections !== undefined
    const added = await this.act('file', () => this.ctx.remote.localModels.addFile(path, copy))
    return added?.ok === true ? this.load(added.modelId, owner) : undefined
  }

  /**
   * Choose whether opened files are copied into the active collection.
   * @param copy - true copies; false loads files in place.
   */
  setCopyIntoCollection(copy: boolean): void {
    this.store.update((draft) => { draft.copyIntoCollection = copy })
  }

  /**
   * Make a collection active.
   * @param id - collection id; `''` is the models folder itself.
   * @returns resolves once the Host answered.
   */
  async setActiveCollection(id: string): Promise<void> {
    await this.act('collection', () => this.ctx.remote.localModels.setActiveCollection(id))
  }

  /**
   * Create a collection and make it active.
   * @param name - folder name.
   * @returns whether the collection was created.
   */
  async createCollection(name: string): Promise<boolean> {
    return (await this.act('collection', () => this.ctx.remote.localModels.createCollection(name))) !== undefined
  }

  /**
   * Track a mounted composer seat, so a dropped file switches the conversation on screen.
   * @param seat - the seat's Session and visibility check.
   * @returns the disposer.
   */
  registerSeat(seat: SeatRegistration): () => void {
    this.seats.add(seat)
    return () => { this.seats.delete(seat) }
  }

  /**
   * The Session whose composer is on screen.
   * @returns its id, or `undefined` when no seat is visible.
   */
  visibleSession(): SeatSessionId | undefined {
    for (const seat of this.seats) if (seat.visible()) return seat.sessionId
    return undefined
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
    this.settleDialog?.(undefined)
    this.settleDialog = undefined
  }
}
