/**
 * The `localModels` Remote: catalog, load, and unload operations for
 * configuration surfaces such as the desktop Local models page.
 *
 * @module dsh-llm-local-gguf/service
 */

import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-agent-default-model'
import { Remote, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'
import { LOCAL_PROVIDER } from './adapter.ts'
import type { LocalModelRuntime } from './runtime.ts'
import type { LocalModelLoadOptions, LocalModelSettings, LocalModelsActionResult, LocalModelsFileResult, LocalModelsSnapshot } from './types.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    localModels: LocalModelsService
  }
}

/** Remote service over one plugin instance's {@link LocalModelRuntime}. */
export class LocalModelsService extends TypertRemoteService {
  /**
   * @param ctx - Host context.
   * @param runtime - the owning plugin's runtime.
   */
  constructor(ctx: Context, private readonly runtime: LocalModelRuntime) {
    super(ctx, 'localModels')
  }

  /**
   * Rescan model directories and report the catalog with the engine state.
   * @returns the current snapshot.
   */
  @Remote('snapshot')
  async snapshot(): Promise<LocalModelsSnapshot> {
    // Directory failures are reported inside the snapshot; engine failures surface per model.
    await this.runtime.refresh().catch((_reportedInSnapshot: unknown) => [])
    return this.runtime.snapshot()
  }

  /**
   * Load one catalog model into memory, replacing the loaded one, and make it
   * the default model of new conversations when the deployment has one.
   * @param modelId - catalog model id.
   * @param options - settings for this load, saved for later loads when `remember` is set; omitted uses saved settings.
   * @returns the snapshot after loading, or the failure reason.
   */
  @Remote('load')
  async load(modelId: string, options?: LocalModelLoadOptions): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.load(modelId, options)
      await this.ctx.get('agentDefaultModel')?.saveSelection({ provider: LOCAL_PROVIDER, model: modelId })
      return { ok: true, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Add a GGUF file from anywhere on disk to the catalog, remember it across restarts, load it, and make it the
   * default model of new conversations when the deployment has one.
   * @param path - absolute path to a `.gguf` file, such as one chosen in the desktop file dialog.
   * @returns the model id and snapshot after loading, or the failure reason.
   */
  @Remote('loadFile')
  async loadFile(path: string): Promise<LocalModelsFileResult> {
    try {
      const modelId = await this.runtime.loadFile(path)
      await this.ctx.get('agentDefaultModel')?.saveSelection({ provider: LOCAL_PROVIDER, model: modelId })
      return { ok: true, modelId, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Add a GGUF file from anywhere on disk to the catalog without loading it, so a configuration surface can ask for
   * load settings first: copied into the active collection, or remembered in place across restarts.
   * @param path - absolute path to a `.gguf` file.
   * @param copy - copy the file into the active collection instead of listing it in place.
   * @returns the model id and snapshot, or the failure reason.
   */
  @Remote('addFile')
  async addFile(path: string, copy: boolean): Promise<LocalModelsFileResult> {
    try {
      const modelId = copy ? await this.runtime.copyFile(path) : await this.runtime.addFile(path)
      return { ok: true, modelId, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Save load and sampling settings for a model file.
   * @param modelId - catalog model id.
   * @param settings - the settings; invalid fields are dropped.
   * @returns the snapshot listing the saved settings, or the failure reason.
   */
  @Remote('saveSettings')
  async saveSettings(modelId: string, settings: LocalModelSettings): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.saveSettings(modelId, settings)
      return { ok: true, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Copy a GGUF file from anywhere on disk into the active collection, load the copy, and make it the default model
   * of new conversations when the deployment has one.
   * @param path - absolute path to a `.gguf` file, such as one chosen in the desktop file dialog or dropped on the window.
   * @returns the copy's model id and the snapshot after loading, or the failure reason.
   */
  @Remote('importFile')
  async importFile(path: string): Promise<LocalModelsFileResult> {
    try {
      const modelId = await this.runtime.importFile(path)
      await this.ctx.get('agentDefaultModel')?.saveSelection({ provider: LOCAL_PROVIDER, model: modelId })
      return { ok: true, modelId, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Create a collection folder under the models root and make it the active collection.
   * @param name - folder name: 1–64 characters, portable on Windows.
   * @returns the snapshot listing the new, empty collection, or the failure reason.
   */
  @Remote('createCollection')
  async createCollection(name: string): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.createCollection(name)
      return { ok: true, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Make a collection active, so the catalog lists its models; a loaded model stays loaded.
   * @param id - collection id from the snapshot; `''` selects the models root.
   * @returns the snapshot after switching, or the failure reason.
   */
  @Remote('setActiveCollection')
  async setActiveCollection(id: string): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.setActiveCollection(id)
      return { ok: true, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }

  /**
   * Release the loaded model's memory.
   * @returns the snapshot after unloading, or the failure reason.
   */
  @Remote('unload')
  async unload(): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.unload()
      return { ok: true, snapshot: this.runtime.snapshot() }
    } catch (error) {
      return { ok: false, reason: (error as Error).message, snapshot: this.runtime.snapshot() }
    }
  }
}
