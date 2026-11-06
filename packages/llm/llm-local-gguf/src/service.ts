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
import type { LocalModelsActionResult, LocalModelsSnapshot } from './types.ts'

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
   * @returns the snapshot after loading, or the failure reason.
   */
  @Remote('load')
  async load(modelId: string): Promise<LocalModelsActionResult> {
    try {
      await this.runtime.load(modelId)
      await this.ctx.get('agentDefaultModel')?.saveSelection({ provider: LOCAL_PROVIDER, model: modelId })
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
