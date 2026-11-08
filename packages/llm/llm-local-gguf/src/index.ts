/**
 * Offline local-model provider: serves the `local` route from GGUF files on
 * disk through the `dsh-gguf-engine` child process, which opens no sockets.
 *
 * @module dsh-llm-local-gguf
 */

import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/cordis-plugin-loader'
import { LocalGgufAdapter, LOCAL_PROVIDER } from './adapter.ts'
import { Config, placementOf, resolveEnginePath, resolveImportedModelsFile, resolveModelDirectories, samplingOf } from './config.ts'
import { LocalModelRuntime } from './runtime.ts'
import { LocalModelsService } from './service.ts'

export { Config, ENGINE_ENV, engineFileName, resolveEnginePath, resolveImportedModelsFile, resolveModelDirectories } from './config.ts'
export { LocalGgufAdapter, LOCAL_PROVIDER } from './adapter.ts'
export { EngineError, EngineProcess, ENGINE_PROTOCOL_VERSION } from './engine-process.ts'
export { OutputParser } from './output-parser.ts'
export { engineMessages, engineTools, IMAGE_PLACEHOLDER } from './request.ts'
export { LocalModelRuntime } from './runtime.ts'
export { LocalModelsService } from './service.ts'
export type * from './types.ts'

export const name = 'llm-local-gguf'
export const inject = ['llm']

/** `process.resourcesPath` inside Electron; absent under plain Node. */
function electronResourcesPath(): string | undefined {
  const value = (process as NodeJS.Process & { resourcesPath?: unknown }).resourcesPath
  return typeof value === 'string' ? value : undefined
}

/**
 * Register the `local` provider route and the `localModels` Remote.
 * @param ctx - Host context.
 * @param config - validated plugin configuration.
 */
export function apply(ctx: Context, config: Config): void {
  const env = process.env
  const engine = resolveEnginePath(config.enginePath, env, electronResourcesPath())
  const runtime = new LocalModelRuntime({
    engine: () => engine,
    threads: () => (config.threads === 0 ? undefined : config.threads),
    directories: () => resolveModelDirectories(config.modelDirectories.get(), env),
    contextLength: () => config.contextLength.get(),
    cacheSlots: () => config.cacheSlots,
    placement: () => placementOf(config),
    importsFile: () => resolveImportedModelsFile(config.importedModelsFile, env),
    shutdownGraceMillis: () => config.shutdownGraceMillis,
  }, () => { ctx.emit('llm/models-updated', LOCAL_PROVIDER) })
  ctx.effect(() => () => runtime.dispose(), 'llm-local-gguf: engine process')
  ctx.llm.registerAdapter([LOCAL_PROVIDER], new LocalGgufAdapter(runtime, {
    sampling: () => samplingOf(config),
    maxOutputTokens: () => config.maxOutputTokens.get(),
  }))
  ctx.llm.registerConfigurableProviders([{
    provider: LOCAL_PROVIDER,
    displayName: 'Local models',
    settingsNs: ctx.fiber.entry?.options.id ?? name,
    settingsPath: [],
    ...'reason' in engine ? { error: engine.reason } : {},
  }])
  ctx.plugin(LocalModelsService, runtime)
  ctx.on('loader/volatile-update', () => { void runtime.refresh().catch((_reportedOnNextSnapshot: unknown) => []) })
}
