/**
 * The Local models settings section, browser half: lists GGUF files the Host
 * found, loads one into the local engine, and releases it. Loading a model
 * also makes it the default for new conversations.
 */

// Type-only: pulls the locale plugin's Context merge (ctx.locale).
import type {} from '@deepseek-ai/dsh-client-locale/client'
// Type-only: the settings shell's SlotMap merge (the 'settings.section' entry).
import type {} from '@deepseek-ai/dsh-client-ui-settings/client'
import type {} from '@deepseek-ai/dsh-client-ui-renderer/client'
// Type-only: the ctx.remote Context merge and the forwarded-event key face.
import type {} from '@deepseek-ai/dsh-api-remotes/client'
import type { Context as ClientContext } from '@deepseek-ai/cordis'
// Type-only: the conversation SlotMap merge (the composer's 'conversation.input.model' seat).
import type {} from '@deepseek-ai/dsh-client-ui-conversation/client'
import { LocalModelsController } from './controller.ts'
import type { ModelFilePicker, SeatSessionId } from './controller.ts'
import { ModelFileButton } from './ModelFileButton.tsx'
import type { ModelFileButtonInjected } from './ModelFileButton.tsx'
import { LocalModelsSection } from './LocalModelsSection.tsx'
import type { LocalModelsSectionInjected } from './LocalModelsSection.tsx'
import { de, en, es, fr, ja, pt, ru, zh } from './locales.ts'

export type { LocalModelsSectionInjected, LocalModelsSectionProps } from './LocalModelsSection.tsx'
export type { LocalModelsPageState, ModelFilePicker } from './controller.ts'
export type { ModelFileButtonInjected, ModelFileButtonProps } from './ModelFileButton.tsx'
export type { LocalModelsLocaleKey } from './locales.ts'

/** Dictionary namespace owned by this plugin. */
export const NS = 'settings.localModels'

/** Required services (cordis fiber inject). */
export const inject = ['slots', 'locale', 'remote', 'remote.localModels']

/**
 * Register the Local models section.
 * @param ctx - the browser plugin context.
 */
export function apply(ctx: ClientContext): void {
  const t = ctx.locale.bind(NS)
  ctx.effect(() => ctx.locale.register(NS, { zh, en, ja, es, fr, de, ru, pt }), 'ui-settings-local-models: dictionaries')
  const controller = new LocalModelsController(ctx)
  ctx.effect(() => () => { controller.dispose() }, 'ui-settings-local-models: controller')
  // Catalog and engine-state changes arrive as `llm/models-updated` for the `local` route.
  ctx.effect(() => {
    const disposers = [
      ctx.remote.$on('llm/models-updated', (provider) => { if (provider === 'local') controller.refresh() }),
      ctx.on('connection/reset', () => { controller.refresh() }),
    ]
    controller.refresh()
    return () => { for (const dispose of disposers) dispose() }
  }, 'ui-settings-local-models: invalidations')
  // The desktop preload exposes the native GGUF dialog; elsewhere the file flows are absent.
  const picker = (globalThis as typeof globalThis & { __DSH_GGUF_PICKER__?: ModelFilePicker }).__DSH_GGUF_PICKER__
  const injected = (): LocalModelsSectionInjected => ({
    hooks: { models: controller.store },
    refresh: () => { controller.refresh() },
    load: (modelId) => { void controller.load(modelId) },
    ...picker === undefined ? {} : { loadFile: () => { void controller.loadFile(picker) } },
    unload: () => { void controller.unload() },
    setCollection: (id) => { void controller.setActiveCollection(id) },
    createCollection: name => controller.createCollection(name),
    setCopy: (copy) => { controller.setCopyIntoCollection(copy) },
  })
  if (picker !== undefined) {
    // The composer's model seat opens the file dialog, then switches its conversation to the loaded model.
    ctx.inject(['remote.session'], (scope: ClientContext) => {
      const select = async (sessionId: SeatSessionId, modelId: string | undefined) => {
        if (modelId === undefined) return
        const selected = await scope.remote.session.selectModel({ sessionId, provider: 'local', model: modelId })
        if (!selected.ok) controller.fail(selected.error.message)
      }
      scope.slots.inject('conversation.input.model', () => scope.slots.register({
        name: 'conversation.input.model',
        locale: NS,
        inject: (sessionId): ModelFileButtonInjected => ({
          hooks: { models: controller.store },
          pick: () => { void controller.loadFile(picker).then(modelId => select(sessionId, modelId)) },
          setCollection: (id) => { void controller.setActiveCollection(id) },
          createCollection: name => controller.createCollection(name),
          setCopy: (copy) => { controller.setCopyIntoCollection(copy) },
          register: visible => controller.registerSeat({ sessionId, visible }),
        }),
      }, ModelFileButton))
      // The FreakSeek shell announces GGUF files dropped on the window; the first one opens like a chosen file.
      scope.effect(() => {
        // Only a window can receive the shell's event; other hosts of this bundle have none.
        if (typeof globalThis.addEventListener !== 'function') return () => {}
        const onDrop = (event: Event) => {
          const detail = (event as CustomEvent<{ paths?: unknown }>).detail
          const paths = Array.isArray(detail.paths) ? detail.paths : []
          const path = paths.find((candidate): candidate is string => typeof candidate === 'string' && /\.gguf$/iu.test(candidate))
          if (path === undefined) return
          const sessionId = controller.visibleSession()
          void controller.openPath(path).then(modelId => sessionId === undefined ? undefined : select(sessionId, modelId))
        }
        globalThis.addEventListener('freakseek:gguf-dropped', onDrop)
        return () => { globalThis.removeEventListener('freakseek:gguf-dropped', onDrop) }
      }, 'ui-settings-local-models: dropped GGUF files')
    })
  }
  ctx.slots.inject('settings.section', () => ctx.slots.register({
    name: 'settings.section',
    id: 'local-models',
    order: 5,
    label: () => t('nav'),
    locale: NS,
    inject: injected,
  }, LocalModelsSection))
}
