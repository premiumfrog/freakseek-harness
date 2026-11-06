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
import { LocalModelsController } from './controller.ts'
import { LocalModelsSection } from './LocalModelsSection.tsx'
import type { LocalModelsSectionInjected } from './LocalModelsSection.tsx'
import { de, en, es, fr, ja, pt, ru, zh } from './locales.ts'

export type { LocalModelsSectionInjected, LocalModelsSectionProps } from './LocalModelsSection.tsx'
export type { LocalModelsPageState } from './controller.ts'
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
  const injected = (): LocalModelsSectionInjected => ({
    hooks: { models: controller.store },
    refresh: () => { controller.refresh() },
    load: (modelId) => { void controller.load(modelId) },
    unload: () => { void controller.unload() },
  })
  ctx.slots.inject('settings.section', () => ctx.slots.register({
    name: 'settings.section',
    id: 'local-models',
    order: 5,
    label: () => t('nav'),
    locale: NS,
    inject: injected,
  }, LocalModelsSection))
}
