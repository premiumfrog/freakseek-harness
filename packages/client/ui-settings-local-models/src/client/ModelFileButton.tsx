/**
 * The composer's model seat in the offline desktop: one button that opens the native file dialog, loads the chosen
 * GGUF file, and switches the conversation to it.
 */

import { IconDataOutlineRegular, StateDot } from '@deepseek-ai/dsh-client-ui-primitives'
import type { HostObservable, InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { LocalModelsPageState } from './controller.ts'
import css from './ModelFileButton.module.css'

/** Registration-side face of the seat. */
export type ModelFileButtonInjected = {
  hooks: {
    /** Page state store shared with the settings section. */
    models: HostObservable<LocalModelsPageState>
  }
  /** Open the file dialog, load the chosen file, and select it for this conversation. */
  pick: () => void
}

/** Props the renderer binds for the seat. */
export type ModelFileButtonProps =
  PropsRuntime<'conversation.input.model'>
  & PropsLocale<'settings.localModels'>
  & InjectFace<ModelFileButtonInjected>

/** Render the seat. */
export function ModelFileButton({ t, locked, useModels, pick }: ModelFileButtonProps) {
  const page = useModels(value => value)
  const state = page.snapshot?.state
  const loadedId = state?.kind === 'loaded' ? state.modelId : undefined
  const loaded = loadedId === undefined ? undefined : page.snapshot?.models.find(model => model.id === loadedId)
  const busy = page.pending === 'file' || state?.kind === 'loading'
  const label = busy ? t('picker.loading') : loaded?.name ?? t('picker.choose')
  const title = page.error !== undefined ? t('picker.failed', { reason: page.error }) : loaded === undefined ? t('picker.choose') : t('picker.loaded', { model: loaded.name })
  return (
    <button
      type="button"
      className={css.trigger}
      title={title}
      aria-label={title}
      aria-busy={busy}
      data-failed={page.error !== undefined ? 'true' : undefined}
      disabled={locked || page.pending !== undefined}
      onClick={pick}
    >
      <IconDataOutlineRegular className={css.icon} size={16} />
      <span className={css.label}>{label}</span>
      {busy && <StateDot state="ongoing" />}
    </button>
  )
}
