/** Local models settings section: the GGUF catalog with load and unload actions. */

import { Button, StateDot, Tag } from '@deepseek-ai/dsh-client-ui-primitives'
import type { HostObservable, InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { LocalModelEntry } from '@deepseek-ai/dsh-api-remotes/client'
import type { LocalModelsPageState } from './controller.ts'
import type { LocalModelsLocaleKey } from './locales.ts'
import css from './LocalModelsSection.module.css'

/** Registration-side face of the section. */
export type LocalModelsSectionInjected = {
  hooks: {
    /** Page state store. */
    models: HostObservable<LocalModelsPageState>
  }
  refresh: () => void
  load: (modelId: string) => void
  unload: () => void
}

/** Props the renderer binds for the section. */
export type LocalModelsSectionProps =
  PropsRuntime<'settings.section'>
  & PropsLocale<'settings.localModels'>
  & InjectFace<LocalModelsSectionInjected>

/** File size in binary units, as model downloads report it. */
function fileSize(bytes: number, t: LocalModelsSectionProps['t']): string {
  return bytes >= 2 ** 30
    ? t('sizeGiB', { size: (bytes / 2 ** 30).toFixed(1) })
    : t('sizeMiB', { size: Math.max(1, Math.round(bytes / 2 ** 20)) })
}

/** Parameter count in the B/M notation model names use. */
function parameterCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(count < 1e10 ? 1 : 0)}B`
  return `${Math.max(1, Math.round(count / 1e6))}M`
}

/** Render the section. */
export function LocalModelsSection({ t, useModels, refresh, load, unload }: LocalModelsSectionProps) {
  const page = useModels(value => value)
  const snapshot = page.snapshot
  const state = snapshot?.state
  const busy = page.pending !== undefined
  const loadedId = state?.kind === 'loaded' ? state.modelId : undefined
  const row = (model: LocalModelEntry) => {
    const loaded = model.id === loadedId
    const loading = page.pending === model.id || (state?.kind === 'loading' && state.modelId === model.id)
    return (
      <li key={model.id} className={css.model} data-loaded={loaded ? 'true' : undefined}>
        <div className={css.modelMain}>
          <div className={css.modelName}>
            {loaded && <StateDot state="done" />}
            <span>{model.name}</span>
          </div>
          <div className={css.modelFacts}>
            <span>{model.architecture}</span>
            {model.parameterCount > 0 && <span>{t('parameters', { count: parameterCount(model.parameterCount) })}</span>}
            {model.quantization !== undefined && <span>{model.quantization}</span>}
            <span>{fileSize(model.fileBytes, t)}</span>
            {model.tools && <Tag tone="neutral">{t('tools')}</Tag>}
            {model.reasoning && <Tag tone="neutral">{t('reasoning')}</Tag>}
          </div>
          {loaded && state?.kind === 'loaded' && <div className={css.modelStatus}>{t('inUse', { context: state.contextLength })}</div>}
          {!model.supported && <div className={css.modelProblem}>{t('unsupported', { reason: model.unsupportedReason ?? '' })}</div>}
          <div className={css.modelPath} title={model.path}>{t('fileName', { id: model.id })}</div>
        </div>
        <div className={css.modelActions}>
          {loaded ? (
            <Button variant="outline" size="sm" disabled={busy} onClick={unload}>
              {page.pending === 'unload' ? t('unloading') : t('unload')}
            </Button>
          ) : (
            <Button variant="primary" size="sm" disabled={busy || !model.supported} onClick={() => { load(model.id) }}>
              {loading ? t('loading') : t('load')}
            </Button>
          )}
        </div>
      </li>
    )
  }
  return (
    <div className={css.section}>
      <div className={css.header}>
        <h2 className={css.heading}>{t('title')}</h2>
        <Button variant="outline" size="sm" disabled={busy} onClick={refresh}>{t('refresh')}</Button>
      </div>
      <p className={css.intro}>{t('intro')}</p>
      {page.unavailable && <p className={css.problem}>{t('unavailable')}</p>}
      {state?.kind === 'engine-unavailable' && <p className={css.problem}>{t('engineUnavailable', { reason: state.reason })}</p>}
      {state?.kind === 'failed' && <p className={css.problem}>{t('loadFailed', { model: state.modelId, reason: state.reason })}</p>}
      {page.error !== undefined && state?.kind !== 'failed' && <p className={css.problem}>{t('actionFailed', { reason: page.error })}</p>}
      {snapshot !== undefined && (
        <>
          <ul className={css.models}>
            {snapshot.models.map(row)}
          </ul>
          {snapshot.models.length === 0 && <p className={css.empty}>{t('empty')}</p>}
          <div className={css.folders}>
            <div className={css.foldersTitle}>{t('folders')}</div>
            <p className={css.foldersHint}>{t('foldersHint')}</p>
            <ul className={css.folderList}>
              {snapshot.directories.map(directory => <li key={directory} className={css.folder}>{directory}</li>)}
            </ul>
            {snapshot.problems.map(problem => (
              <p key={problem.directory} className={css.problem}>{t('folderProblem', { directory: problem.directory, reason: problem.reason })}</p>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap {
    /** Local models settings section copy. */
    'settings.localModels': LocalModelsLocaleKey
  }
}
