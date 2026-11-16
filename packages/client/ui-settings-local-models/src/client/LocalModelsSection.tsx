/** Local models settings section: the GGUF catalog with load and unload actions. */

import { useState } from 'react'
import { Button, StateDot, Tag } from '@deepseek-ai/dsh-client-ui-primitives'
import type { HostObservable, InjectFace, PropsLocale, PropsRuntime } from '@deepseek-ai/dsh-client-ui-slots'
import type { LocalModelEntry, LocalModelSettings } from '@deepseek-ai/dsh-api-remotes/client'
import type { LocalModelsPageState } from './controller.ts'
import type { LocalModelsLocaleKey } from './locales.ts'
import css from './LocalModelsSection.module.css'
import { estimate, LoadOptionsDialog } from './LoadOptionsDialog.tsx'
import { liveStats } from './ModelFileButton.tsx'

/** Registration-side face of the section. */
export type LocalModelsSectionInjected = {
  hooks: {
    /** Page state store. */
    models: HostObservable<LocalModelsPageState>
  }
  refresh: () => void
  load: (modelId: string) => void
  /** Load the dialog's model with the chosen settings, saving them when `remember` is set. */
  confirmLoad: (settings: LocalModelSettings, remember: boolean) => void
  /** Close the load options dialog without loading. */
  cancelLoad: () => void
  /** Open the desktop file dialog and load the chosen GGUF file; absent outside the desktop app. */
  loadFile?: () => void
  unload: () => void
  /** Make a collection active; `''` is the models folder itself. */
  setCollection: (id: string) => void
  /** Create a collection and make it active; resolves to whether it was created. */
  createCollection: (name: string) => Promise<boolean>
  /** Choose whether chosen files are copied into the active collection. */
  setCopy: (copy: boolean) => void
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

/** Bytes as gibibytes with one decimal, the unit GPU tools report memory in. */
function gigabytes(bytes: number): string {
  return (bytes / 2 ** 30).toFixed(1)
}

/** Render the section. */
export function LocalModelsSection({
  t, useModels, refresh, load, confirmLoad, cancelLoad, loadFile, unload, setCollection, createCollection, setCopy,
}: LocalModelsSectionProps) {
  const page = useModels(value => value)
  const [name, setName] = useState('')
  const collections = page.snapshot?.collections
  const engine = page.snapshot?.engine
  const collectionName = (id: string) => id === '' ? t('collection.default') : id
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
            {model.structure?.hasMtp === true && <Tag tone="neutral">{t('card.mtp')}</Tag>}
          </div>
          {model.structure !== undefined && (
            <div className={css.modelFacts}>
              <span>{t('card.layers', { layers: model.structure.layers, attention: model.structure.attentionLayers })}</span>
              {model.structure.experts > 0 && <span>{t('card.experts', { total: model.structure.experts, used: model.structure.expertsUsed })}</span>}
              {model.trainedContextLength !== undefined && <span>{t('card.context', { tokens: Math.round(model.trainedContextLength / 1024) })}</span>}
              <span>{t('card.kv', { size: gigabytes(estimate(model, 32768, engine).kv) })}</span>
              {model.structure.weightTypes.length > 0 && (
                <span>{model.structure.weightTypes.slice(0, 3).map(([type, bytes]) => `${type} ${String(Math.round(bytes / model.fileBytes * 100))}%`).join(' · ')}</span>
              )}
            </div>
          )}
          {model.settings !== undefined && (
            <div className={css.modelFacts}>
              <span>{model.settings.contextLength === undefined
                ? t('card.savedDefault')
                : t('card.saved', { tokens: Math.round(model.settings.contextLength / 1024) })}</span>
              {model.settings.askOnLoad && <span>{t('card.asks')}</span>}
            </div>
          )}
          {loading && state?.kind === 'loading' && liveStats(state, t) !== undefined && <div className={css.modelStatus}>{liveStats(state, t)}</div>}
          {loaded && state?.kind === 'loaded' && <div className={css.modelStatus}>{state.device === 'cpu' ? t('inUse', { context: state.contextLength }) : t('inUseGpu', { context: state.contextLength, device: state.device, layers: state.gpuLayers })}</div>}
          {loaded && state?.kind === 'loaded' && state.stats.warning !== undefined && <div className={css.modelProblem}>{t('statWarning', { reason: state.stats.warning })}</div>}
          {loaded && state?.kind === 'loaded' && (
            <div className={css.modelFacts}>
              {state.stats.experts > 0 && <span>{t('statExperts', { gpu: state.stats.gpuExperts, total: state.stats.experts })}</span>}
              {state.stats.vramTotalBytes !== null && state.stats.vramFreeBytes !== null && (
                <span>{t('statVram', { used: gigabytes(state.stats.vramTotalBytes - state.stats.vramFreeBytes), total: gigabytes(state.stats.vramTotalBytes) })}</span>
              )}
              <span>{t('statLoad', { seconds: (state.stats.loadMs / 1000).toFixed(1) })}</span>
              {state.stats.lastRun !== undefined && (
                <span>{t('statPrompt', { tokens: state.stats.lastRun.promptTokens, cached: state.stats.lastRun.cachedTokens, rate: state.stats.lastRun.promptTokensPerSecond })}</span>
              )}
              {state.stats.lastRun !== undefined && (
                <span>{t('statSpeed', { rate: state.stats.lastRun.tokensPerSecond, tokens: state.stats.lastRun.completionTokens })}</span>
              )}
            </div>
          )}
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
  const dialogModel = page.dialog?.owner === 'section' ? snapshot?.models.find(model => model.id === page.dialog?.modelId) : undefined
  return (
    <div className={css.section}>
      {dialogModel !== undefined && (
        <LoadOptionsDialog t={t} model={dialogModel} engine={engine} onCancel={cancelLoad} onLoad={confirmLoad} />
      )}
      <div className={css.header}>
        <h2 className={css.heading}>{t('title')}</h2>
        <div className={css.headerActions}>
          {loadFile !== undefined && (
            <Button variant="primary" size="sm" disabled={busy} onClick={loadFile}>
              {page.pending === 'file' ? t('picker.loading') : t('loadFile')}
            </Button>
          )}
          <Button variant="outline" size="sm" disabled={busy} onClick={refresh}>{t('refresh')}</Button>
        </div>
      </div>
      <p className={css.intro}>{t('intro')}</p>
      {engine !== undefined && (
        <p className={css.engine}>
          {engine.accelerator === 'cuda'
            ? t('engine.cuda', { gpus: engine.devices.filter(d => d.supported).map(d => d.name).join(', ') })
            : t('engine.cpu')}
          {engine.fallbackReason !== undefined && <span className={css.engineReason}>{t('engine.fallback', { reason: engine.fallbackReason })}</span>}
        </p>
      )}
      {collections !== undefined && (
        <div className={css.collections}>
          <div className={css.foldersTitle}>{t('collections.title')}</div>
          <p className={css.foldersHint}>{t('collections.hint', { root: collections.root })}</p>
          <ul className={css.collectionList}>
            {collections.items.map(item => (
              <li key={item.id} className={css.collection} data-active={item.id === collections.active ? 'true' : undefined}>
                <span className={css.collectionName}>{collectionName(item.id)}</span>
                <span className={css.collectionCount}>{item.modelCount === 1 ? t('collection.countOne') : t('collection.countMany', { count: item.modelCount })}</span>
                {item.id === collections.active
                  ? <Tag tone="neutral">{t('collections.active')}</Tag>
                  : <Button variant="outline" size="sm" disabled={busy} onClick={() => { setCollection(item.id) }}>{t('collections.use')}</Button>}
              </li>
            ))}
          </ul>
          <form className={css.collectionCreate} onSubmit={(event) => {
            event.preventDefault()
            void createCollection(name).then((created) => { if (created) setName('') })
          }}>
            <input className={css.collectionInput} value={name} maxLength={64} placeholder={t('collection.namePlaceholder')}
              aria-label={t('collection.namePlaceholder')} onChange={(event) => { setName(event.target.value) }} />
            <Button type="submit" variant="outline" size="sm" disabled={busy || name.trim() === ''}>{t('collection.create')}</Button>
          </form>
          {loadFile !== undefined && (
            <label className={css.copyToggle}>
              <input type="checkbox" checked={page.copyIntoCollection} onChange={(event) => { setCopy(event.target.checked) }} />
              <span>{t('copy.toggle')}</span>
            </label>
          )}
        </div>
      )}
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
