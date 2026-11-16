/**
 * Load options for one model: context window, GPU layers, sampling, and reasoning, with a memory estimate.
 * The choices can be saved for the model file, and the dialog can stay on for every later load of it.
 */

import { useState } from 'react'
import { Button, Checkbox, Modal } from '@deepseek-ai/dsh-client-ui-primitives'
import type { LocalEngineInfo, LocalModelEntry, LocalModelSettings } from '@deepseek-ai/dsh-api-remotes/client'
import type { LocalModelsLocaleKey } from './locales.ts'
import css from './LoadOptionsDialog.module.css'

/** Context windows the slider offers, clamped to the model's trained length. */
const CONTEXT_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
const DEFAULT_CONTEXT = 32768
const GIB = 2 ** 30

/** Translate function of this plugin's namespace. */
type Translate = (key: LocalModelsLocaleKey, values?: Record<string, string | number>) => string

/** Bytes as GiB with one decimal. */
function gib(bytes: number): string {
  return (bytes / GIB).toFixed(1)
}

/** Token count in k units, for example `64k`. */
function tokens(count: number): string {
  return count >= 1024 ? `${String(Math.round(count / 1024))}k` : String(count)
}

/**
 * Context windows offered for a model.
 * @param trained - trained context length; unknown offers the default only.
 * @returns ascending window sizes ending at the trained length.
 */
export function contextSteps(trained: number | undefined): number[] {
  if (trained === undefined) return [DEFAULT_CONTEXT]
  const steps = CONTEXT_STEPS.filter(step => step < trained)
  return [...steps, trained]
}

/**
 * Estimated memory of a load: weights, KV cache, and recurrent state, and how much VRAM the GPU offers.
 * @param model - catalog entry.
 * @param contextLength - chosen window.
 * @param engine - engine facts with the CUDA devices.
 * @returns byte counts; `vram` is `undefined` without a GPU.
 */
export function estimate(model: LocalModelEntry, contextLength: number, engine: LocalEngineInfo | undefined) {
  const kv = (model.structure?.kvBytesPerToken ?? 0) * contextLength + (model.structure?.stateBytes ?? 0)
  const device = engine?.accelerator === 'cuda' ? engine.devices.find(d => d.supported) : undefined
  return { weights: model.fileBytes, kv, total: model.fileBytes + kv, vram: device?.totalBytes }
}

/** Props of the dialog. */
export interface LoadOptionsDialogProps {
  t: Translate
  model: LocalModelEntry
  engine: LocalEngineInfo | undefined
  onCancel: () => void
  onLoad: (settings: LocalModelSettings, remember: boolean) => void
}

/** Render the load options dialog. */
export function LoadOptionsDialog({ t, model, engine, onCancel, onLoad }: LoadOptionsDialogProps) {
  const saved = model.settings
  const steps = contextSteps(model.trainedContextLength)
  const initialContext = saved?.contextLength ?? Math.min(DEFAULT_CONTEXT, model.trainedContextLength ?? DEFAULT_CONTEXT)
  const distance = (index: number) => Math.abs((steps[index] ?? 0) - initialContext)
  const nearest = steps.reduce((best, _step, index) => (distance(index) < distance(best) ? index : best), 0)
  const [contextIndex, setContextIndex] = useState(nearest)
  const layers = model.structure?.layers ?? 0
  const [autoLayers, setAutoLayers] = useState(saved?.gpuLayers === undefined)
  const [gpuLayers, setGpuLayers] = useState(saved?.gpuLayers ?? layers)
  const [modelSampling, setModelSampling] = useState(saved?.temperature === undefined)
  const [temperature, setTemperature] = useState(saved?.temperature ?? 0.7)
  const [thinking, setThinking] = useState<'default' | 'on' | 'off'>(saved?.thinking === undefined ? 'default' : saved.thinking ? 'on' : 'off')
  const [remember, setRemember] = useState(true)
  const [askOnLoad, setAskOnLoad] = useState(saved?.askOnLoad ?? true)
  const contextLength = steps[contextIndex] ?? DEFAULT_CONTEXT
  const memory = estimate(model, contextLength, engine)
  const fits = memory.vram === undefined ? undefined : memory.total <= memory.vram * 0.9
  const settings: LocalModelSettings = {
    askOnLoad,
    contextLength,
    ...autoLayers ? {} : { gpuLayers },
    ...modelSampling ? {} : { temperature },
    ...thinking === 'default' ? {} : { thinking: thinking === 'on' },
  }
  const experts = model.structure !== undefined && model.structure.experts > 0
  return (
    <Modal
      open
      title={t('dialog.title', { model: model.name })}
      closeLabel={t('dialog.cancel')}
      onClose={onCancel}
      footer={(
        <>
          <Button variant="outline" onClick={onCancel}>{t('dialog.cancel')}</Button>
          <Button variant="primary" data-modal-autofocus onClick={() => { onLoad(settings, remember) }}>{t('dialog.load')}</Button>
        </>
      )}
    >
      <div className={css.body}>
        <label className={css.field}>
          <span className={css.row}>
            <span>{t('dialog.context')}</span>
            <span className={css.value}>{t('dialog.contextValue', { tokens: tokens(contextLength) })}</span>
          </span>
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            step={1}
            value={contextIndex}
            disabled={steps.length < 2}
            onChange={(event) => { setContextIndex(Number(event.target.value)) }}
          />
          <span className={css.hint}>{t('dialog.contextHint', { kv: gib(memory.kv), trained: tokens(model.trainedContextLength ?? contextLength) })}</span>
        </label>
        {layers > 0 && (
          <div className={css.field}>
            <Checkbox checked={autoLayers} onChange={setAutoLayers} label={experts ? t('dialog.autoLayersExperts') : t('dialog.autoLayers')} />
            {!autoLayers && (
              <label className={css.field}>
                <span className={css.row}>
                  <span>{t('dialog.gpuLayers')}</span>
                  <span className={css.value}>{t('dialog.gpuLayersValue', { count: gpuLayers, total: layers })}</span>
                </span>
                <input type="range" min={0} max={layers} step={1} value={gpuLayers} onChange={(event) => { setGpuLayers(Number(event.target.value)) }} />
              </label>
            )}
          </div>
        )}
        <div className={css.estimate} data-fit={fits === undefined ? undefined : fits ? 'yes' : 'no'}>
          {t('dialog.memory', { weights: gib(memory.weights), kv: gib(memory.kv), total: gib(memory.total) })}
          {memory.vram !== undefined && <> · {fits === true ? t('dialog.fits', { vram: gib(memory.vram) }) : t('dialog.spills', { vram: gib(memory.vram) })}</>}
        </div>
        <div className={css.field}>
          <Checkbox checked={modelSampling} onChange={setModelSampling} label={t('dialog.modelSampling')} />
          {!modelSampling && (
            <label className={css.field}>
              <span className={css.row}>
                <span>{t('dialog.temperature')}</span>
                <span className={css.value}>{temperature.toFixed(2)}</span>
              </span>
              <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(event) => { setTemperature(Number(event.target.value)) }} />
            </label>
          )}
        </div>
        {model.reasoning && (
          <label className={css.row}>
            <span>{t('dialog.thinking')}</span>
            <select className={css.select} value={thinking} onChange={(event) => { setThinking(event.target.value as 'default' | 'on' | 'off') }}>
              <option value="default">{t('dialog.thinkingDefault')}</option>
              <option value="on">{t('dialog.thinkingOn')}</option>
              <option value="off">{t('dialog.thinkingOff')}</option>
            </select>
          </label>
        )}
        <div className={css.separator} />
        <Checkbox checked={remember} onChange={setRemember} label={t('dialog.remember')} />
        <Checkbox checked={askOnLoad} onChange={setAskOnLoad} label={t('dialog.askOnLoad')} disabled={!remember} />
      </div>
    </Modal>
  )
}
