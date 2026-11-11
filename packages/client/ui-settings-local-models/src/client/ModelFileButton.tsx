/**
 * The composer's model seat in the offline desktop: a collection switcher and one button that opens the native file
 * dialog, loads the chosen GGUF file, and switches the conversation to it.
 */

import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import {
  IconCheckOutlineRegular, IconChevronDownOutlineRegular, IconDataOutlineRegular, MenuSurface, StateDot,
} from '@deepseek-ai/dsh-client-ui-primitives'
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
  /** Make a collection active; `''` is the models folder itself. */
  setCollection: (id: string) => void
  /** Create a collection and make it active; resolves to whether it was created. */
  createCollection: (name: string) => Promise<boolean>
  /** Choose whether chosen files are copied into the active collection. */
  setCopy: (copy: boolean) => void
  /** Track this seat while mounted, so a dropped file switches the conversation on screen. */
  register: (visible: () => boolean) => () => void
}

/** Props the renderer binds for the seat. */
export type ModelFileButtonProps =
  PropsRuntime<'conversation.input.model'>
  & PropsLocale<'settings.localModels'>
  & InjectFace<ModelFileButtonInjected>

/** Render the seat. */
export function ModelFileButton({ t, locked, useModels, pick, setCollection, createCollection, setCopy, register }: ModelFileButtonProps) {
  const page = useModels(value => value)
  const root = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  useEffect(() => register(() => root.current !== null && root.current.offsetParent !== null), [register])
  useEffect(() => {
    if (!open) return undefined
    const close = (event: MouseEvent) => {
      if (root.current === null || !(event.target instanceof Node) || root.current.contains(event.target)) return
      setOpen(false)
      setNaming(false)
    }
    document.addEventListener('mousedown', close)
    return () => { document.removeEventListener('mousedown', close) }
  }, [open])
  const state = page.snapshot?.state
  const loadedId = state?.kind === 'loaded' ? state.modelId : undefined
  const loaded = loadedId === undefined ? undefined : page.snapshot?.models.find(model => model.id === loadedId)
  const busy = page.pending === 'file' || state?.kind === 'loading'
  const label = busy ? t('picker.loading') : loaded?.name ?? (loadedId === undefined ? t('picker.choose') : loadedId)
  const title = page.error !== undefined ? t('picker.failed', { reason: page.error }) : loadedId === undefined ? t('picker.choose') : t('picker.loaded', { model: loaded?.name ?? loadedId })
  const collections = page.snapshot?.collections
  const collectionName = (id: string) => id === '' ? t('collection.default') : id
  const submit = () => {
    void createCollection(name).then((created) => { if (created) { setName(''); setNaming(false); setOpen(false) } })
  }
  const onNameKey = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') { event.preventDefault(); submit() }
    if (event.key === 'Escape') { event.preventDefault(); setNaming(false) }
  }
  const onMenuKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape' && !naming) { event.preventDefault(); setOpen(false) }
  }
  return (
    <div ref={root} className={css.seat}>
      {collections !== undefined && (
        <button
          type="button"
          className={css.trigger}
          aria-haspopup="menu"
          aria-expanded={open}
          title={t('collection.label', { name: collectionName(collections.active) })}
          aria-label={t('collection.label', { name: collectionName(collections.active) })}
          disabled={locked || page.pending !== undefined}
          onClick={() => { setOpen(value => !value); setNaming(false) }}
        >
          <span className={css.label}>{collectionName(collections.active)}</span>
          <IconChevronDownOutlineRegular className={css.caret} size={12} aria-hidden />
        </button>
      )}
      {open && collections !== undefined && (
        <MenuSurface compact className={css.menu} role="menu" aria-label={t('collection.menu')} onKeyDown={onMenuKey}>
          {collections.items.map(item => (
            <button
              key={item.id}
              type="button"
              role="menuitemradio"
              aria-checked={item.id === collections.active}
              className={css.item}
              onClick={() => { setCollection(item.id); setOpen(false) }}
            >
              <span className={css.check}>{item.id === collections.active && <IconCheckOutlineRegular size={14} aria-hidden />}</span>
              <span className={css.itemName}>{collectionName(item.id)}</span>
              <span className={css.count}>{item.modelCount === 1 ? t('collection.countOne') : t('collection.countMany', { count: item.modelCount })}</span>
            </button>
          ))}
          <div className={css.separator} role="separator" />
          {naming
            ? (
              <div className={css.naming}>
                <input
                  className={css.input}
                  autoFocus
                  value={name}
                  maxLength={64}
                  placeholder={t('collection.namePlaceholder')}
                  aria-label={t('collection.namePlaceholder')}
                  onChange={(event) => { setName(event.target.value) }}
                  onKeyDown={onNameKey}
                />
                <button type="button" className={css.create} disabled={name.trim() === ''} onClick={submit}>{t('collection.create')}</button>
              </div>
            )
            : <button type="button" role="menuitem" className={css.item} onClick={() => { setNaming(true) }}>{t('collection.new')}</button>}
          <label className={css.toggle}>
            <input type="checkbox" checked={page.copyIntoCollection} onChange={(event) => { setCopy(event.target.checked) }} />
            <span>{t('copy.toggle')}</span>
          </label>
        </MenuSurface>
      )}
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
    </div>
  )
}
