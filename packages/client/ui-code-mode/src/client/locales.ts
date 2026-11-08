/** Locale bundles for the code mode header toggle. */
import type {} from '@deepseek-ai/dsh-client-ui-slots'

declare module '@deepseek-ai/dsh-client-ui-slots' {
  interface LocaleNamespaceMap {
    codeMode: CodeModeLocaleKey
  }
}

/** Copy namespace. */
export const NS = 'codeMode'

/** English copy. */
export const en = {
  label: 'Code',
  enter: 'Code mode: files and a terminal beside the conversation',
  leave: 'Leave code mode',
  shortcut: 'Toggle code mode',
  noSession: 'Open a conversation to use code mode.',
} satisfies Record<string, string>

/** Locale keys the toggle renders. */
export type CodeModeLocaleKey = keyof typeof en

/** Simplified Chinese copy. */
export const zh = {
  label: '代码',
  enter: '代码模式：在对话旁显示文件和终端',
  leave: '退出代码模式',
  shortcut: '切换代码模式',
  noSession: '打开一个对话以使用代码模式。',
} satisfies Record<CodeModeLocaleKey, string>

/** Japanese copy. */
export const ja = {
  label: 'コード',
  enter: 'コードモード：会話の横にファイルとターミナルを表示',
  leave: 'コードモードを終了',
  shortcut: 'コードモードの切り替え',
  noSession: 'コードモードを使うには会話を開いてください。',
} satisfies Record<CodeModeLocaleKey, string>

/** Spanish copy. */
export const es = {
  label: 'Código',
  enter: 'Modo código: archivos y un terminal junto a la conversación',
  leave: 'Salir del modo código',
  shortcut: 'Alternar el modo código',
  noSession: 'Abre una conversación para usar el modo código.',
} satisfies Record<CodeModeLocaleKey, string>

/** French copy. */
export const fr = {
  label: 'Code',
  enter: 'Mode code : fichiers et terminal à côté de la conversation',
  leave: 'Quitter le mode code',
  shortcut: 'Basculer le mode code',
  noSession: 'Ouvrez une conversation pour utiliser le mode code.',
} satisfies Record<CodeModeLocaleKey, string>

/** German copy. */
export const de = {
  label: 'Code',
  enter: 'Codemodus: Dateien und ein Terminal neben der Unterhaltung',
  leave: 'Codemodus verlassen',
  shortcut: 'Codemodus umschalten',
  noSession: 'Öffnen Sie eine Unterhaltung, um den Codemodus zu verwenden.',
} satisfies Record<CodeModeLocaleKey, string>

/** Russian copy. */
export const ru = {
  label: 'Код',
  enter: 'Режим кода: файлы и терминал рядом с беседой',
  leave: 'Выйти из режима кода',
  shortcut: 'Переключить режим кода',
  noSession: 'Откройте беседу, чтобы использовать режим кода.',
} satisfies Record<CodeModeLocaleKey, string>

/** Portuguese copy. */
export const pt = {
  label: 'Código',
  enter: 'Modo código: arquivos e um terminal ao lado da conversa',
  leave: 'Sair do modo código',
  shortcut: 'Alternar o modo código',
  noSession: 'Abra uma conversa para usar o modo código.',
} satisfies Record<CodeModeLocaleKey, string>
