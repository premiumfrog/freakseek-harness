/** Window-owned native open dialogs (workspace directories and GGUF model files) for the local Desktop renderer. */

import { dialog, ipcMain, type BrowserWindow, type OpenDialogOptions } from 'electron'
import { DESKTOP_IPC, assertDesktopSender } from './ipc.ts'

/** Dialog options for the GGUF model picker. */
export const GGUF_PICKER_OPTIONS: OpenDialogOptions = { properties: ['openFile'], filters: [{ name: 'GGUF models', extensions: ['gguf'] }] }

/**
 * Install one application-lifetime open-dialog IPC handler that answers the first selected path or `null`.
 * @param channel - IPC channel the app renderer invokes.
 * @param options - Electron open-dialog options.
 * @param getWindow - Current local application window; shell pages and subframes cannot open dialogs.
 * @param purpose - Dialog name for the rejection message.
 */
function installOpenDialog(channel: string, options: OpenDialogOptions, getWindow: () => BrowserWindow | undefined, purpose: string): void {
  const pending = new WeakMap<BrowserWindow, Promise<string | null>>()
  ipcMain.handle(channel, async (event) => {
    const window = getWindow()
    if (window === undefined || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) {
      throw new Error(`dsh desktop: rejected ${purpose} from an unowned renderer`)
    }
    assertDesktopSender(event, ['app'])
    const existing = pending.get(window)
    if (existing !== undefined) return existing
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
    const result = dialog.showOpenDialog(window, options).then(
      ({ canceled, filePaths }) => window.isDestroyed() || canceled ? null : filePaths[0] ?? null,
    ).finally(() => { pending.delete(window) })
    pending.set(window, result)
    return result
  })
}

/**
 * Install the application-lifetime directory picker IPC handler.
 * @param getWindow - Current local application window; shell pages and subframes cannot open dialogs.
 */
export function installDesktopDirectoryPicker(getWindow: () => BrowserWindow | undefined): void {
  installOpenDialog(DESKTOP_IPC.directoryPick, { properties: ['openDirectory', 'createDirectory'] }, getWindow, 'directory picker')
}

/**
 * Install the application-lifetime GGUF model file picker IPC handler.
 * @param getWindow - Current local application window; shell pages and subframes cannot open dialogs.
 */
export function installDesktopGgufPicker(getWindow: () => BrowserWindow | undefined): void {
  installOpenDialog(DESKTOP_IPC.ggufPick, GGUF_PICKER_OPTIONS, getWindow, 'model file picker')
}
