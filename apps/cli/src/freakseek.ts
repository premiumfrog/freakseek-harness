/**
 * The `freakseek` command: FreakSeek's terminal surface over the `cli` profile.
 * `freakseek "<task>"` runs one task on a local model; `freakseek chat` opens an interactive chat with one.
 * @module @deepseek-ai/dsh/freakseek
 */

/** Profile every FreakSeek terminal task boots. */
export const FREAKSEEK_PROFILE = 'cli'

/** What one `freakseek` invocation asks for. */
export type FreakSeekCommand =
  | { readonly kind: 'help' }
  | { readonly kind: 'version' }
  | { readonly kind: 'error'; readonly message: string }
  /** Interactive chat; `modelId` names a catalog model, omitted lets the chat choose. */
  | { readonly kind: 'chat'; readonly modelId?: string }
  /** Launcher arguments for the `dsh` CLI, starting with the profile selection. */
  | { readonly kind: 'run'; readonly launcherArgs: readonly string[] }

/** Help text, printed for `freakseek`, `freakseek -h`, and `freakseek --help`. */
export const FREAKSEEK_HELP = `FreakSeek — run tasks on local GGUF models from the terminal.

Usage:
  freakseek "<task>"                         run one task in the current folder, print the result, and exit
  freakseek chat [model-id]                  chat with a local model in the terminal (/help lists commands)
  freakseek --session-id <id> "<task>"       continue a Session with another task
  freakseek --json "<task>"                  print the run as JSON lines
  freakseek --permission-mode <preset> "…"   choose a permission preset for this task
  freakseek --add-dir <path> "<task>"        allow writes to another directory
  freakseek --version                        print the version

The model is the only supported .gguf file in the active model folder, or the id in DSH_LOCAL_MODEL.
`

/**
 * Resolve `freakseek` arguments.
 * @param argv - arguments after the command name.
 * @returns the requested action.
 */
export function parseFreakSeekArgs(argv: readonly string[]): FreakSeekCommand {
  const [first] = argv
  if (first === undefined || first === '-h' || first === '--help' || first === 'help') return { kind: 'help' }
  if (first === '-V' || first === '--version') return { kind: 'version' }
  if (first === 'chat') {
    const rest = argv.slice(1)
    if (rest.length > 1 || rest[0]?.startsWith('-') === true) return { kind: 'error', message: 'usage: freakseek chat [model-id]' }
    return rest[0] === undefined ? { kind: 'chat' } : { kind: 'chat', modelId: rest[0] }
  }
  return { kind: 'run', launcherArgs: ['--profile', FREAKSEEK_PROFILE, ...argv] }
}
