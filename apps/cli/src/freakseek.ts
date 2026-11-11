/**
 * The `freakseek` command: FreakSeek's terminal surface over the `cli` profile.
 * `freakseek "<task>"` runs one task on a local model; the `chat` subcommand is reserved.
 * @module @deepseek-ai/dsh/freakseek
 */

/** Profile every FreakSeek terminal task boots. */
export const FREAKSEEK_PROFILE = 'cli'

/** Subcommands the command reserves for later releases. */
const RESERVED_SUBCOMMANDS = new Set(['chat'])

/** What one `freakseek` invocation asks for. */
export type FreakSeekCommand =
  | { readonly kind: 'help' }
  | { readonly kind: 'version' }
  | { readonly kind: 'error'; readonly message: string }
  /** Launcher arguments for the `dsh` CLI, starting with the profile selection. */
  | { readonly kind: 'run'; readonly launcherArgs: readonly string[] }

/** Help text, printed for `freakseek`, `freakseek -h`, and `freakseek --help`. */
export const FREAKSEEK_HELP = `FreakSeek — run tasks on local GGUF models from the terminal.

Usage:
  freakseek "<task>"                         run one task in the current folder, print the result, and exit
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
  if (RESERVED_SUBCOMMANDS.has(first)) {
    return { kind: 'error', message: `freakseek ${first} is not available yet; run freakseek "<task>"` }
  }
  return { kind: 'run', launcherArgs: ['--profile', FREAKSEEK_PROFILE, ...argv] }
}
