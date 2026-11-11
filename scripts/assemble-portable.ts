/**
 * Assemble the portable FreakSeek folder from built parts.
 *
 * Run after `pnpm run build`. The folder needs no installer: the Tauri shell, a bundled
 * Node.js, the deployed runtime tree, the GGUF engines, an empty models folder, CLI
 * launchers, and README-FIRST.txt. Conversations and settings are written to `data/` inside
 * the folder on first launch.
 *
 * ```sh
 * pnpm run assemble:portable -- --node <node binary> --shell <FreakSeek binary> \
 *   [--engine-cpu <dsh-gguf-engine binary>] [--engine-cuda-dir <dir with the CUDA engine and DLLs>] \
 *   [--out out/FreakSeek] [--platform win32|linux|darwin] [--app <existing deploy dir>]
 * ```
 *
 * @module scripts/assemble-portable
 */

import { spawnSync } from 'node:child_process'
import {
  chmodSync, copyFileSync, cpSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, readlinkSync, realpathSync,
  rmSync, statSync, writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { parseArgs } from 'node:util'

const ROOT = resolve(import.meta.dirname, '..')

/** Directory names a workspace package keeps for development only. */
const DEV_ONLY = new Set(['node_modules', 'tests', 'test', 'coverage', '.cache', '.turbo', '.desktop-build'])

/**
 * Runtime packages the portable folder leaves out. The Office skills need the desktop primary
 * runtime, which the portable folder does not ship, so their LibreOffice payload is never loaded.
 */
const PRUNED_PACKAGES = new Set(['@deepseek-ai/libreoffice-kit-wasm'])

/** Parsed command line. */
interface Options {
  readonly out: string
  readonly node: string
  readonly shell: string
  /** CPU engine; omitted when a later step adds `engine/` (the CI release job builds engines in parallel). */
  readonly engineCpu: string | undefined
  readonly engineCudaDir: string | undefined
  readonly platform: NodeJS.Platform
  readonly app: string | undefined
}

function options(): Options {
  const { values } = parseArgs({
    options: {
      'out': { type: 'string', default: join(ROOT, 'out', 'FreakSeek') },
      'node': { type: 'string' },
      'shell': { type: 'string' },
      'engine-cpu': { type: 'string' },
      'engine-cuda-dir': { type: 'string' },
      'platform': { type: 'string', default: process.platform },
      'app': { type: 'string' },
    },
  })
  const required = (name: 'node' | 'shell'): string => {
    const value = values[name]
    if (value === undefined || value === '') throw new Error(`assemble-portable: --${name} is required`)
    if (!existsSync(value)) throw new Error(`assemble-portable: --${name} ${value} does not exist`)
    return resolve(value)
  }
  const existing = (value: string, name: string): string => {
    if (!existsSync(value)) throw new Error(`assemble-portable: --${name} ${value} does not exist`)
    return resolve(value)
  }
  const platform = values.platform as NodeJS.Platform
  if (!['win32', 'linux', 'darwin'].includes(platform)) throw new Error(`assemble-portable: unsupported --platform ${platform}`)
  const cuda = values['engine-cuda-dir']
  if (cuda !== undefined && !statSync(cuda).isDirectory()) throw new Error(`assemble-portable: --engine-cuda-dir ${cuda} is not a directory`)
  return {
    out: resolve(values.out),
    node: required('node'),
    shell: required('shell'),
    engineCpu: values['engine-cpu'] === undefined ? undefined : existing(values['engine-cpu'], 'engine-cpu'),
    engineCudaDir: cuda === undefined ? undefined : resolve(cuda),
    platform,
    app: values.app === undefined ? undefined : resolve(values.app),
  }
}

/**
 * Deploy the desktop-host runtime with its production dependencies in a flat layout.
 * @returns the deploy directory, which still links some workspace packages.
 */
function deployRuntime(): string {
  const target = mkdtempSync(join(tmpdir(), 'freakseek-deploy-'))
  rmSync(target, { recursive: true, force: true })
  const result = spawnSync('pnpm', [
    '--filter', '@deepseek-ai/dsh-desktop-host', 'deploy', '--prod', '--legacy',
    // A flat tree resolves every peer and survives zipping; patches for dev-only packages stay unused.
    '--config.node-linker=hoisted', '--config.allow-unused-patches=true', target,
  ], { cwd: ROOT, stdio: 'inherit', shell: process.platform === 'win32' })
  if (result.status !== 0) throw new Error(`assemble-portable: pnpm deploy failed with status ${String(result.status)}`)
  return target
}

/**
 * Copy one workspace package's runtime files: everything except development-only folders.
 * @param source - the package directory a deploy symlink points to.
 * @param destination - where the package lives in the portable tree.
 */
function materializePackage(source: string, destination: string): void {
  cpSync(source, destination, {
    recursive: true,
    dereference: true,
    filter: (path) => {
      if (path === source) return true
      const name = basename(path)
      return !DEV_ONLY.has(name) && !name.endsWith('.tsbuildinfo')
    },
  })
}

/**
 * Copy a deploy tree into the portable folder with no symlinks left: `.bin` folders are
 * dropped and linked workspace packages are copied without their development folders.
 * @param source - deploy directory.
 * @param destination - `runtime/app`.
 * @returns the number of linked packages materialized.
 */
function copyRuntimeTree(source: string, destination: string): number {
  let linked = 0
  const walk = (from: string, to: string): void => {
    mkdirSync(to, { recursive: true })
    for (const entry of readdirSync(from, { withFileTypes: true })) {
      const path = join(from, entry.name)
      const target = join(to, entry.name)
      if (entry.name === '.bin' && basename(from) === 'node_modules') continue
      if (PRUNED_PACKAGES.has(`${basename(from)}/${entry.name}`)) continue
      if (entry.isSymbolicLink()) {
        const resolved = realpathSync(resolve(dirname(path), readlinkSync(path)))
        if (statSync(resolved).isDirectory()) {
          materializePackage(resolved, target)
          linked++
        } else {
          copyFileSync(resolved, target)
        }
      } else if (entry.isDirectory()) {
        walk(path, target)
      } else if (entry.isFile()) {
        copyFileSync(path, target)
        if ((lstatSync(path).mode & 0o111) !== 0) chmodSync(target, 0o755)
      }
    }
  }
  walk(source, destination)
  return linked
}

function cliLaunchers(out: string): void {
  writeFileSync(join(out, 'freakseek.cmd'), [
    '@echo off',
    'setlocal',
    'set "FREAKSEEK_ROOT=%~dp0"',
    'set "DSH_HOME=%FREAKSEEK_ROOT%data"',
    'set "DSH_AGENTS_HOME=%FREAKSEEK_ROOT%data\\agents"',
    'set "DSH_LOCAL_MODELS_ROOT=%FREAKSEEK_ROOT%models"',
    'set "DSH_GGUF_ENGINE_DIR=%FREAKSEEK_ROOT%engine"',
    'set "DSH_TELEMETRY_DISABLED=1"',
    '"%FREAKSEEK_ROOT%runtime\\node.exe" --expose-internals "%FREAKSEEK_ROOT%runtime\\app\\node_modules\\@deepseek-ai\\dsh\\lib\\freakseek.js" %*',
    '',
  ].join('\r\n'))
  const sh = join(out, 'freakseek')
  writeFileSync(sh, [
    '#!/bin/sh',
    'root=$(cd "$(dirname "$0")" && pwd)',
    'export DSH_HOME="$root/data" DSH_AGENTS_HOME="$root/data/agents" DSH_LOCAL_MODELS_ROOT="$root/models" DSH_GGUF_ENGINE_DIR="$root/engine" DSH_TELEMETRY_DISABLED=1',
    'exec "$root/runtime/node" --expose-internals "$root/runtime/app/node_modules/@deepseek-ai/dsh/lib/freakseek.js" "$@"',
    '',
  ].join('\n'))
  chmodSync(sh, 0o755)
}

const README = `FreakSeek - Fast Rust Embodied AI Kernel
=========================================

Everything FreakSeek needs is in this folder. Nothing is installed, and nothing
leaves this computer except the web searches and page fetches a model makes.

Start
  Run FreakSeek.exe. Closing the window keeps FreakSeek in the system tray;
  choose Quit from the tray icon to stop it.

Models
  Put .gguf model files in a collection folder inside "models", for example
  models\\coding\\qwen2.5-coder-7b-instruct-q4_k_m.gguf, or use the model button
  in the composer to pick a file. Create and switch collections in the app.

Command line
  freakseek "explain this repository"     one task in the current folder
  freakseek --help                        every option

GPU
  NVIDIA RTX 20-series and newer run models on the GPU automatically with a
  current NVIDIA driver; other PCs use the CPU engine.

Your data
  Conversations, settings, and logs are kept in the "data" folder next to this
  file. Copy the whole FreakSeek folder to move or back up everything.
`

function main(): void {
  const opts = options()
  const exe = opts.platform === 'win32' ? '.exe' : ''
  rmSync(opts.out, { recursive: true, force: true })
  mkdirSync(join(opts.out, 'runtime'), { recursive: true })

  const deployed = opts.app ?? deployRuntime()
  const linked = copyRuntimeTree(deployed, join(opts.out, 'runtime', 'app'))
  if (opts.app === undefined) rmSync(deployed, { recursive: true, force: true })
  if (!existsSync(join(opts.out, 'runtime', 'app', 'lib', 'index.js'))) {
    throw new Error('assemble-portable: the deployed runtime has no lib/index.js; run pnpm run build first')
  }

  copyFileSync(opts.node, join(opts.out, 'runtime', `node${exe}`))
  chmodSync(join(opts.out, 'runtime', `node${exe}`), 0o755)
  copyFileSync(opts.shell, join(opts.out, `FreakSeek${exe}`))
  chmodSync(join(opts.out, `FreakSeek${exe}`), 0o755)

  if (opts.engineCpu !== undefined) {
    mkdirSync(join(opts.out, 'engine', 'cpu'), { recursive: true })
    copyFileSync(opts.engineCpu, join(opts.out, 'engine', 'cpu', `dsh-gguf-engine${exe}`))
    chmodSync(join(opts.out, 'engine', 'cpu', `dsh-gguf-engine${exe}`), 0o755)
  }
  if (opts.engineCudaDir !== undefined) cpSync(opts.engineCudaDir, join(opts.out, 'engine', 'cuda'), { recursive: true })

  mkdirSync(join(opts.out, 'models'), { recursive: true })
  writeFileSync(join(opts.out, 'models', '.keep'), '')
  cliLaunchers(opts.out)
  writeFileSync(join(opts.out, 'README-FIRST.txt'), opts.platform === 'win32' ? README.replaceAll('\n', '\r\n') : README)
  console.log(`assemble-portable: ${opts.out} (${String(linked)} workspace packages materialized)`)
}

main()
