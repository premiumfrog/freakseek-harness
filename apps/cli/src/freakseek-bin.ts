#!/usr/bin/env node
/**
 * Installed `freakseek` command entry.
 * @module @deepseek-ai/dsh/freakseek-bin
 */

/* v8 ignore file -- built-bin acceptance exercises this self-executing dispatch; parseFreakSeekArgs carries the logic. */

import { getDshRuntimeVersion } from '@deepseek-ai/dsh-app-boot'
import { FREAKSEEK_HELP, parseFreakSeekArgs } from './freakseek.ts'

const command = parseFreakSeekArgs(process.argv.slice(2))
switch (command.kind) {
  case 'help':
    process.stdout.write(FREAKSEEK_HELP)
    break
  case 'version':
    process.stdout.write(`freakseek ${getDshRuntimeVersion()}\n`)
    break
  case 'error':
    process.stderr.write(`${command.message}\n`)
    process.exitCode = 2
    break
  case 'run': {
    const { runCli } = await import('./bin.ts')
    await runCli({}, command.launcherArgs)
    break
  }
  default:
    command satisfies never
}
