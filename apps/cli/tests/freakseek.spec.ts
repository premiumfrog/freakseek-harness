/** The freakseek command maps its arguments onto the cli profile. */
import { describe, expect, it } from 'vitest'
import { FREAKSEEK_HELP, FREAKSEEK_PROFILE, parseFreakSeekArgs } from '../src/freakseek.ts'

describe('parseFreakSeekArgs', () => {
  it.each([[[]], [['-h']], [['--help']], [['help']]])('prints help for %j', (argv) => {
    expect(parseFreakSeekArgs(argv)).toEqual({ kind: 'help' })
  })

  it.each([[['-V']], [['--version']]])('prints the version for %j', (argv) => {
    expect(parseFreakSeekArgs(argv)).toEqual({ kind: 'version' })
  })

  it('runs a task on the cli profile with its options in argv order', () => {
    expect(parseFreakSeekArgs(['--session-id', 's1', '--json', 'fix the tests'])).toEqual({
      kind: 'run', launcherArgs: ['--profile', FREAKSEEK_PROFILE, '--session-id', 's1', '--json', 'fix the tests'],
    })
    expect(FREAKSEEK_PROFILE).toBe('cli')
  })

  it('reserves the chat subcommand', () => {
    expect(parseFreakSeekArgs(['chat'])).toEqual({ kind: 'error', message: 'freakseek chat is not available yet; run freakseek "<task>"' })
  })

  it('names the product and the one-shot usage in its help', () => {
    expect(FREAKSEEK_HELP).toMatch(/^FreakSeek — /u)
    expect(FREAKSEEK_HELP).toContain('freakseek "<task>"')
  })
})
