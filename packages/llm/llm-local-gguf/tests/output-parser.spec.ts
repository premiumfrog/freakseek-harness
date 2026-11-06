/** Raw local-model text becomes reasoning, text, and tool-call blocks regardless of delta boundaries. */
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, StreamChunk } from '@deepseek-ai/dsh-llm'
import { expect, it } from 'vitest'
import { OutputParser } from '../src/output-parser.ts'

function run(deltas: readonly string[], opts: { opensReasoning?: boolean; tools?: string[] } = {}): ContentBlock[] {
  let n = 0
  const parser = new OutputParser({
    opensReasoning: opts.opensReasoning ?? false,
    toolNames: new Set(opts.tools ?? []),
    newCallId: () => `call_${++n}`,
  })
  const chunks: StreamChunk[] = []
  for (const d of deltas) chunks.push(...parser.push(d))
  chunks.push(...parser.finish(), { type: 'finish', reason: { kind: 'stop' } })
  const assembler = new BlockAssembler()
  for (const c of chunks) assembler.push(c)
  return assembler.blocks()
}

/** Every split of `text` into two deltas must parse the same. */
function everySplit(text: string, opts?: Parameters<typeof run>[1]): ContentBlock[] {
  const whole = run([text], opts)
  for (let i = 1; i < text.length; i++) expect(run([text.slice(0, i), text.slice(i)], opts)).toEqual(whole)
  return whole
}

it('passes plain text through', () => {
  expect(everySplit('Hello, world.')).toEqual([{ type: 'text', text: 'Hello, world.' }])
})

it('separates a leading think block from the answer', () => {
  expect(everySplit('<think>\nplan it\n</think>\n\nAnswer.')).toEqual([
    { type: 'reasoning', text: 'plan it' },
    { type: 'text', text: 'Answer.' },
  ])
})

it('treats output as reasoning when the prompt opened a think block', () => {
  expect(everySplit('step one</think>Done', { opensReasoning: true })).toEqual([
    { type: 'reasoning', text: 'step one' },
    { type: 'text', text: 'Done' },
  ])
})

it('parses Hermes tool calls with object arguments', () => {
  const blocks = everySplit('Let me look.\n<tool_call>\n{"name": "read", "arguments": {"path": "a.txt"}}\n</tool_call>', { tools: ['read'] })
  expect(blocks).toEqual([
    { type: 'text', text: 'Let me look.' },
    { type: 'tool-call', id: 'call_1', name: 'read', arguments: '{"path":"a.txt"}' },
  ])
})

it('parses several calls in one reply', () => {
  const blocks = run(['<tool_call>{"name":"a","arguments":{}}</tool_call>\n<tool_call>{"name":"b","arguments":"{\\"x\\":1}"}</tool_call>'], { tools: ['a', 'b'] })
  expect(blocks.map(b => b.type === 'tool-call' ? [b.name, b.arguments] : b.type)).toEqual([['a', '{}'], ['b', '{"x":1}']])
})

it('keeps a malformed tool call visible as text', () => {
  expect(run(['<tool_call>not json</tool_call>'], { tools: ['a'] })).toEqual([{ type: 'text', text: '<tool_call>not json</tool_call>' }])
})

it('parses Mistral tool-call arrays', () => {
  expect(run(['[TOOL_CALLS][{"name":"s","arguments":{"q":"x"}}]'], { tools: ['s'] })).toEqual([
    { type: 'tool-call', id: 'call_1', name: 's', arguments: '{"q":"x"}' },
  ])
})

it('parses a Llama bare-JSON call only for a declared tool', () => {
  expect(run(['{"name": "search", "parameters": {"q": "x"}}'], { tools: ['search'] })).toEqual([
    { type: 'tool-call', id: 'call_1', name: 'search', arguments: '{"q":"x"}' },
  ])
  expect(run(['{"name": "other", "parameters": {}}'], { tools: ['search'] })).toEqual([
    { type: 'text', text: '{"name": "other", "parameters": {}}' },
  ])
})

it('holds a partial marker until it resolves as text', () => {
  expect(run(['a <tool', '_x> b'], { tools: ['a'] })).toEqual([{ type: 'text', text: 'a <tool_x> b' }])
})
