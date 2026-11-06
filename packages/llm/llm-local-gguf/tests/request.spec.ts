/** Harness history becomes the OpenAI-style messages GGUF chat templates render. */
import { brandString } from '@deepseek-ai/dsh-brand'
import { createAssistantMessage, createSystemMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ToolCallId } from '@deepseek-ai/dsh-llm'
import { expect, it } from 'vitest'
import { engineMessages, engineTools, IMAGE_PLACEHOLDER } from '../src/request.ts'

const callId = brandString<ToolCallId>('call_1')
const user = (text: string) => createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] })

it('merges leading system text and maps a tool round trip', () => {
  const messages = engineMessages({
    provider: 'local',
    model: 'm',
    system: 'one-shot',
    messages: [
      createSystemMessage('prompt'),
      user('read a.txt'),
      createAssistantMessage({
        content: [
          { type: 'reasoning', text: 'need the file' },
          { type: 'text', text: 'Reading.' },
          { type: 'tool-call', id: callId, name: 'read', arguments: '{"path":"a.txt"}' },
        ],
        source: { provider: 'local', model: 'm' },
      }),
      createToolResultMessage({ callId, content: [{ type: 'text', text: 'denied' }], isError: true }),
    ],
  })
  expect(messages).toEqual([
    { role: 'system', content: 'one-shot\n\nprompt' },
    { role: 'user', content: 'read a.txt' },
    {
      role: 'assistant',
      content: 'Reading.',
      reasoning_content: 'need the file',
      tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'read', arguments: { path: 'a.txt' } } }],
    },
    { role: 'tool', tool_call_id: 'call_1', name: 'read', content: 'Error: denied' },
  ])
})

it('keeps malformed arguments as a string and replaces images with a placeholder', () => {
  const attachment = { attachmentId: 'a', mediaType: 'image/png', width: 1, height: 1, bytes: 1 } as never
  const messages = engineMessages({
    provider: 'local',
    model: 'm',
    messages: [
      createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'see ' }, { type: 'image', attachment }] }),
      createAssistantMessage({ content: [{ type: 'tool-call', id: callId, name: 'x', arguments: '{oops' }], source: { provider: 'local', model: 'm' } }),
    ],
  })
  expect(messages[0]).toEqual({ role: 'user', content: `see ${IMAGE_PLACEHOLDER}` })
  expect(messages[1]?.tool_calls?.[0]?.function.arguments).toBe('{oops')
})

it('omits deferred tools', () => {
  expect(engineTools({
    provider: 'local',
    model: 'm',
    messages: [],
    tools: [
      { name: 'a', description: 'A', parameters: { type: 'object' } },
      { name: 'b', description: 'B', parameters: {}, deferLoading: true },
    ],
  })).toEqual([{ type: 'function', function: { name: 'a', description: 'A', parameters: { type: 'object' } } }])
})
