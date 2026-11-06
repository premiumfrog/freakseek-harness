/**
 * Harness request history translated into the OpenAI-style messages that
 * GGUF chat templates render.
 *
 * @module dsh-llm-local-gguf/request
 */

import { LlmError } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, RequestMessage } from '@deepseek-ai/dsh-llm'
import type { EngineMessage, EngineTool, EngineToolCall } from './types.ts'

/** Placeholder for an image a text-only local model cannot see. */
export const IMAGE_PLACEHOLDER = '[image omitted: this local model reads text only]'

function textOf(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        parts.push(block.text)
        break
      case 'image':
        parts.push(IMAGE_PLACEHOLDER)
        break
      default:
        // Reasoning, tool calls, and tool-change blocks are projected separately or not at all.
        break
    }
  }
  return parts.join('')
}

function reasoningOf(blocks: readonly ContentBlock[]): string {
  return blocks.filter(b => b.type === 'reasoning').map(b => b.text).join('')
}

function argumentsObject(raw: string): unknown {
  try {
    const value: unknown = JSON.parse(raw)
    return value
  }
  catch (_malformedArguments) {
    // Templates render arguments with `tojson`; a malformed string stays visible as a string.
    return raw
  }
}

/**
 * Translate one request's history.
 * @param options - assembled request.
 * @returns engine messages, with every leading system text merged into the first message.
 */
export function engineMessages(options: GenerateOptions): EngineMessage[] {
  const out: EngineMessage[] = []
  const system: string[] = options.system === undefined || options.system === '' ? [] : [options.system]
  const toolNames = new Map<string, string>()
  let leading = true
  for (const message of options.messages as readonly RequestMessage[]) {
    if (message.role === 'system' && leading) {
      const text = textOf(message.content)
      if (text !== '') system.push(text)
      continue
    }
    leading = false
    switch (message.role) {
      case 'system':
      case 'developer': {
        // Templates accept one leading system message; later instructions travel as user turns.
        const text = textOf(message.content)
        if (text !== '') out.push({ role: 'user', content: text })
        break
      }
      case 'user':
        out.push({ role: 'user', content: textOf(message.content) })
        break
      case 'assistant': {
        const calls: EngineToolCall[] = []
        for (const block of message.content) {
          if (block.type !== 'tool-call') continue
          toolNames.set(block.id, block.name)
          calls.push({ id: block.id, type: 'function', function: { name: block.name, arguments: argumentsObject(block.arguments) } })
        }
        const reasoning = reasoningOf(message.content)
        out.push({
          role: 'assistant',
          content: textOf(message.content),
          ...reasoning === '' ? {} : { reasoning_content: reasoning },
          ...calls.length === 0 ? {} : { tool_calls: calls },
        })
        break
      }
      case 'tool': {
        const content = textOf(message.content)
        out.push({
          role: 'tool',
          tool_call_id: message.toolCallId,
          name: toolNames.get(message.toolCallId) ?? 'tool',
          content: message.isError === true ? `Error: ${content}` : content,
        })
        break
      }
      default: {
        const role: never = message
        throw new LlmError(`local-gguf: unsupported message role ${JSON.stringify(role)}`, 'UNSUPPORTED_CONTENT')
      }
    }
  }
  if (system.length > 0) out.unshift({ role: 'system', content: system.join('\n\n') })
  return out
}

/**
 * Translate tool declarations, omitting deferred tools.
 * @param options - assembled request.
 * @returns OpenAI-style function declarations.
 */
export function engineTools(options: GenerateOptions): EngineTool[] {
  return (options.tools ?? [])
    .filter(tool => tool.deferLoading !== true)
    .map(tool => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }))
}
