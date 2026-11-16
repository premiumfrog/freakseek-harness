/** Sampling controls resolve per model family and yield to configured values. */
import { expect, it } from 'vitest'
import { Config, samplingOf } from '../src/config.ts'
import { samplingDefaults } from '../src/sampling-defaults.ts'

it('chooses the family recommendation from architecture, reasoning support, and distill name', () => {
  expect(samplingDefaults({ architecture: 'qwen35moe', name: 'Qwen3.6 35B A3B', reasoning: true }))
    .toEqual({ temperature: 0.6, topK: 20, topP: 0.95, minP: 0, repeatPenalty: 1 })
  expect(samplingDefaults({ architecture: 'qwen3', name: 'Qwen3 4B', reasoning: false }))
    .toMatchObject({ temperature: 0.7, topP: 0.8, topK: 20 })
  expect(samplingDefaults({ architecture: 'qwen2', name: 'Qwen2.5 7B Instruct', reasoning: false }))
    .toMatchObject({ temperature: 0.7, topP: 0.8, repeatPenalty: 1.05 })
  expect(samplingDefaults({ architecture: 'qwen2', name: 'DeepSeek R1 Distill Qwen 7B', reasoning: true }))
    .toMatchObject({ temperature: 0.6, topP: 0.95 })
  expect(samplingDefaults({ architecture: 'llama', name: 'Llama 3.1 8B', reasoning: false })).toMatchObject({ temperature: 0.6, topP: 0.9 })
  expect(samplingDefaults({ architecture: 'gemma', name: 'x', reasoning: false })).toEqual(samplingDefaults(undefined))
})

it('keeps each configured control over the family default', () => {
  const facts = { architecture: 'qwen2', name: 'Qwen2.5', reasoning: false }
  expect(samplingOf(Config({}), facts)).toEqual({ temperature: 0.7, topK: 20, topP: 0.8, minP: 0, repeatPenalty: 1.05, repeatLastN: 64 })
  expect(samplingOf(Config({ temperature: 0, topK: 5 }), facts)).toMatchObject({ temperature: 0, topK: 5, topP: 0.8 })
  expect(Config({}).constrainToolCalls.get()).toBe(true)
})
