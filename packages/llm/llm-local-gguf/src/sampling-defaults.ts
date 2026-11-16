/**
 * Sampling defaults per model family, applied to the controls a deployment leaves unset.
 *
 * Values follow each vendor's published recommendation for precise, coding-style work:
 * Qwen 3 thinking 0.6 / 0.95 / 20 and non-thinking 0.7 / 0.8 / 20 with `min_p` 0
 * (Qwen3 model cards and https://qwen.readthedocs.io/), Qwen 2.5 `generation_config.json`
 * 0.7 / 0.8 / 20 with repetition penalty 1.05, DeepSeek-R1 distills 0.6 / 0.95
 * (https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations), and
 * Llama 3 `generation_config.json` 0.6 / 0.9.
 *
 * @module dsh-llm-local-gguf/sampling-defaults
 */

import type { EngineSampling } from './types.ts'

/** The sampling controls a family default covers. */
export type SamplingDefaults = Pick<EngineSampling, 'temperature' | 'topK' | 'topP' | 'minP' | 'repeatPenalty'>

/** Model facts the family is recognized from. */
export interface SamplingModelFacts {
  /** `general.architecture` of the GGUF file. */
  readonly architecture: string
  /** Display name, which carries the fine-tune's family for distills. */
  readonly name: string
  /** The chat template opens reasoning blocks. */
  readonly reasoning: boolean
}

const GENERIC: SamplingDefaults = { temperature: 0.6, topK: 40, topP: 0.95, minP: 0.05, repeatPenalty: 1 }

/**
 * Choose the sampling defaults for a model.
 * @param facts - the model's architecture, name, and reasoning support; omitted for a model the catalog does not know.
 * @returns the family's recommended controls.
 */
export function samplingDefaults(facts: SamplingModelFacts | undefined): SamplingDefaults {
  if (facts === undefined) return GENERIC
  if (/deepseek[-_ ]?r1/i.test(facts.name)) return { temperature: 0.6, topK: 40, topP: 0.95, minP: 0, repeatPenalty: 1 }
  switch (facts.architecture) {
    case 'qwen3':
    case 'qwen3moe':
    case 'qwen35':
    case 'qwen35moe':
      return facts.reasoning
        ? { temperature: 0.6, topK: 20, topP: 0.95, minP: 0, repeatPenalty: 1 }
        : { temperature: 0.7, topK: 20, topP: 0.8, minP: 0, repeatPenalty: 1 }
    case 'qwen2':
      return { temperature: 0.7, topK: 20, topP: 0.8, minP: 0, repeatPenalty: 1.05 }
    case 'llama':
      return { temperature: 0.6, topK: 40, topP: 0.9, minP: 0.05, repeatPenalty: 1 }
    default:
      // Architectures without a published recommendation, including later additions, use the generic values.
      return GENERIC
  }
}
