/**
 * Writes deterministic GGUF fixtures for engine-backed tests.
 *
 * The model is a bigram table: every transformer layer contributes zero, each
 * token embeds as a one-hot vector, and the output head maps chosen tokens to
 * fixed successors. Greedy decoding after the ChatML generation prompt spells
 * `reply` and then emits `<|im_end|>`.
 */
import { writeFile } from 'node:fs/promises'

const ALIGN = 32
const DIM = 256
const VOCAB = 259
const IM_END = 257

type KvValue = { u32: number } | { i32arr: number[] } | { str: string } | { strarr: string[] } | { bool: boolean } | { f32: number }
type Kv = readonly [string, KvValue]

class Writer {
  private parts: Buffer[] = []
  get length(): number { return this.parts.reduce((n, b) => n + b.length, 0) }
  u32(v: number): void { const b = Buffer.alloc(4); b.writeUInt32LE(v); this.parts.push(b) }
  i32(v: number): void { const b = Buffer.alloc(4); b.writeInt32LE(v); this.parts.push(b) }
  u64(v: number): void { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); this.parts.push(b) }
  f32(v: number): void { const b = Buffer.alloc(4); b.writeFloatLE(v); this.parts.push(b) }
  str(v: string): void { const b = Buffer.from(v, 'utf8'); this.u64(b.length); this.parts.push(b) }
  raw(b: Buffer): void { this.parts.push(b) }
  pad(): void { const n = (ALIGN - (this.length % ALIGN)) % ALIGN; if (n > 0) this.parts.push(Buffer.alloc(n)) }
  done(): Buffer { return Buffer.concat(this.parts) }
}

function byteTable(): string[] {
  const out: string[] = []
  let n = 0
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
    out.push(String.fromCodePoint(printable ? b : 256 + n++))
  }
  return out
}

/**
 * Write a llama-architecture fixture whose greedy reply is `reply`.
 * @param path - destination file.
 * @param reply - ASCII reply with no repeated character, so each token has one successor.
 * @param contextLength - trained context length the file declares.
 */
export async function writeBigramModel(path: string, reply: string, contextLength = 1024): Promise<void> {
  const successor = new Map<number, number>()
  let prev = '\n'.charCodeAt(0)
  for (const ch of reply) {
    successor.set(prev, ch.charCodeAt(0))
    prev = ch.charCodeAt(0)
  }
  successor.set(prev, IM_END)
  const tokens = [...byteTable(), '<|im_start|>', '<|im_end|>', '<|endoftext|>']
  const types = [...Array<number>(256).fill(1), 3, 3, 3]
  const kv: Kv[] = [
    ['general.architecture', { str: 'llama' }],
    ['general.name', { str: 'Bigram fixture' }],
    ['general.file_type', { u32: 0 }],
    ['llama.block_count', { u32: 1 }],
    ['llama.embedding_length', { u32: DIM }],
    ['llama.feed_forward_length', { u32: 32 }],
    ['llama.context_length', { u32: contextLength }],
    ['llama.attention.head_count', { u32: 2 }],
    ['llama.attention.head_count_kv', { u32: 1 }],
    ['llama.attention.layer_norm_rms_epsilon', { f32: 1e-5 }],
    ['tokenizer.ggml.model', { str: 'gpt2' }],
    ['tokenizer.ggml.pre', { str: 'qwen2' }],
    ['tokenizer.ggml.tokens', { strarr: tokens }],
    ['tokenizer.ggml.token_type', { i32arr: types }],
    ['tokenizer.ggml.merges', { strarr: [] }],
    ['tokenizer.ggml.eos_token_id', { u32: IM_END }],
    ['tokenizer.ggml.add_bos_token', { bool: false }],
  ]
  const head = DIM / 2
  const tensors: { name: string; shape: number[]; data: Float32Array }[] = []
  const add = (name: string, shape: number[], fill?: (data: Float32Array) => void) => {
    const data = new Float32Array(shape.reduce((a, b) => a * b, 1))
    fill?.(data)
    tensors.push({ name, shape, data })
  }
  add('token_embd.weight', [VOCAB, DIM], (d) => { for (let t = 0; t < VOCAB; t++) d[t * DIM + (t % DIM)] = 1 })
  add('output_norm.weight', [DIM], d => d.fill(1))
  add('output.weight', [VOCAB, DIM], (d) => { for (const [from, to] of successor) d[to * DIM + from] = 1 })
  add('blk.0.attn_norm.weight', [DIM], d => d.fill(1))
  add('blk.0.attn_q.weight', [DIM, DIM])
  add('blk.0.attn_k.weight', [head, DIM])
  add('blk.0.attn_v.weight', [head, DIM])
  add('blk.0.attn_output.weight', [DIM, DIM])
  add('blk.0.ffn_norm.weight', [DIM], d => d.fill(1))
  add('blk.0.ffn_gate.weight', [32, DIM])
  add('blk.0.ffn_up.weight', [32, DIM])
  add('blk.0.ffn_down.weight', [DIM, 32])

  const w = new Writer()
  w.u32(0x46554747)
  w.u32(3)
  w.u64(tensors.length)
  w.u64(kv.length)
  for (const [key, value] of kv) {
    w.str(key)
    if ('u32' in value) { w.u32(4); w.u32(value.u32) }
    else if ('f32' in value) { w.u32(6); w.f32(value.f32) }
    else if ('bool' in value) { w.u32(7); w.raw(Buffer.from([value.bool ? 1 : 0])) }
    else if ('str' in value) { w.u32(8); w.str(value.str) }
    else if ('strarr' in value) { w.u32(9); w.u32(8); w.u64(value.strarr.length); for (const s of value.strarr) w.str(s) }
    else { w.u32(9); w.u32(5); w.u64(value.i32arr.length); for (const v of value.i32arr) w.i32(v) }
  }
  let offset = 0
  for (const t of tensors) {
    w.str(t.name)
    w.u32(t.shape.length)
    // GGUF lists dimensions innermost first.
    for (const d of [...t.shape].reverse()) w.u64(d)
    w.u32(0)
    w.u64(offset)
    offset += Math.ceil(t.data.byteLength / ALIGN) * ALIGN
  }
  w.pad()
  for (const t of tensors) {
    w.raw(Buffer.from(t.data.buffer))
    w.pad()
  }
  await writeFile(path, w.done())
}
