//! Tokenizer reconstructed from GGUF `tokenizer.ggml.*` metadata.
//!
//! Supports the two vocabularies GGUF chat models ship with: SentencePiece-style
//! score merging (`llama`) and byte-level BPE with ranked merges (`gpt2`).
//! Control and user-defined tokens are matched atomically before merging.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use fancy_regex::Regex;

/// GGUF token type of a control token (`<|im_end|>`, `<s>`).
const TOKEN_TYPE_CONTROL: i32 = 3;
/// GGUF token type of a user-defined token rendered as text (`<think>`).
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
/// GGUF token type of a SentencePiece byte-fallback token (`<0x0A>`).
const TOKEN_TYPE_BYTE: i32 = 6;

/// Vocabulary algorithm declared by `tokenizer.ggml.model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Spm,
    Bpe,
}

/// Tokenizer state for one loaded model.
pub struct Tokenizer {
    kind: Kind,
    tokens: Vec<String>,
    types: Vec<i32>,
    scores: Vec<f32>,
    ids: HashMap<String, u32>,
    merges: HashMap<(String, String), usize>,
    /// Control and user-defined token texts, longest first, for atomic matching.
    specials: Vec<(String, u32)>,
    /// First bytes of special token texts; positions starting elsewhere skip the special scan.
    special_first: [bool; 256],
    pre: Option<Regex>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    byte_tokens: [Option<u32>; 256],
    add_space_prefix: bool,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    /// End-of-generation tokens: EOS plus template turn terminators found in the vocabulary.
    pub eog: Vec<u32>,
}

/// GPT-2 reversible byte-to-unicode table used by byte-level BPE vocabularies.
fn byte_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

/// Pre-tokenizer pattern for one `tokenizer.ggml.pre` identifier.
fn pre_pattern(pre: &str) -> &'static str {
    match pre {
        "qwen2" | "deepseek-r1-qwen" | "stablelm2" | "hunyuan" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        "gpt-2" | "gpt2" | "starcoder" | "refact" | "olmo" | "phi-2" => {
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        }
        // llama-bpe (Llama 3), smaug, dbrx, tekken and unknown identifiers.
        _ => r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    }
}

fn meta<'a>(content: &'a Content, key: &str) -> Option<&'a Value> {
    content.metadata.get(key)
}

fn meta_u32(content: &Content, key: &str) -> Option<u32> {
    meta(content, key).and_then(|v| v.to_u32().ok().or_else(|| v.to_i32().ok().map(|x| x as u32)))
}

fn meta_bool(content: &Content, key: &str) -> Option<bool> {
    meta(content, key).and_then(|v| v.to_bool().ok())
}

impl Tokenizer {
    /// Build the tokenizer from GGUF metadata.
    /// @param content - parsed GGUF header.
    pub fn from_gguf(content: &Content) -> Result<Self> {
        let model = meta(content, "tokenizer.ggml.model")
            .context("GGUF has no tokenizer.ggml.model")?
            .to_string()?
            .clone();
        let kind = match model.as_str() {
            "llama" | "replit" => Kind::Spm,
            "gpt2" => Kind::Bpe,
            other => bail!("unsupported tokenizer model `{other}`"),
        };
        let tokens: Vec<String> = meta(content, "tokenizer.ggml.tokens")
            .context("GGUF has no tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().cloned())
            .collect::<std::result::Result<_, _>>()?;
        let n = tokens.len();
        let types: Vec<i32> = match meta(content, "tokenizer.ggml.token_type") {
            Some(v) => v.to_vec()?.iter().map(|t| t.to_i32().unwrap_or(1)).collect(),
            None => vec![1; n],
        };
        let scores: Vec<f32> = match meta(content, "tokenizer.ggml.scores") {
            Some(v) => v.to_vec()?.iter().map(|t| t.to_f32().unwrap_or(0.0)).collect(),
            None => vec![0.0; n],
        };
        let mut merges = HashMap::new();
        if let Some(v) = meta(content, "tokenizer.ggml.merges") {
            for (rank, m) in v.to_vec()?.iter().enumerate() {
                let m = m.to_string()?;
                if let Some((a, b)) = m.split_once(' ') {
                    merges.insert((a.to_string(), b.to_string()), rank);
                }
            }
        }
        let mut ids = HashMap::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let ty = types.get(*i).copied().unwrap_or(1);
                (ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED) && !t.is_empty()
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let mut special_first = [false; 256];
        for (t, _) in &specials {
            special_first[t.as_bytes()[0] as usize] = true;
        }
        let byte_encoder = byte_table();
        let byte_decoder = byte_encoder.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let mut byte_tokens = [None; 256];
        for b in 0..256usize {
            if let Some(id) = ids.get(&format!("<0x{b:02X}>")) {
                byte_tokens[b] = Some(*id);
            }
        }
        let pre = match kind {
            Kind::Bpe => {
                let id = meta(content, "tokenizer.ggml.pre").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
                Some(Regex::new(pre_pattern(&id)).context("invalid pre-tokenizer pattern")?)
            }
            Kind::Spm => None,
        };
        let bos = meta_u32(content, "tokenizer.ggml.bos_token_id");
        let eos = meta_u32(content, "tokenizer.ggml.eos_token_id");
        let add_bos = meta_bool(content, "tokenizer.ggml.add_bos_token").unwrap_or(kind == Kind::Spm);
        let add_space_prefix = meta_bool(content, "tokenizer.ggml.add_space_prefix").unwrap_or(kind == Kind::Spm);
        let mut eog: Vec<u32> = eos.into_iter().collect();
        if let Some(e) = meta_u32(content, "tokenizer.ggml.eot_token_id") {
            eog.push(e);
        }
        for name in ["<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|eom_id|>", "<｜end▁of▁sentence｜>", "</s>"] {
            if let Some(id) = ids.get(name) {
                eog.push(*id);
            }
        }
        eog.sort_unstable();
        eog.dedup();
        Ok(Self {
            kind, tokens, types, scores, ids, merges, specials, special_first, pre, byte_encoder, byte_decoder, byte_tokens,
            add_space_prefix, bos, eos, add_bos, eog,
        })
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Text of one token, as stored in the vocabulary.
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Encode text, matching control and user-defined tokens atomically.
    /// @param text - rendered prompt text.
    /// @returns token ids without an added BOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        let mut first = true;
        while !rest.is_empty() {
            let mut cut = None;
            for (pos, _) in rest.char_indices() {
                if !self.special_first[rest.as_bytes()[pos] as usize] {
                    continue;
                }
                if let Some((s, id)) = self.specials.iter().find(|(s, _)| rest[pos..].starts_with(s.as_str())) {
                    cut = Some((pos, s.len(), *id));
                    break;
                }
            }
            match cut {
                Some((pos, len, id)) => {
                    if pos > 0 {
                        self.encode_plain(&rest[..pos], first, &mut out);
                    }
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_plain(rest, first, &mut out);
                    rest = "";
                }
            }
            first = false;
        }
        out
    }

    /// Encode text without matching special tokens, as llama.cpp does with `parse_special = false`.
    /// @param text - literal text.
    pub fn encode_plain_text(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if !text.is_empty() {
            self.encode_plain(text, true, &mut out);
        }
        out
    }

    fn encode_plain(&self, text: &str, first: bool, out: &mut Vec<u32>) {
        match self.kind {
            Kind::Spm => self.encode_spm(text, first, out),
            Kind::Bpe => self.encode_bpe(text, out),
        }
    }

    fn encode_spm(&self, text: &str, first: bool, out: &mut Vec<u32>) {
        let mut s = String::with_capacity(text.len() + 3);
        if first && self.add_space_prefix {
            s.push('▁');
        }
        s.push_str(&text.replace(' ', "▁"));
        for sym in self.spm_merge(&s) {
            match self.ids.get(sym) {
                Some(&id) => out.push(id),
                None => {
                    for b in sym.as_bytes() {
                        if let Some(id) = self.byte_tokens[*b as usize] {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// SentencePiece merge: repeatedly join the adjacent pair with the highest vocabulary score.
    /// Uses a linked list plus a max-heap so long prompts stay O(n log n).
    fn spm_merge<'a>(&self, s: &'a str) -> Vec<&'a str> {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct Cand { score: f32, left: usize, len: usize }
        impl Eq for Cand {}
        impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
        impl Ord for Cand {
            fn cmp(&self, o: &Self) -> Ordering {
                self.score.total_cmp(&o.score).then_with(|| o.left.cmp(&self.left))
            }
        }
        // Symbols as byte ranges in `s`; prev/next form a doubly linked list.
        let mut start: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
        let n = start.len();
        let mut len: Vec<usize> = (0..n).map(|i| if i + 1 < n { start[i + 1] - start[i] } else { s.len() - start[i] }).collect();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 < n as isize { i + 1 } else { -1 }).collect();
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Cand>, start: &[usize], len: &[usize], l: usize, r: usize| {
            let text = &s[start[l]..start[r] + len[r]];
            if let Some(&id) = self.ids.get(text) {
                heap.push(Cand { score: self.scores[id as usize], left: l, len: text.len() });
            }
        };
        for i in 1..n {
            push(&mut heap, &start, &len, i - 1, i);
        }
        while let Some(c) = heap.pop() {
            let l = c.left;
            let r = next[l];
            // Skip candidates invalidated by an earlier merge.
            if len[l] == 0 || r < 0 || len[l] + len[r as usize] != c.len {
                continue;
            }
            let r = r as usize;
            len[l] += len[r];
            len[r] = 0;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            if prev[l] >= 0 {
                push(&mut heap, &start, &len, prev[l] as usize, l);
            }
            if next[l] >= 0 {
                push(&mut heap, &start, &len, l, next[l] as usize);
            }
        }
        let _ = &mut start;
        let mut out = Vec::new();
        let mut i: isize = if n > 0 { 0 } else { -1 };
        while i >= 0 {
            let u = i as usize;
            out.push(&s[start[u]..start[u] + len[u]]);
            i = next[u];
        }
        out
    }

    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        let pre = self.pre.as_ref().expect("bpe tokenizer has a pre-tokenizer");
        for piece in pre.find_iter(text) {
            let Ok(piece) = piece else { continue };
            let mapped: String = piece.as_str().bytes().map(|b| self.byte_encoder[b as usize]).collect();
            if let Some(&id) = self.ids.get(&mapped) {
                out.push(id);
                continue;
            }
            let mut symbols: Vec<String> = mapped.chars().map(String::from).collect();
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in 0..symbols.len().saturating_sub(1) {
                    if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                        if best.map_or(true, |(_, r)| rank < r) {
                            best = Some((i, rank));
                        }
                    }
                }
                match best {
                    Some((i, _)) => {
                        let next = symbols.remove(i + 1);
                        symbols[i].push_str(&next);
                    }
                    None => break,
                }
            }
            for sym in symbols {
                match self.ids.get(&sym) {
                    Some(&id) => out.push(id),
                    None => {
                        // Unmergeable symbols fall back to single mapped bytes.
                        for c in sym.chars() {
                            if let Some(&id) = self.ids.get(&c.to_string()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Raw bytes one token contributes to the output; control tokens contribute none.
    /// @param id - token id.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let i = id as usize;
        let Some(text) = self.tokens.get(i) else { return Vec::new() };
        let ty = self.types.get(i).copied().unwrap_or(1);
        if ty == TOKEN_TYPE_CONTROL {
            return Vec::new();
        }
        if ty == TOKEN_TYPE_USER_DEFINED {
            return text.as_bytes().to_vec();
        }
        match self.kind {
            Kind::Spm => {
                if ty == TOKEN_TYPE_BYTE {
                    if let Some(hex) = text.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                        if let Ok(b) = u8::from_str_radix(hex, 16) {
                            return vec![b];
                        }
                    }
                }
                text.replace('▁', " ").into_bytes()
            }
            Kind::Bpe => text.chars().filter_map(|c| self.byte_decoder.get(&c).copied()).collect(),
        }
    }

    /// Whether the decoder strips one leading space from the first SentencePiece output token.
    pub fn strips_leading_space(&self) -> bool {
        self.kind == Kind::Spm && self.add_space_prefix
    }

}

/// Incremental UTF-8 assembly of token bytes, holding incomplete sequences.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Append bytes and return the longest complete UTF-8 prefix.
    /// @param bytes - bytes from one decoded token.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if e.error_len().is_some() {
                    // Invalid bytes cannot become valid; emit them lossily.
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }

    /// Flush any held bytes lossily at end of generation.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}
