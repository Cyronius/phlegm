//! A byte-level identity codec, ported from
//! `tools/server/tokenizer.py::PlaceholderTokenizer`.
//!
//! The real tokenizer lives in `tokenizer.rs` (`Tokenizer`, real BPE + jinja
//! chat template) and needs a `tokenizer.json` on disk. This module is the
//! swap-point fallback so the HTTP layer (`server.rs`) is fully exercisable
//! with zero model files: `encode` returns a string's UTF-8 bytes as ids
//! (0..255, genuinely reversible), `decode` renders 0..255 as their byte and
//! any id >= 256 (e.g. a real vocab id) as a visible `⟨id⟩` marker.

use crate::tokenizer::ChatMessage;

pub struct PlaceholderTokenizer;

impl PlaceholderTokenizer {
    pub fn new() -> PlaceholderTokenizer {
        PlaceholderTokenizer
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Must be incremental-prefix-safe: decoding a growing id list is
    /// monotonic (each call's output starts with the previous call's). True
    /// here because it's a byte-level identity codec.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::<u8>::new();
        let mut pieces = String::new();
        for &i in ids {
            if i < 256 {
                out.push(i as u8);
            } else {
                if !out.is_empty() {
                    pieces.push_str(&String::from_utf8_lossy(&out));
                    out.clear();
                }
                pieces.push('⟨');
                pieces.push_str(&i.to_string());
                pieces.push('⟩');
            }
        }
        if !out.is_empty() {
            pieces.push_str(&String::from_utf8_lossy(&out));
        }
        pieces
    }

    /// Minimal flatten (no real chat template — that's `Tokenizer`'s job):
    /// `"{role}: {content}"` per turn, joined with `\n`, plus a trailing
    /// `"assistant:"` generation-prompt line.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> String {
        let mut parts: Vec<String> = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect();
        parts.push("assistant:".to_string());
        parts.join("\n")
    }

    /// Unknown without a real vocab (mirrors `PlaceholderTokenizer.eos_id = None`).
    pub fn eos_id(&self) -> Option<u32> {
        None
    }
}

impl Default for PlaceholderTokenizer {
    fn default() -> PlaceholderTokenizer {
        PlaceholderTokenizer::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii() {
        let t = PlaceholderTokenizer::new();
        let ids = t.encode("hello world");
        assert_eq!(t.decode(&ids), "hello world");
    }

    #[test]
    fn renders_out_of_range_ids_as_markers() {
        let t = PlaceholderTokenizer::new();
        assert_eq!(t.decode(&[b'a' as u32, 300, b'b' as u32]), "a⟨300⟩b");
    }

    #[test]
    fn chat_template_flattens_turns() {
        let t = PlaceholderTokenizer::new();
        let msgs = vec![ChatMessage { role: "user".into(), content: "hi".into() }];
        assert_eq!(t.apply_chat_template(&msgs), "user: hi\nassistant:");
    }
}
