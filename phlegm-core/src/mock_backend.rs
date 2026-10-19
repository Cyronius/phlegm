//! Deterministic, hardware-free `Backend`, ported from
//! `tools/server/backends.py::MockBackend`.
//!
//! Echoes a short, prompt-derived reply as a stream of UTF-8 byte ids (which
//! `placeholder_tokenizer::PlaceholderTokenizer` decodes back to text), so
//! both streaming and non-streaming HTTP responses are meaningful without any
//! NPU device. A small per-token sleep simulates latency so SSE framing is
//! observable (mirrors Python's `time.sleep(0.005)`).

use crate::backend::{Backend, GenParams};
use std::thread::sleep;
use std::time::Duration;

pub struct MockBackend;

impl MockBackend {
    pub fn new() -> MockBackend {
        MockBackend
    }
}

impl Default for MockBackend {
    fn default() -> MockBackend {
        MockBackend::new()
    }
}

impl Backend for MockBackend {
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &mut GenParams,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<(), String> {
        let reply = format!(
            "[mock npu] received {} prompt tokens. interval-3 is finite here.",
            prompt_ids.len()
        );
        let out_ids: Vec<u32> = reply
            .bytes()
            .map(|b| b as u32)
            .take(params.max_tokens)
            .collect();
        for tid in out_ids {
            if params.stop_ids.contains(&tid) {
                return Ok(());
            }
            sleep(Duration::from_millis(5));
            on_token(tid);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Sampler;
    use std::collections::HashSet;

    #[test]
    fn echoes_prompt_len_and_respects_max_tokens() {
        let mut backend = MockBackend::new();
        let mut params = GenParams {
            max_tokens: 5,
            sampler: Sampler::new(0.0, 0, 1.0, 1.0, Some(1)),
            stop_ids: HashSet::new(),
        };
        let mut out = Vec::new();
        backend
            .generate(&[1, 2, 3], &mut params, &mut |tid| out.push(tid))
            .unwrap();
        assert_eq!(out.len(), 5);
        let expected = "[mock npu] received 3 prompt tokens. interval-3 is finite here.";
        let expected_prefix: Vec<u32> = expected.bytes().map(|b| b as u32).take(5).collect();
        assert_eq!(out, expected_prefix);
    }

    #[test]
    fn stops_at_stop_id() {
        let mut backend = MockBackend::new();
        let mut stop_ids = HashSet::new();
        // '[' is the first byte of the reply; stopping on it should yield nothing.
        stop_ids.insert(b'[' as u32);
        let mut params = GenParams {
            max_tokens: 100,
            sampler: Sampler::new(0.0, 0, 1.0, 1.0, Some(1)),
            stop_ids,
        };
        let mut out = Vec::new();
        backend
            .generate(&[1], &mut params, &mut |tid| out.push(tid))
            .unwrap();
        assert!(out.is_empty());
    }
}
