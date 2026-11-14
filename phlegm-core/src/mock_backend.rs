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
        on_token: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
    ) -> Result<(), String> {
        if params.sampler.is_constrained() {
            // Hardware-free serving still exercises the actual sampler mask. Prefer closing a
            // value and EOS over extending it, so arbitrary supported schemas finish promptly.
            let mut logits = vec![0.0; 257];
            for b in [b' ', b'\n', b'\r', b'\t'] {
                logits[b as usize] = -100.0;
            }
            for b in [b'}', b']', b'"'] {
                logits[b as usize] = 10.0;
            }
            logits[256] = 20.0;
            let mut history = Vec::new();
            for _ in 0..params.max_tokens {
                let id = params.sampler.try_sample(&logits, &history)? as u32;
                if params.stop_ids.contains(&id) {
                    break;
                }
                history.push(i64::from(id));
                if on_token(crate::backend::TokenEvent {
                    id,
                    logits: Some(&logits),
                })
                .is_break()
                {
                    break;
                }
            }
            return Ok(());
        }
        let prompt = crate::placeholder_tokenizer::PlaceholderTokenizer::new().decode(prompt_ids);
        // A deterministic, no-argument tool fixture for SDK smoke tests. Real
        // inference never uses this backend. Other requests retain the byte echo.
        let tool = prompt
            .lines()
            .find_map(|line| line.strip_prefix("__phlegm_mock_tools__"))
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
        let required = prompt.contains("You must call at least one")
            || prompt.contains("You must call the function");
        let reply = if let Some(result) = prompt
            .lines()
            .filter_map(|s| s.strip_prefix("tool: "))
            .last()
        {
            format!("[mock npu] tool result: {result}")
        } else if required
            && tool
                .as_ref()
                .and_then(|v| v[0]["function"]["name"].as_str())
                .is_some()
        {
            let name = tool.as_ref().unwrap()[0]["function"]["name"]
                .as_str()
                .unwrap();
            format!("Checking.<tool_call><function={name}></function></tool_call>")
        } else {
            format!(
                "[mock npu] received {} prompt tokens. interval-3 is finite here.",
                prompt_ids.len()
            )
        };
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
            let mut logits = [0.0; 257];
            logits[tid as usize] = 10.0;
            if on_token(crate::backend::TokenEvent {
                id: tid,
                logits: Some(&logits),
            })
            .is_break()
            {
                break;
            }
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
            reuse_mark: None,
        };
        let mut out = Vec::new();
        backend
            .generate(&[1, 2, 3], &mut params, &mut |event| {
                out.push(event.id);
                std::ops::ControlFlow::Continue(())
            })
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
            reuse_mark: None,
        };
        let mut out = Vec::new();
        backend
            .generate(&[1], &mut params, &mut |event| {
                out.push(event.id);
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert!(out.is_empty());
    }
}
