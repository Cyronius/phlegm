//! Shared contract between generate-loop implementations (NPU-backed, and a
//! hardware-free mock) and anything that drives them (the CLI, the HTTP
//! server). Kept deliberately tiny and stable so those pieces can be built
//! independently against it.

use crate::sampler::Sampler;
use std::collections::HashSet;

pub struct GenParams {
    pub max_tokens: usize,
    pub sampler: Sampler,
    pub stop_ids: HashSet<u32>,
    /// Where the conversation's messages end in the prompt and the generation prompt begins
    /// (SERVER-PROMPT-REUSE): the next turn's prompt shares the tokens before it, so a backend
    /// that keeps its context saves its recurrent state there. None: the prompt's end.
    pub reuse_mark: Option<usize>,
}

/// A verified output token. Logits are the raw, untransformed model distribution
/// before penalties, constraints, temperature or sampling filters. Borrowed only
/// during the callback; consumers compute log probabilities only on demand.
pub struct TokenEvent<'a> {
    pub id: u32,
    pub logits: Option<&'a [f32]>,
}

pub trait Backend {
    /// Generate up to `params.max_tokens` tokens from `prompt_ids`, calling
    /// `on_token` for each one (in order) as it's produced. Stops early on a
    /// stop id (not passed to `on_token`) or callback `Break`. On `Break`, do
    /// not decode that token or issue another bridge command. Kept context must
    /// describe only tokens actually run by the bridge (the last output token
    /// need not be in it). Only verified tokens may reach this callback, including
    /// in a future speculative implementation. `Err` is a hard failure the caller
    /// should surface (e.g. NPU device busy/unavailable) — mirrors Python's
    /// `DeviceBusyError` distinction (HTTP 503 vs 500).
    ///
    /// `params` is `&mut` (not `&`) specifically so implementations can call
    /// `params.sampler.sample(...)` directly — `Sampler::sample` needs `&mut
    /// self` for its RNG state, and `Sampler` deliberately isn't `Clone` (it
    /// owns that state), so a shared reference here would leave every caller
    /// unable to use the sampler it was actually given.
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &mut GenParams,
        on_token: &mut dyn FnMut(TokenEvent<'_>) -> std::ops::ControlFlow<()>,
    ) -> Result<(), String>;

    /// Tokens the backend can hold (prompt plus output), when it has a fixed context. The
    /// server bounds an open-ended request by it, as llama.cpp's server does.
    fn context_capacity(&self) -> Option<usize> {
        None
    }

    /// Prompt tokens the last `generate` took from the previous requests' context instead of
    /// prefilling them (OpenAI's `usage.prompt_tokens_details.cached_tokens`).
    fn cached_tokens(&self) -> usize {
        0
    }
}
