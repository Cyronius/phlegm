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
}

pub trait Backend {
    /// Generate up to `params.max_tokens` tokens from `prompt_ids`, calling
    /// `on_token` for each one (in order) as it's produced. Stops early on a
    /// stop id (not passed to `on_token`). `Err` is a hard failure the caller
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
        on_token: &mut dyn FnMut(u32),
    ) -> Result<(), String>;
}
