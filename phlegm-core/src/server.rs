//! OpenAI-compatible HTTP server, ported from `tools/server/app.py` +
//! `tools/server/config.py`.
//!
//! Endpoints:
//!   GET  /health                 -> {"status":"ok","backend":...}
//!   GET  /v1/models              -> OpenAI model list
//!   POST /v1/chat/completions    -> chat completion; stream=true gives SSE deltas
//!   POST /v1/completions         -> raw prompt completion (no chat template)
//!   POST /v1/embeddings          -> embeddings
//!   POST /v1/images/generations  -> native image generation (multi-model mode)
//!   POST /v1/images/edits        -> native image edits (multi-model mode)
//!
//! Backend-agnostic: `serve()` takes any `Box<dyn Backend + Send>` (a
//! `NpuBackend` for normal serving, `MockBackend` for explicit tests).
//! Generation is serialized behind a `Mutex` around the backend, mirroring
//! Python's per-backend lock ("the NPU is a single shared device").
//!
//! Single-model threading: a single accept loop calls `Server::recv()` and handles
//! each request to completion (including writing the full SSE stream) before
//! looping back for the next one. tiny_http's own accept thread still queues
//! new connections concurrently (see `Server::from_listener`'s internal
//! `TaskPool`), so this isn't "one connection at a time on the wire" — it's
//! "one *response body* being produced at a time", which is what the shared
//! backend lock would force anyway. Lazy multimodal serving in `models` uses bounded
//! concurrent handlers and one FIFO model lease; discovery stays available during generation.

use crate::backend::{Backend, GenParams};
use crate::placeholder_tokenizer::PlaceholderTokenizer;
use crate::reasoning_effort::{template_kwargs, Levels, TemplateVars, Thinking};
use crate::sampler::{Penalties, Ramp, Sampler};
use crate::tokenizer::{ChatMessage, Reasoning, Tokenizer};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Request, Response};

#[path = "server_model_slot.rs"]
mod model_slot;
#[path = "server_models.rs"]
pub mod models;
#[path = "server_embeddings.rs"]
mod embeddings_route;
#[path = "server_generation.rs"]
mod generation;

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------

/// Mirrors `tools/server/config.py`: plain fields with env-var overrides.
#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_id: String,
    /// Output tokens for a request without `max_tokens`; 0 = until a stop token or the end of
    /// the context (llama.cpp's server default, `n_predict = -1`).
    pub default_max_tokens: usize,
    /// Upper bound on any request's `max_tokens`; 0 = none beyond the context.
    pub max_tokens_cap: usize,
    /// Directory containing `tokenizer.json` / `tokenizer_config.json` /
    /// `chat_template.jinja` (or an inline template in tokenizer_config.json).
    /// Required for an NPU backend; mock mode may use placeholder tokens.
    pub model_dir: Option<PathBuf>,
    /// Reported verbatim in `GET /health`; selected by the CLI after load.
    pub backend_name: String,
    /// Sampling and template defaults for a request that does not set them (the model's
    /// resolved settings, SERVER-SETTINGS).
    pub defaults: GenDefaults,
    /// Bearer authentication, disabled when absent.
    pub api_key: Option<String>,
    /// `loopback`, comma-separated exact origins, or `*`.
    pub allowed_origins: String,
}

/// What a request leaves out: the model's resolved `temperature`, `top_k`, `top_p`,
/// `repetition_penalty`, `think` and `reasoning_effort` settings, and the registry entry's
/// table of the template's effort values.
#[derive(Clone, Debug, PartialEq)]
pub struct GenDefaults {
    pub speculative: crate::speculative::SpeculativeMode,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub thinking: Thinking,
    pub effort_levels: Option<Levels>,
    /// presence / frequency (a request may set them), the repetition penalty's scope and the
    /// length / think-end ramps (`ids` filled per request: the stop tokens, `</think>`).
    pub penalties: Penalties,
}

impl Default for GenDefaults {
    /// The settings' built-in defaults: greedy, the template's thinking on.
    fn default() -> Self {
        GenDefaults {
            speculative: Default::default(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            thinking: Thinking::On,
            effort_levels: None,
            penalties: Penalties::default(),
        }
    }
}

impl GenDefaults {
    pub fn from_settings(r: &crate::settings::Resolved) -> Result<GenDefaults, String> {
        Ok(GenDefaults {
            speculative: r.speculative()?,
            temperature: r.f32("temperature"),
            top_k: r.usize("top_k"),
            top_p: r.f32("top_p"),
            repetition_penalty: r.f32("repetition_penalty"),
            thinking: Thinking::from_settings(&r.str("reasoning_effort"), r.bool("think"))?,
            effort_levels: Levels::from_entry(r.entry.as_ref())?,
            penalties: penalties_from_settings(r),
        })
    }
}

/// The settings' penalties, with the ramps' token ids left empty.
pub fn penalties_from_settings(r: &crate::settings::Resolved) -> Penalties {
    let ramp = |strength: &str, start: &str, window: &str, until_seen: bool| Ramp {
        ids: Vec::new(),
        strength: r.f32(strength),
        start: r.usize(start),
        window: r.usize(window),
        until_seen,
    };
    Penalties {
        presence: r.f32("presence_penalty"),
        frequency: r.f32("frequency_penalty"),
        repetition_reply_only: r.str("repetition_penalty_scope") == "response",
        ramps: vec![
            ramp("length_bias", "length_bias_start", "length_bias_window", false),
            ramp("think_end_bias", "think_end_start", "think_end_window", true),
        ],
    }
}

/// The penalties for one request: the ramps get their tokens (the length ramp the stop tokens, the
/// think-end ramp the reasoning block's closing token; without one it stays off).
pub fn penalties_for(mut p: Penalties, stop_ids: &HashSet<u32>, think_end: Option<u32>) -> Penalties {
    if let [length, think, ..] = p.ramps.as_mut_slice() {
        length.ids = stop_ids.iter().copied().collect();
        length.ids.sort_unstable();
        think.ids = think_end.into_iter().collect();
    }
    p.ramps.retain(|r| r.strength != 0.0 && !r.ids.is_empty());
    p
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

impl ServerConfig {
    pub fn from_env() -> ServerConfig {
        ServerConfig {
            host: env_or("PHLEGM_HOST", "127.0.0.1"),
            port: env_or("PHLEGM_PORT", "52625").parse().unwrap_or(52625),
            model_id: env_or("PHLEGM_MODEL_ID", "phlegm-mock"),
            default_max_tokens: env_or("PHLEGM_MAX_TOKENS", "0").parse().unwrap_or(0),
            max_tokens_cap: env_or("PHLEGM_MAX_TOKENS_CAP", "0").parse().unwrap_or(0),
            model_dir: std::env::var("PHLEGM_MODEL_DIR").ok().map(PathBuf::from),
            backend_name: env_or("PHLEGM_BACKEND", "mock"),
            defaults: GenDefaults::default(),
            api_key: std::env::var("PHLEGM_API_KEY").ok().filter(|v| !v.is_empty()),
            allowed_origins: env_or("PHLEGM_ALLOWED_ORIGINS", "loopback"),
        }
    }
}

// --------------------------------------------------------------------------
// Tokenizer selection (a real model must load; mock mode can use a placeholder)
// --------------------------------------------------------------------------

enum ServerTokenizer {
    Real(Tokenizer),
    Placeholder(PlaceholderTokenizer),
}

impl ServerTokenizer {
    fn decode_stream(&self) -> Box<dyn FnMut(u32) -> Result<Option<String>, String> + '_> {
        match self {
            Self::Real(t) => Box::new(t.decode_stream()),
            Self::Placeholder(_) => {
                let mut pending = Vec::new();
                Box::new(move |id| {
                    let byte = u8::try_from(id).map_err(|_| "non-byte mock output token")?;
                    pending.push(byte);
                    match std::str::from_utf8(&pending) {
                        Ok(text) => { let text = text.to_owned(); pending.clear(); Ok(Some(text)) }
                        Err(e) if e.error_len().is_none() => Ok(None),
                        Err(_) => Err("invalid UTF-8 mock output".into()),
                    }
                })
            }
        }
    }

    fn final_decode(&self, ids: &[u32]) -> Result<String, String> {
        match self {
            Self::Real(t) => t.decode(ids),
            Self::Placeholder(t) => Ok(t.decode(ids)),
        }
    }

    fn build(cfg: &ServerConfig) -> Result<ServerTokenizer, String> {
        if let Some(dir) = &cfg.model_dir {
            match Tokenizer::load(dir) {
                Ok(t) => return Ok(ServerTokenizer::Real(t)),
                Err(e) if cfg.backend_name != "mock" => return Err(e),
                Err(e) => eprintln!("server: tokenizer unavailable ({e}); using mock tokenizer"),
            }
        }
        if cfg.backend_name != "mock" {
            return Err(
                "NPU serving requires PHLEGM_MODEL_DIR with a tokenizer and chat template".into(),
            );
        }
        Ok(ServerTokenizer::Placeholder(PlaceholderTokenizer::new()))
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            ServerTokenizer::Real(t) => t.encode(text).unwrap_or_else(|e| {
                eprintln!("server: tokenizer encode error: {e}");
                Vec::new()
            }),
            ServerTokenizer::Placeholder(t) => t.encode(text),
        }
    }

    /// add_generation_prompt=true (a completion always follows), with the template variables.
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        kwargs: &serde_json::Map<String, Value>,
    ) -> Result<String, String> {
        match self {
            ServerTokenizer::Real(t) => t.render_chat(messages, true, kwargs),
            ServerTokenizer::Placeholder(t) => {
                let mut prompt = t.apply_chat_template(messages);
                if let Some(tools) = kwargs.get("tools").filter(|v| v.is_array()) {
                    prompt.push_str(&format!("\n__phlegm_mock_tools__{tools}"));
                }
                Ok(prompt)
            },
        }
    }

    /// The prompt position where the conversation's messages end and the generation prompt
    /// begins: how far the prompt with the generation prompt and the one without agree
    /// (SERVER-PROMPT-REUSE). None when the template does not render without it.
    fn conversation_end(
        &self,
        messages: &[ChatMessage],
        kwargs: &serde_json::Map<String, Value>,
        prompt_ids: &[u32],
    ) -> Option<usize> {
        let ServerTokenizer::Real(t) = self else { return None };
        let text = t.render_chat(messages, false, kwargs).ok()?;
        let ids = t.encode(&text).ok()?;
        Some(ids.iter().zip(prompt_ids).take_while(|(a, b)| a == b).count())
    }

    fn think_end_id(&self) -> Option<u32> {
        match self {
            ServerTokenizer::Real(t) => t.think_end_id(),
            ServerTokenizer::Placeholder(_) => None,
        }
    }

    fn template_vars(&self) -> TemplateVars {
        match self {
            ServerTokenizer::Real(t) => t.template_vars(),
            ServerTokenizer::Placeholder(_) => TemplateVars::default(),
        }
    }

    fn reasoning_for(&self, prompt: &str) -> Option<Reasoning> {
        match self {
            ServerTokenizer::Real(t) => t.reasoning_for(prompt),
            ServerTokenizer::Placeholder(_) => None,
        }
    }

    fn stop_ids(&self) -> HashSet<u32> {
        match self {
            ServerTokenizer::Real(t) => t.stop_ids(),
            ServerTokenizer::Placeholder(t) => t.eos_id().into_iter().collect(),
        }
    }
}

/// Without a fixed context (the mock backend) an open-ended request still needs a bound.
const UNBOUNDED_FALLBACK_TOKENS: usize = 4096;

/// The output-token budget for one request, the way llama.cpp's server sets it: the request's
/// `max_tokens`, else the configured default (0 = open-ended), under the configured cap (0 =
/// none), and never past the end of the backend's context (a request that reaches it finishes
/// with `length`). A prompt that leaves no room for one token is an error.
fn output_budget(
    requested: Option<usize>,
    cfg: &ServerConfig,
    context_capacity: Option<usize>,
    prompt_len: usize,
) -> Result<usize, String> {
    let room = match context_capacity {
        // The backend holds prompt + max_tokens - 1 (the last token is never fed back).
        Some(cap) if prompt_len > cap => {
            return Err(format!(
                "prompt is {prompt_len} tokens; the model's context holds {cap}"
            ))
        }
        Some(cap) => cap + 1 - prompt_len,
        None => UNBOUNDED_FALLBACK_TOKENS,
    };
    let mut n = match requested.unwrap_or(cfg.default_max_tokens) {
        0 => room,
        n => n,
    };
    if cfg.max_tokens_cap > 0 {
        n = n.min(cfg.max_tokens_cap);
    }
    Ok(n.min(room).max(1))
}

// --------------------------------------------------------------------------
// Server state / entry points
// --------------------------------------------------------------------------

struct ServerState {
    backend: Mutex<Box<dyn Backend + Send>>,
    /// The backend's fixed context, read once at start (None for a backend without one).
    context_capacity: Option<usize>,
    tokenizer: ServerTokenizer,
    cfg: ServerConfig,
    embedder: Option<Mutex<Box<dyn crate::embeddings::EmbeddingBackend>>>,
    embedding_model: Option<String>,
    embedding_only: bool,
}

/// Bind the listening socket without starting the accept loop. Split out from
/// `serve()` so tests (and anything else that wants the OS-assigned port from
/// `:0`) can learn the real address before requests start flowing.
pub fn bind(cfg: &ServerConfig) -> Result<tiny_http::Server, String> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    tiny_http::Server::http(&addr).map_err(|e| format!("binding {addr}: {e}"))
}

/// Run the accept loop against an already-bound server. Blocks forever (or
/// until the socket errors out / is closed).
pub fn run(
    server: tiny_http::Server,
    backend: Box<dyn Backend + Send>,
    cfg: ServerConfig,
) -> Result<(), String> {
    run_inner(server, backend, cfg, None, false)
}

fn run_inner(
    server: tiny_http::Server,
    backend: Box<dyn Backend + Send>,
    cfg: ServerConfig,
    embedder: Option<Box<dyn crate::embeddings::EmbeddingBackend>>,
    embedding_only: bool,
) -> Result<(), String> {
    let tokenizer = if embedding_only { ServerTokenizer::Placeholder(PlaceholderTokenizer::new()) } else { ServerTokenizer::build(&cfg)? };
    let backend_name = cfg.backend_name.clone();
    let model_id = embedder.as_ref().filter(|_| embedding_only).map(|e| e.model_id().to_owned()).unwrap_or_else(|| cfg.model_id.clone());
    let context_capacity = backend.context_capacity();
    let embedding_model = embedder.as_ref().map(|e| e.model_id().to_owned());
    let state = Arc::new(ServerState {
        backend: Mutex::new(backend),
        context_capacity,
        tokenizer,
        cfg,
        embedder: embedder.map(Mutex::new),
        embedding_model,
        embedding_only,
    });

    println!(
        "phlegm server on http://{}:{}",
        state.cfg.host, state.cfg.port
    );
    println!("  backend: {backend_name}   model: {model_id}");
    if embedding_only {
        println!("  POST /v1/embeddings | GET /v1/models | GET /health");
    } else {
        println!("  POST /v1/chat/completions | POST /v1/completions | GET /v1/models | GET /health");
    }

    loop {
        match server.recv() {
            Ok(request) => handle_request(request, &state),
            Err(e) => {
                eprintln!("server: recv error: {e}");
                break;
            }
        }
    }
    Ok(())
}

/// Serve one embedding model. The NPU is already held by the embedder; no chat model is loaded.
pub fn serve_embeddings(embedder: Box<dyn crate::embeddings::EmbeddingBackend>, mut cfg: ServerConfig) -> Result<(), String> {
    cfg.model_dir = None;
    let server = bind(&cfg)?;
    cfg.backend_name = "trit-embed".into();
    run_inner(server, Box::new(crate::mock_backend::MockBackend::new()), cfg, Some(embedder), true)
}

/// Bind + run in one call — the entry point `phlegm serve` calls.
pub fn serve(backend: Box<dyn Backend + Send>, cfg: ServerConfig) -> Result<(), String> {
    let server = bind(&cfg)?;
    run(server, backend, cfg)
}

// --------------------------------------------------------------------------
// Routing
// --------------------------------------------------------------------------

fn loopback_origin(origin: &str) -> bool {
    let Some(authority) = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://")) else { return false };
    ["localhost", "127.0.0.1", "[::1]"].iter().any(|host| {
        authority.strip_prefix(host).is_some_and(|rest| {
            rest.is_empty() || rest.strip_prefix(':').is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok())
        })
    })
}

fn authorize(request: &Request, cfg: &ServerConfig) -> bool {
    let origin = request.headers().iter().find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str());
    let allowed = if cfg.allowed_origins.trim() == "*" {
        Some("*".to_string())
    } else if cfg.allowed_origins.trim() == "loopback" {
        origin.filter(|o| loopback_origin(o)).map(str::to_string)
    } else {
        origin.filter(|o| cfg.allowed_origins.split(',').any(|a| a.trim() == *o))
            .map(str::to_string)
    };
    CORS_ORIGIN.with(|slot| *slot.borrow_mut() = allowed);
    let method = request.method().clone();
    let url = request.url().to_string();
    if method == Method::Options { return true; }
    // `/health` and the chat page stay public; the page asks for the key itself.
    if let Some(key) = cfg.api_key.as_ref().filter(|_| !(method == Method::Get && (url == "/health" || url == "/"))) {
        return request.headers().iter().any(|h| h.field.equiv("Authorization")
            && h.value.as_str().strip_prefix("Bearer ") == Some(key.as_str()));
    }
    true
}

/// The built-in chat page (U10), served at `/` by the single-model and multi-model servers.
pub(crate) fn respond_chat_page(request: Request) {
    let response = Response::from_string(include_str!("chat.html"))
        .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap());
    let _ = request.respond(response);
}

fn handle_request(request: Request, state: &Arc<ServerState>) {
    if !authorize(&request, &state.cfg) {
        respond_json(request, 401, &json!({"error": {"message": "Invalid API key", "type": "authentication_error"}}));
        return;
    }
    let method = request.method().clone();
    let url = request.url().to_string();
    if method == Method::Options {
        respond_json(request, 200, &json!({}));
        return;
    }
    if method == Method::Get && url == "/" {
        return respond_chat_page(request);
    }
    match (method, url.as_str()) {
        (Method::Get, "/health") => {
            respond_json(
                request,
                200,
                &json!({"status": "ok", "backend": state.cfg.backend_name}),
            );
        }
        (Method::Get, "/v1/models") => {
            respond_json(request, 200, &models_payload(state));
        }
        (Method::Post, "/v1/chat/completions") if !state.embedding_only => {
            handle_chat_completions(request, state, false);
        }
        (Method::Post, "/v1/completions") if !state.embedding_only => {
            handle_chat_completions(request, state, true);
        }
        (Method::Post, "/v1/embeddings") => embeddings_route::handle(request, state),
        _ => {
            let msg = format!("unknown path {url}");
            respond_json(
                request,
                404,
                &json!({"error": {"message": msg, "type": "not_found"}}),
            );
        }
    }
}

fn models_payload(state: &ServerState) -> Value {
    let mut ids = Vec::new();
    if !state.embedding_only { ids.push(state.cfg.model_id.as_str()); }
    if let Some(id) = state.embedding_model.as_deref() { if !ids.contains(&id) { ids.push(id); } }
    json!({
        "object": "list",
        "data": ids.into_iter().map(|id| json!({
            "id": id,
            "object": "model",
            "created": now_unix(),
            "owned_by": "fastflowlm-open-npu",
        })).collect::<Vec<_>>(),
    })
}

// --------------------------------------------------------------------------
// /v1/chat/completions
// --------------------------------------------------------------------------

fn handle_chat_completions(mut request: Request, state: &Arc<ServerState>, legacy: bool) {
    let mut body = Vec::new();
    if let Err(e) = request.as_reader().read_to_end(&mut body) {
        let msg = format!("invalid JSON body: {e}");
        respond_json(
            request,
            400,
            &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
        );
        return;
    }
    handle_chat_body(request, state, legacy, &body);
}

fn handle_chat_body(request: Request, state: &Arc<ServerState>, legacy: bool, body: &[u8]) {
    let req_json: Value = if body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("invalid JSON body: {e}");
                respond_json(
                    request,
                    400,
                    &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
                );
                return;
            }
        }
    };

    let options = match generation::Options::parse(&req_json, legacy) {
        Ok(v) => v,
        Err(e) => { respond_json(request, 400, &json!({"error":{"message":e,"type":"invalid_request_error"}})); return; }
    };
    let messages: Vec<ChatMessage> = if legacy { vec![] } else {
        match req_json.get("messages").and_then(Value::as_array) {
            Some(arr) if !arr.is_empty() => arr.iter().map(parse_message).collect(),
            _ => { respond_json(request, 400, &json!({"error":{"message":"'messages' must be a non-empty array","type":"invalid_request_error"}})); return; }
        }
    };
    let model = req_json
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.cfg.model_id)
        .to_string();
    if model != state.cfg.model_id {
        respond_json(request, 404, &json!({"error":{"message":"model not found","type":"not_found"}}));
        return;
    }
    let requested_max_tokens = req_json
        .get("max_completion_tokens").filter(|v| !v.is_null())
        .or_else(|| req_json.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let temperature = req_json
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let top_p = req_json
        .get("top_p")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let seed = req_json.get("seed").and_then(Value::as_u64);
    // llama.cpp's names: top_k, repeat_penalty (HF's repetition_penalty is accepted too).
    let top_k_field = req_json.get("top_k").filter(|v| !v.is_null());
    let top_k = top_k_field.and_then(Value::as_u64).map(|v| v as usize);
    let penalty_field = ["repetition_penalty", "repeat_penalty"]
        .iter()
        .find_map(|k| req_json.get(*k).filter(|v| !v.is_null()));
    let repetition_penalty = penalty_field.and_then(Value::as_f64).map(|v| v as f32);
    // OpenAI's presence_penalty / frequency_penalty, in [-2, 2], on the reply's tokens.
    let reply_penalty = |key: &str| {
        let field = req_json.get(key).filter(|v| !v.is_null());
        (field.is_some(), field.and_then(Value::as_f64).map(|v| v as f32))
    };
    let (presence_given, presence_penalty) = reply_penalty("presence_penalty");
    let (frequency_given, frequency_penalty) = reply_penalty("frequency_penalty");
    let bad_reply_penalty = |given: bool, v: Option<f32>| given && !v.is_some_and(|v| v.is_finite() && (-2.0..=2.0).contains(&v));
    // Thinking (SERVER-REASONING-EFFORT): OpenAI's reasoning_effort; else enable_thinking as vLLM
    // and llama.cpp take it (chat_template_kwargs), or OpenFlowLM's `think`.
    let kwargs_field = req_json.get("chat_template_kwargs").filter(|v| !v.is_null());
    let effort_field = req_json.get("reasoning_effort").filter(|v| !v.is_null());
    let think = kwargs_field
        .and_then(|k| k.get("enable_thinking"))
        .or_else(|| req_json.get("think"))
        .and_then(Value::as_bool);
    let thinking = match (effort_field, kwargs_field) {
        (Some(e), _) if !e.is_string() => Err("reasoning_effort must be a string".to_string()),
        (_, Some(k)) if !k.is_object() => Err("chat_template_kwargs must be an object".to_string()),
        _ => state
            .cfg
            .defaults
            .thinking
            .for_request(effort_field.and_then(Value::as_str), think),
    };
    let thinking = match thinking {
        Ok(t) => t,
        Err(msg) => {
            respond_json(
                request,
                400,
                &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
            );
            return;
        }
    };

    if temperature.is_some_and(|v| !v.is_finite() || v < 0.0)
        || top_p.is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        || (top_k_field.is_some() && top_k.is_none())
        || (penalty_field.is_some()
            && !repetition_penalty.is_some_and(|v| v.is_finite() && v > 0.0))
        || bad_reply_penalty(presence_given, presence_penalty)
        || bad_reply_penalty(frequency_given, frequency_penalty)
    {
        respond_json(
            request,
            400,
            &json!({"error": {"message": "temperature must be nonnegative, top_p in (0, 1], top_k a nonnegative integer, repetition_penalty positive and presence_penalty / frequency_penalty in [-2, 2]", "type": "invalid_request_error"}}),
        );
        return;
    }

    let d = &state.cfg.defaults;
    // The translated switches, then the request's own template variables as written.
    let mut kwargs = template_kwargs(&thinking, state.tokenizer.template_vars(), d.effort_levels.as_ref());
    if let Some(Value::Object(extra)) = kwargs_field {
        kwargs.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    let tools = match crate::tool_calls::Tools::from_request(if legacy { &Value::Null } else { &req_json }) {
        Ok(tools) => tools,
        Err(e) => {
            respond_json(
                request,
                400,
                &json!({"error":{"message":e,"type":"invalid_request_error"}}),
            );
            return;
        }
    };
    kwargs.insert("tools".into(), tools.tools.clone());
    kwargs.insert("tool_choice".into(), tools.choice.clone());
    let mut messages = messages;
    if tools.choice == "required" || tools.named.is_some() {
        let instruction = tools.named.as_ref().map_or_else(
            || "You must call at least one of the available tools.".to_owned(),
            |name| format!("You must call the function {name}."),
        );
        // Most publisher templates do not inspect tool_choice; express the choice in their system turn.
        if messages[0].role == "system" {
            messages[0].content.push_str(&format!("\n{instruction}"));
        } else {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: instruction,
                    ..Default::default()
                },
            );
        }
    }
    let tool_format = match &state.tokenizer {
        ServerTokenizer::Real(t) => t.tool_format(),
        _ => Some(crate::tool_calls::Format::Function),
    }
    .filter(|_| tools.enabled);
    if tools.enabled && tool_format.is_none() {
        respond_json(
            request,
            400,
            &json!({"error":{"message":"model template has no supported tool-call format","type":"invalid_request_error"}}),
        );
        return;
    }
    let format = req_json
        .get("response_format")
        .filter(|v| !legacy && !v.is_null() && v.get("type") != Some(&json!("text")));
    if format.is_some() && (tools.choice == "required" || tools.named.is_some()) {
        respond_json(
            request,
            400,
            &json!({"error":{"message":"response_format cannot be combined with required or named tool_choice; use auto or none","type":"invalid_request_error"}}),
        );
        return;
    }
    let constraint = match format
        .map(|f| {
            let vocab = match &state.tokenizer {
                ServerTokenizer::Real(t) => t.structured_vocabulary()?,
                ServerTokenizer::Placeholder(_) => {
                    Arc::new(crate::structured_output::Vocabulary::new(
                        (0u16..256)
                            .map(|b| Some(vec![b as u8]))
                            .chain([None])
                            .collect(),
                        [256].into_iter().collect(),
                    ))
                }
            };
            crate::structured_output::Constraint::from_response_format(f, vocab)
        })
        .transpose()
    {
        Ok(c) => c,
        Err(e) => {
            respond_json(
                request,
                400,
                &json!({"error":{"message":e,"type":"invalid_request_error"}}),
            );
            return;
        }
    };
    if constraint.is_some() {
        kwargs.insert("enable_thinking".into(), json!(false));
    }
    let prompt = match if legacy { Ok(String::new()) } else { state.tokenizer.apply_chat_template(&messages, &kwargs) } {
        Ok(p) => p,
        Err(e) => {
            respond_json(
                request,
                400,
                &json!({"error": {"message": e, "type": "invalid_request_error"}}),
            );
            return;
        }
    };
    let prompt_ids = if legacy {
        match generation::prompt_ids(&req_json, &state.tokenizer) {
            Ok(ids) => ids,
            Err(e) => { respond_json(request, 400, &json!({"error":{"message":e,"type":"invalid_request_error"}})); return; }
        }
    } else { state.tokenizer.encode(&prompt) };
    let max_tokens = match output_budget(
        requested_max_tokens,
        &state.cfg,
        state.context_capacity,
        prompt_ids.len(),
    ) {
        Ok(n) => n,
        Err(msg) => {
            respond_json(
                request,
                400,
                &json!({"error": {"message": msg, "type": "invalid_request_error"}}),
            );
            return;
        }
    };
    let mut stop_ids = state.tokenizer.stop_ids();
    if constraint.is_some() && matches!(state.tokenizer, ServerTokenizer::Placeholder(_)) {
        stop_ids.insert(256);
    }
    let reasoning = if legacy || constraint.is_some() {
        None
    } else {
        state.tokenizer.reasoning_for(&prompt)
    };

    let sampler = Sampler::new(
        temperature.unwrap_or(d.temperature),
        top_k.unwrap_or(d.top_k),
        top_p.unwrap_or(d.top_p),
        repetition_penalty.unwrap_or(d.repetition_penalty),
        seed,
    )
    .with_penalties(penalties_for(
        Penalties {
            presence: presence_penalty.unwrap_or(d.penalties.presence),
            frequency: frequency_penalty.unwrap_or(d.penalties.frequency),
            ..d.penalties.clone()
        },
        &stop_ids,
        state.tokenizer.think_end_id(),
    ));
    let sampler = if let Some(c) = constraint {
        sampler.with_constraint(c)
    } else {
        sampler
    };
    let params = GenParams {
        speculative: state.cfg.defaults.speculative,
        max_tokens,
        sampler,
        stop_ids,
        reuse_mark: if legacy { None } else { state.tokenizer.conversation_end(&messages, &kwargs, &prompt_ids) },
    };

    generation::respond(request, state, prompt_ids, params, model, reasoning,
        tool_format.map(|f| (f, tools)), options);
}

fn parse_message(m: &Value) -> ChatMessage {
    let role = m
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string();
    let content = match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let extra = m
        .as_object()
        .map(|o| {
            o.iter()
                .filter(|(k, _)| !matches!(k.as_str(), "role" | "content"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();
    ChatMessage {
        role,
        content,
        extra,
    }
}

/// Parse calls before splitting reasoning: harmony's recipient lives in the header, not its final channel.
fn assistant_message(
    text: &str,
    reasoning: Option<&Reasoning>,
    tools: Option<&(crate::tool_calls::Format, crate::tool_calls::Tools)>,
) -> (Value, bool) {
    let parsed = tools.map(|(format, cfg)| {
        let mut parsed = format.parse(text);
        cfg.restore_argument_types(&mut parsed.calls);
        if cfg.accepts(&parsed.calls) {
            parsed
        } else {
            crate::tool_calls::Parsed {
                content: text.into(),
                calls: vec![],
            }
        }
    });
    let raw = parsed.as_ref().map_or(text, |p| p.content.as_str());
    let mut message = if let Some(r) = reasoning {
        let (thought, answer) =
            if tools.is_some_and(|(format, _)| *format == crate::tool_calls::Format::Harmony) {
                crate::tool_calls::harmony_parts(raw)
            } else {
                r.split(raw)
            };
        json!({"role":"assistant","content":answer,"reasoning_content":thought})
    } else {
        json!({"role":"assistant","content":raw})
    };
    let has_calls = parsed.as_ref().is_some_and(|p| !p.calls.is_empty());
    if let Some(mut p) = parsed.filter(|p| !p.calls.is_empty()) {
        for c in &mut p.calls {
            c["id"] = json!(format!(
                "call_{}",
                cmpl_id().trim_start_matches("chatcmpl-")
            ));
        }
        message["tool_calls"] = json!(p.calls);
        if message["content"] == "" {
            message["content"] = Value::Null;
        }
    }
    (message, has_calls)
}

fn final_stream_tail<'a>(full: &'a str, emitted: &str) -> Result<&'a str, String> {
    let tail = full
        .strip_prefix(emitted)
        .ok_or("stream decoder changed already emitted text")?;
    if tail.ends_with('\u{fffd}') {
        return Err("stream ended with unresolved UTF-8/replacement text".into());
    }
    Ok(tail)
}

fn sse_frame(payload: &Value) -> String {
    format!("data: {}\n\n", payload)
}

// --------------------------------------------------------------------------
// Raw HTTP/1.1 chunked SSE framing over `Request::into_writer()`
// --------------------------------------------------------------------------
//
// tiny_http's `Response` type picks chunked transfer encoding automatically
// whenever `data_length` is `None` (see `response.rs::raw_print`), but it
// does so by wrapping the whole reader in one `io::copy` call — fine for a
// bounded body, wrong for SSE, where each token must hit the socket as soon
// as it's produced. `Request::into_writer()` hands back the raw
// `Box<dyn Write + Send>` underneath (the same mechanism tiny_http documents
// for CGI-style raw output), with no headers written for it — exactly what
// Python's `BaseHTTPRequestHandler` does by hand in `_send_sse_headers` /
// `_sse_write`, so this mirrors that 1:1 instead of guessing at a
// streaming-`Response` API that doesn't exist in this crate.

fn write_sse_headers(writer: &mut dyn Write) -> std::io::Result<()> {
    writer.write_all(b"HTTP/1.1 200 OK\r\n")?;
    writer.write_all(b"Content-Type: text/event-stream\r\n")?;
    writer.write_all(b"Cache-Control: no-cache\r\n")?;
    writer.write_all(b"Connection: keep-alive\r\n")?;
    writer.write_all(b"Transfer-Encoding: chunked\r\n")?;
    CORS_ORIGIN.with(|slot| -> std::io::Result<()> {
        if let Some(origin) = slot.borrow().as_ref() {
            write!(writer, "Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\n")?;
        }
        Ok(())
    })?;
    writer.write_all(b"\r\n")?;
    writer.flush()
}

fn sse_write(writer: &mut dyn Write, data: &str) -> std::io::Result<()> {
    let payload = data.as_bytes();
    write!(writer, "{:X}\r\n", payload.len())?;
    writer.write_all(payload)?;
    writer.write_all(b"\r\n")?;
    writer.flush()
}

fn sse_end(writer: &mut dyn Write) -> std::io::Result<()> {
    writer.write_all(b"0\r\n\r\n")?;
    writer.flush()
}

// --------------------------------------------------------------------------
// Small helpers
// --------------------------------------------------------------------------

fn json_content_type() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap()
}

thread_local! { static CORS_ORIGIN: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) }; }

pub(crate) fn respond_json(request: Request, status: u16, value: &Value) {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let mut response = Response::from_data(body)
        .with_status_code(status)
        .with_header(json_content_type())
        .with_header(Header::from_bytes("Access-Control-Allow-Methods", "GET, POST, OPTIONS").unwrap())
        .with_header(Header::from_bytes("Access-Control-Allow-Headers", "Authorization, Content-Type").unwrap())
        .with_header(Header::from_bytes("Vary", "Origin").unwrap());
    CORS_ORIGIN.with(|slot| {
        if let Some(origin) = slot.borrow().as_ref() {
            if let Ok(header) = Header::from_bytes("Access-Control-Allow-Origin", origin.as_bytes()) {
                response.add_header(header);
            }
        }
    });
    let _ = request.respond(response);
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Mirrors `"chatcmpl-" + uuid.uuid4().hex[:24]` closely enough for the
/// purpose (an opaque, non-colliding completion id) without adding a `uuid`
/// dependency: nanosecond timestamp + a process-wide counter, both hex.
fn cmpl_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "chatcmpl-{:012x}{:012x}",
        nanos & 0xFFFF_FFFF_FFFF,
        seq & 0xFFFF_FFFF_FFFF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    include!("server_tools_tests.rs");
    include!("server_generation_tests.rs");
    include!("server_speculative_tests.rs");

    #[test]
    fn stream_finalization_flushes_only_unemitted_text_and_rejects_rewrites() {
        assert_eq!(final_stream_tail("hello!", "hello").unwrap(), "!");
        assert_eq!(final_stream_tail("🙂", "🙂").unwrap(), "");
        assert!(final_stream_tail("🙂", "�").is_err());
        assert!(final_stream_tail("hello�", "hello").is_err());
    }

    #[test]
    fn real_tokenizer_sse_buffers_utf8_and_reports_truncated_terminal_bytes() {
        struct Tokens(Vec<u32>);
        impl Backend for Tokens {
            fn generate(
                &mut self,
                _: &[u32],
                _: &mut GenParams,
                emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
            ) -> Result<(), String> {
                for &id in &self.0 {
                    if emit(crate::backend::TokenEvent { id, logits: None }).is_break() { break; }
                }
                Ok(())
            }
        }
        for (ids, expected, error) in [(vec![3, 4, 5, 6, 2], "🙂A", false), (vec![2, 3], "A", true)]
        {
            let (dir, tokenizer) = crate::tokenizer::tests::fixture(false, false);
            let cfg = ServerConfig {
                host: "127.0.0.1".into(),
                port: 0,
                model_id: "fixture".into(),
                default_max_tokens: 16,
                max_tokens_cap: 16,
                model_dir: None,
                backend_name: "fixture".into(),
                defaults: GenDefaults::default(),
                api_key: None,
                allowed_origins: "*".into(),
            };
            let server = bind(&cfg).unwrap();
            let addr = server.server_addr().to_ip().unwrap().to_string();
            let state = Arc::new(ServerState {
                backend: Mutex::new(Box::new(Tokens(ids))),
                context_capacity: None,
                tokenizer: ServerTokenizer::Real(tokenizer),
                cfg,
                embedder: None,
                embedding_model: None,
                embedding_only: false,
            });
            let thread = std::thread::spawn(move || handle_request(server.recv().unwrap(), &state));
            let (status, _, body) = http_request(
                &addr,
                "POST",
                "/v1/chat/completions",
                Some(r#"{"messages":[{"role":"user","content":"A"}],"stream":true}"#),
            );
            assert_eq!(status, 200);
            let frames: Vec<Value> = body
                .split("\n\n")
                .filter_map(|frame| frame.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| serde_json::from_str(data).unwrap())
                .collect();
            let text: String = frames
                .iter()
                .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
                .collect();
            assert_eq!(text, expected);
            assert_eq!(frames.iter().any(|v| v.get("error").is_some()), error);
            if error {
                assert!(!frames
                    .iter()
                    .any(|v| v["choices"][0]["finish_reason"].is_string()));
            }
            assert!(body.contains("data: [DONE]"));
            thread.join().unwrap();
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    use crate::mock_backend::MockBackend;
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;

    /// Bind on an OS-assigned port, spawn the accept loop on a background
    /// thread, return the concrete address to connect to. The thread is
    /// intentionally leaked (daemon-style) — the test process exits when the
    /// test function returns.
    fn spawn_test_server(cfg_mut: impl FnOnce(&mut ServerConfig)) -> String {
        let mut cfg = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            model_id: "test-model".to_string(),
            default_max_tokens: 64,
            max_tokens_cap: 512,
            model_dir: None,
            backend_name: "mock".to_string(),
            defaults: GenDefaults::default(),
            api_key: None,
            allowed_origins: "*".into(),
        };
        cfg_mut(&mut cfg);
        let server = bind(&cfg).expect("bind");
        let addr = server.server_addr().to_ip().expect("ip addr");
        std::thread::spawn(move || {
            let _ = run(server, Box::new(MockBackend::new()), cfg);
        });
        format!("{}:{}", addr.ip(), addr.port())
    }

    /// Minimal blocking HTTP/1.1 client: send a raw request, return
    /// (status_code, headers, body_as_string). Reads the body either via
    /// Content-Length or by decoding a chunked transfer-encoding.
    pub(super) fn http_request(
        addr: &str,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> (u16, Vec<String>, String) {
        http_request_headers(addr, method, path, body, "")
    }

    pub(super) fn http_request_headers(addr: &str, method: &str, path: &str, body: Option<&str>, extra: &str) -> (u16, Vec<String>, String) {
        let mut stream = TcpStream::connect(addr).expect("connect");
        let body_bytes = body.unwrap_or("").as_bytes();
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
        req.push_str(extra);
        if body.is_some() {
            req.push_str("Content-Type: application/json\r\n");
            req.push_str(&format!("Content-Length: {}\r\n", body_bytes.len()));
        }
        req.push_str("\r\n");
        stream.write_all(req.as_bytes()).unwrap();
        if let Some(b) = body {
            stream.write_all(b.as_bytes()).unwrap();
        }

        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line).unwrap();
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut headers = Vec::new();
        let mut content_length: Option<usize> = None;
        let mut chunked = false;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let trimmed = line.trim_end().to_string();
            if trimmed.is_empty() {
                break;
            }
            let lower = trimmed.to_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                content_length = v.trim().parse().ok();
            }
            if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
                chunked = true;
            }
            headers.push(trimmed);
        }

        let body_out = if chunked {
            let mut out = Vec::new();
            loop {
                let mut size_line = String::new();
                reader.read_line(&mut size_line).unwrap();
                let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
                if size == 0 {
                    // consume trailing CRLF after the terminating 0-chunk
                    let mut trailer = String::new();
                    reader.read_line(&mut trailer).ok();
                    break;
                }
                let mut chunk = vec![0u8; size];
                std::io::Read::read_exact(&mut reader, &mut chunk).unwrap();
                out.extend_from_slice(&chunk);
                let mut crlf = [0u8; 2];
                std::io::Read::read_exact(&mut reader, &mut crlf).unwrap();
            }
            out
        } else if let Some(len) = content_length {
            let mut out = vec![0u8; len];
            std::io::Read::read_exact(&mut reader, &mut out).unwrap();
            out
        } else {
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut out).unwrap();
            out
        };

        (
            status,
            headers,
            String::from_utf8_lossy(&body_out).to_string(),
        )
    }

    #[test]
    fn chat_page_is_public_html_with_api_key_configured() {
        let addr = spawn_test_server(|cfg| cfg.api_key = Some("secret".into()));
        let (status, headers, body) = http_request(&addr, "GET", "/", None);
        assert_eq!(status, 200);
        assert!(headers.iter().any(|h| h.eq_ignore_ascii_case("content-type: text/html; charset=utf-8")));
        assert!(body.contains("Phlegm chat"));
        assert!(body.contains("/v1/chat/completions"));
        assert_eq!(http_request(&addr, "GET", "/v1/models", None).0, 401);
    }

    #[test]
    fn cors_and_api_key_apply_to_preflight_json_and_streams() {
        let addr = spawn_test_server(|cfg| {
            cfg.api_key = Some("secret".into());
            cfg.allowed_origins = "https://example.test".into();
        });
        let (status, headers, _) = http_request_headers(&addr, "OPTIONS", "/v1/chat/completions", None,
            "Origin: https://example.test\r\n");
        assert_eq!(status, 200);
        assert!(headers.iter().any(|h| h.eq_ignore_ascii_case("access-control-allow-origin: https://example.test")));
        assert_eq!(http_request(&addr, "GET", "/v1/models", None).0, 401);
        assert_eq!(http_request(&addr, "GET", "/health", None).0, 200);
        let (status, headers, _) = http_request_headers(&addr, "POST", "/v1/chat/completions",
            Some(r#"{"messages":[{"role":"user","content":"x"}],"stream":true,"max_tokens":1}"#),
            "Origin: https://example.test\r\nAuthorization: Bearer secret\r\n");
        assert_eq!(status, 200);
        assert!(headers.iter().any(|h| h.eq_ignore_ascii_case("access-control-allow-origin: https://example.test")));
        let (_, headers, _) = http_request_headers(&addr, "GET", "/v1/models", None,
            "Origin: https://evil.test\r\nAuthorization: Bearer secret\r\n");
        assert!(!headers.iter().any(|h| h.to_lowercase().starts_with("access-control-allow-origin:")));
    }

    #[test]
    fn cors_default_limits_origins_to_loopback() {
        let addr = spawn_test_server(|cfg| cfg.allowed_origins = "loopback".into());
        for origin in ["http://localhost", "https://localhost:3000", "http://127.0.0.1:8080", "https://[::1]:65535"] {
            let (_, headers, _) = http_request_headers(&addr, "OPTIONS", "/v1/models", None, &format!("Origin: {origin}\r\n"));
            assert!(headers.iter().any(|h| h.eq_ignore_ascii_case(&format!("access-control-allow-origin: {origin}"))), "{origin}");
        }
        for origin in ["https://evil.test", "http://localhost.evil.test", "http://127.0.0.1@evil.test", "http://[::1].evil", "http://localhost:abc", "null", "ftp://localhost"] {
            let (_, headers, _) = http_request_headers(&addr, "OPTIONS", "/v1/models", None, &format!("Origin: {origin}\r\n"));
            assert!(!headers.iter().any(|h| h.to_lowercase().starts_with("access-control-allow-origin:")), "{origin}");
        }
        let wildcard = spawn_test_server(|cfg| cfg.allowed_origins = "*".into());
        let (_, headers, _) = http_request_headers(&wildcard, "OPTIONS", "/v1/models", None, "Origin: https://evil.test\r\n");
        assert!(headers.iter().any(|h| h.eq_ignore_ascii_case("access-control-allow-origin: *")));
        assert_eq!(ServerConfig::from_env().allowed_origins, std::env::var("PHLEGM_ALLOWED_ORIGINS").unwrap_or_else(|_| "loopback".into()));
    }

    struct FakeEmbedding;
    impl crate::embeddings::EmbeddingBackend for FakeEmbedding {
        fn model_id(&self) -> &str { "fake-embed" }
        fn embed(&mut self, input: crate::embeddings::EmbeddingInput) -> Result<crate::embeddings::EmbeddingOutput, crate::embeddings::EmbeddingError> {
            use crate::embeddings::{EmbeddingError, EmbeddingInput, EmbeddingOutput};
            if matches!(&input, EmbeddingInput::Texts(v) if v.iter().any(|s| s == "backend failure")) {
                return Err(EmbeddingError::Backend("fake failure".into()));
            }
            if matches!(&input, EmbeddingInput::TokenIds(v) if v.iter().flatten().any(|&id| id >= 100)) {
                return Err(EmbeddingError::Invalid("invalid token id".into()));
            }
            let lengths: Vec<usize> = match input {
                EmbeddingInput::Texts(v) => v.iter().map(|s| s.len()).collect(),
                EmbeddingInput::TokenIds(v) => v.iter().map(Vec::len).collect(),
            };
            Ok(EmbeddingOutput { tokens: lengths.iter().sum(), vectors: lengths.iter().map(|n| vec![*n as f32, -1.0]).collect() })
        }
    }

    fn spawn_embedding_server() -> String {
        let mut cfg = ServerConfig::from_env();
        cfg.port = 0;
        cfg.api_key = None;
        cfg.backend_name = "fake-embed".into();
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        std::thread::spawn(move || run_inner(server, Box::new(MockBackend::new()), cfg, Some(Box::new(FakeEmbedding)), true));
        addr
    }

    #[test]
    fn embeddings_shapes_batches_base64_usage_and_errors() {
        let addr = spawn_embedding_server();
        let (_, _, body) = http_request(&addr, "GET", "/v1/models", None);
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"], "fake-embed");
        for (input, expected, tokens) in [
            (json!("cat"), json!([[3.0, -1.0]]), 3),
            (json!(["a", "dogs"]), json!([[1.0, -1.0], [4.0, -1.0]]), 5),
            (json!([1, 2, 3]), json!([[3.0, -1.0]]), 3),
            (json!([[1], [2, 3]]), json!([[1.0, -1.0], [2.0, -1.0]]), 3),
        ] {
            let req = json!({"model":"fake-embed", "input":input});
            let (status, _, body) = http_request(&addr, "POST", "/v1/embeddings", Some(&req.to_string()));
            assert_eq!(status, 200, "{body}");
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["object"], "list");
            assert_eq!(v["model"], "fake-embed");
            assert_eq!(v["usage"]["prompt_tokens"], tokens);
            assert_eq!(v["usage"]["total_tokens"], tokens);
            for (i, row) in v["data"].as_array().unwrap().iter().enumerate() {
                assert_eq!(row["object"], "embedding");
                assert_eq!(row["index"], i);
                assert_eq!(row["embedding"], expected[i]);
            }
        }
        let req = json!({"model":"fake-embed", "input":"a", "encoding_format":"base64"});
        let (_, _, body) = http_request(&addr, "POST", "/v1/embeddings", Some(&req.to_string()));
        let v: Value = serde_json::from_str(&body).unwrap();
        let bytes = base64::decode(v["data"][0]["embedding"].as_str().unwrap()).unwrap();
        assert_eq!(bytes, [1f32.to_le_bytes(), (-1f32).to_le_bytes()].concat());
        for req in [json!({"model":"bad","input":"x"}), json!({"model":"fake-embed","input":[]}),
            json!({"model":"fake-embed","input":[1,-1]}), json!({"model":"fake-embed","input":[[100]]}), json!({"model":"fake-embed","input":[[]]}),
            json!({"model":"fake-embed","input":"x","dimensions":2}), json!({"model":"fake-embed","input":"x","encoding_format":"bogus"}), json!({"input":"x"})] {
            let status = http_request(&addr, "POST", "/v1/embeddings", Some(&req.to_string())).0;
            assert!(status == 400 || status == 404, "{req}: {status}");
        }
        let failure = json!({"model":"fake-embed", "input":"backend failure"});
        assert_eq!(http_request(&addr, "POST", "/v1/embeddings", Some(&failure.to_string())).0, 500);
        assert_eq!(http_request(&addr, "POST", "/v1/chat/completions", Some("{}")).0, 404);
    }

    #[test]
    #[ignore = "requires PHLEGM_TRIT_CPU_PYTHON and PHLEGM_TRIT_MODEL with the CPU reference dependencies"]
    fn embeddings_route_matches_trit_cpu_reference() {
        use crate::embeddings::{EmbeddingBackend, EmbeddingError, EmbeddingInput, EmbeddingOutput};
        use std::process::{Command, Stdio};
        fn cpu(texts: &[String]) -> EmbeddingOutput {
            let code = r#"
import importlib.util, json, sys, torch
sys.dont_write_bytecode = True
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('reference', sys.argv[1] + '/trit_embed.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
model = mod.TritEmbed.from_pretrained(sys.argv[1])
texts = json.load(sys.stdin)
ids = model.tokenizer(texts, truncation=True, max_length=128)['input_ids']
print(json.dumps({'vectors': model.encode(texts).tolist(), 'tokens': sum(map(len, ids))}))
"#;
            let mut process = Command::new(std::env::var("PHLEGM_TRIT_CPU_PYTHON").unwrap())
                .args(["-I", "-c", code, &std::env::var("PHLEGM_TRIT_MODEL").unwrap()])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
            serde_json::to_writer(process.stdin.take().unwrap(), texts).unwrap();
            let result = process.wait_with_output().unwrap();
            assert!(result.status.success(), "CPU reference failed");
            let v: Value = serde_json::from_slice(&result.stdout).unwrap();
            EmbeddingOutput { vectors: serde_json::from_value(v["vectors"].clone()).unwrap(), tokens: v["tokens"].as_u64().unwrap() as usize }
        }
        struct Cpu;
        impl EmbeddingBackend for Cpu {
            fn model_id(&self) -> &str { "trit-cpu" }
            fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError> {
                match input { EmbeddingInput::Texts(texts) => Ok(cpu(&texts)), _ => Err(EmbeddingError::Invalid("texts only".into())) }
            }
        }
        let texts = vec!["how do index funds work".to_string(), "A man is playing a guitar on the street.".to_string(), "Rain is expected tomorrow.".to_string()];
        let expected = cpu(&texts);
        let mut cfg = ServerConfig::from_env(); cfg.port = 0; cfg.api_key = None;
        let server = bind(&cfg).unwrap(); let addr = server.server_addr().to_ip().unwrap().to_string();
        std::thread::spawn(move || run_inner(server, Box::new(MockBackend::new()), cfg, Some(Box::new(Cpu)), true));
        let req = json!({"model": "trit-cpu", "input": texts});
        let (status, _, body) = http_request(&addr, "POST", "/v1/embeddings", Some(&req.to_string()));
        assert_eq!(status, 200, "{body}");
        let response: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(response["usage"]["total_tokens"], expected.tokens);
        for (i, expected_vector) in expected.vectors.iter().enumerate() {
            let got = response["data"][i]["embedding"].as_array().unwrap();
            assert_eq!(got.len(), expected_vector.len());
            for (a, b) in got.iter().zip(expected_vector) { assert!((a.as_f64().unwrap() - *b as f64).abs() < 1e-6); }
        }
    }

    #[test]
    fn completion_token_alias_and_stream_usage() {
        let addr = spawn_test_server(|_| {});
        let body = r#"{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"max_completion_tokens":3,"stream":true,"stream_options":{"include_usage":true}}"#;
        let (status, _, response) = http_request(&addr, "POST", "/v1/chat/completions", Some(body));
        assert_eq!(status, 200);
        let frames: Vec<Value> = response.split("\n\n").filter_map(|f| f.strip_prefix("data: "))
            .filter_map(|s| serde_json::from_str(s).ok()).collect();
        let last = frames.last().unwrap();
        assert_eq!(last["choices"], json!([]));
        assert_eq!(last["usage"]["completion_tokens"], 3);
    }

    #[test]
    fn health_reports_backend_name() {
        let addr = spawn_test_server(|_| {});
        let (status, _headers, body) = http_request(&addr, "GET", "/health", None);
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["status"], "ok");
        assert_eq!(v["backend"], "mock");
    }

    #[test]
    fn models_lists_configured_model_id() {
        let addr = spawn_test_server(|cfg| cfg.model_id = "my-model".to_string());
        let (status, _headers, body) = http_request(&addr, "GET", "/v1/models", None);
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "my-model");
        assert_eq!(v["data"][0]["object"], "model");
        assert!(v["data"][0]["created"].as_u64().unwrap() > 0);
    }

    #[test]
    fn unknown_path_is_404() {
        let addr = spawn_test_server(|_| {});
        let (status, _headers, body) = http_request(&addr, "GET", "/nope", None);
        assert_eq!(status, 404);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "not_found");
    }

    #[test]
    fn invalid_sampling_parameters_are_rejected_without_panicking() {
        let addr = spawn_test_server(|_| {});
        for extra in [r#""top_p":0"#, r#""temperature":-1"#, r#""top_k":-1"#, r#""top_k":1.5"#,
                      r#""repeat_penalty":0"#, r#""repetition_penalty":"x""#,
                      r#""reasoning_effort":"max""#, r#""presence_penalty":2.5"#, r#""frequency_penalty":"x""#, r#""reasoning_effort":2"#,
                      r#""chat_template_kwargs":[1]"#] {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{extra}}}"#);
            let (status, _, response) =
                http_request(&addr, "POST", "/v1/chat/completions", Some(&body));
            assert_eq!(status, 400, "{response}");
        }
    }

    #[test]
    fn requests_take_the_model_defaults_and_override_them() {
        struct Record(Arc<Mutex<Vec<(f32, usize, f32, f32)>>>);
        impl Backend for Record {
            fn generate(&mut self, _: &[u32], p: &mut GenParams, emit: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>)
                -> Result<(), String> {
                let s = &p.sampler;
                self.0.lock().unwrap().push((s.temperature, s.top_k, s.top_p, s.repetition_penalty));
                let _ = emit(crate::backend::TokenEvent { id: 1, logits: None });
                Ok(())
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut cfg = ServerConfig::from_env();
        cfg.port = 0;
        cfg.backend_name = "mock".into();
        cfg.defaults = GenDefaults { temperature: 0.1, top_k: 50, top_p: 0.9, repetition_penalty: 1.1, ..GenDefaults::default() };
        let server = bind(&cfg).unwrap();
        let addr = server.server_addr().to_ip().unwrap().to_string();
        let backend = Box::new(Record(seen.clone()));
        std::thread::spawn(move || run(server, backend, cfg));
        for extra in ["", r#","temperature":0.7,"top_k":5,"top_p":0.5,"repeat_penalty":1.3"#,
                      r#","repetition_penalty":1.2"#] {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
            let (status, _, response) = http_request(&addr, "POST", "/v1/chat/completions", Some(&body));
            assert_eq!(status, 200, "{response}");
        }
        assert_eq!(*seen.lock().unwrap(), vec![(0.1, 50, 0.9, 1.1), (0.7, 5, 0.5, 1.3), (0.1, 50, 0.9, 1.2)]);
    }

    #[test]
    fn empty_messages_is_400() {
        let addr = spawn_test_server(|_| {});
        let (status, _headers, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some("{\"messages\":[]}"),
        );
        assert_eq!(status, 400);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn non_streaming_chat_completion_echoes_mock_reply() {
        let addr = spawn_test_server(|_| {});
        let req = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi there"}],
            "max_tokens": 200,
            "stream": false,
        });
        let (status, _headers, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_string()),
        );
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("[mock npu] received"));
        assert!(content.contains("prompt tokens"));
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert!(v["usage"]["completion_tokens"].as_u64().unwrap() > 0);
        assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    }

    #[test]
    fn output_budget_follows_llama_cpp_server() {
        let mut cfg = ServerConfig::from_env();
        cfg.default_max_tokens = 0;
        cfg.max_tokens_cap = 0;
        // Open-ended: everything the context has left (prompt + n - 1 <= capacity).
        assert_eq!(output_budget(None, &cfg, Some(4096), 100), Ok(3997));
        // An explicit request is kept, but never runs past the context.
        assert_eq!(output_budget(Some(200), &cfg, Some(4096), 100), Ok(200));
        assert_eq!(output_budget(Some(9999), &cfg, Some(4096), 100), Ok(3997));
        // A prompt that fills the context exactly still gets one token; one past it is an error.
        assert_eq!(output_budget(None, &cfg, Some(4096), 4096), Ok(1));
        assert!(output_budget(None, &cfg, Some(4096), 4097).is_err());
        // Configured default and cap still apply when set.
        cfg.default_max_tokens = 64;
        cfg.max_tokens_cap = 512;
        assert_eq!(output_budget(None, &cfg, Some(4096), 100), Ok(64));
        assert_eq!(output_budget(Some(1000), &cfg, Some(4096), 100), Ok(512));
        // No fixed context (mock): the fallback bound.
        cfg.default_max_tokens = 0;
        cfg.max_tokens_cap = 0;
        assert_eq!(output_budget(None, &cfg, None, 10), Ok(UNBOUNDED_FALLBACK_TOKENS));
    }

    #[test]
    fn max_tokens_is_capped() {
        let addr = spawn_test_server(|cfg| cfg.max_tokens_cap = 3);
        let req = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999,
        });
        let (status, _headers, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_string()),
        );
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn streaming_chat_completion_emits_sse_chunks_and_done() {
        let addr = spawn_test_server(|_| {});
        let req = json!({
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": true,
            "max_tokens": 200,
        });
        let (status, headers, body) = http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&req.to_string()),
        );
        assert_eq!(status, 200);
        assert!(headers
            .iter()
            .any(|h| h.to_lowercase() == "content-type: text/event-stream"));
        assert!(headers
            .iter()
            .any(|h| h.to_lowercase().contains("transfer-encoding")
                && h.to_lowercase().contains("chunked")));

        let frames: Vec<&str> = body
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .collect();
        assert!(frames.first().unwrap().starts_with("data: "));
        let first: Value = serde_json::from_str(frames[0].trim_start_matches("data: ")).unwrap();
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

        assert_eq!(frames.last().unwrap().trim(), "data: [DONE]");

        // reassemble delta.content across all chunk frames and compare to the
        // non-streaming reply text.
        let mut reassembled = String::new();
        let mut saw_finish = false;
        for f in &frames[1..frames.len() - 1] {
            let v: Value = serde_json::from_str(f.trim_start_matches("data: ")).unwrap();
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                reassembled.push_str(c);
            }
            if !v["choices"][0]["finish_reason"].is_null() {
                saw_finish = true;
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
        }
        assert!(saw_finish);
        assert!(reassembled.contains("[mock npu] received"));
    }
}
