//! OpenAI-compatible HTTP server, ported from `tools/server/app.py` +
//! `tools/server/config.py`.
//!
//! Endpoints:
//!   GET  /health                 -> {"status":"ok","backend":...}
//!   GET  /v1/models              -> OpenAI model list
//!   POST /v1/chat/completions    -> chat completion; stream=true gives SSE deltas
//!
//! Backend-agnostic: `serve()` takes any `Box<dyn Backend + Send>` (a
//! `NpuBackend` for normal serving, `MockBackend` for explicit tests).
//! Generation is serialized behind a `Mutex` around the backend, mirroring
//! Python's per-backend lock ("the NPU is a single shared device").
//!
//! Threading model: a single accept loop calls `Server::recv()` and handles
//! each request to completion (including writing the full SSE stream) before
//! looping back for the next one. tiny_http's own accept thread still queues
//! new connections concurrently (see `Server::from_listener`'s internal
//! `TaskPool`), so this isn't "one connection at a time on the wire" — it's
//! "one *response body* being produced at a time", which is what the shared
//! backend lock would force anyway.

use crate::backend::{Backend, GenParams};
use crate::placeholder_tokenizer::PlaceholderTokenizer;
use crate::sampler::Sampler;
use crate::tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Request, Response};

// --------------------------------------------------------------------------
// Config
// --------------------------------------------------------------------------

/// Mirrors `tools/server/config.py`: plain fields with env-var overrides.
#[derive(Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub model_id: String,
    pub default_max_tokens: usize,
    pub max_tokens_cap: usize,
    /// Directory containing `tokenizer.json` / `tokenizer_config.json` /
    /// `chat_template.jinja` (or an inline template in tokenizer_config.json).
    /// Required for an NPU backend; mock mode may use placeholder tokens.
    pub model_dir: Option<PathBuf>,
    /// Reported verbatim in `GET /health`; selected by the CLI after load.
    pub backend_name: String,
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
            default_max_tokens: env_or("PHLEGM_MAX_TOKENS", "64").parse().unwrap_or(64),
            max_tokens_cap: env_or("PHLEGM_MAX_TOKENS_CAP", "512")
                .parse()
                .unwrap_or(512),
            model_dir: std::env::var("PHLEGM_MODEL_DIR").ok().map(PathBuf::from),
            backend_name: env_or("PHLEGM_BACKEND", "mock"),
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
            Self::Placeholder(t) => {
                let mut ids = Vec::new();
                let mut emitted = String::new();
                Box::new(move |id| {
                    ids.push(id);
                    let full = t.decode(&ids);
                    let piece = full
                        .strip_prefix(&emitted)
                        .ok_or("nonmonotonic placeholder decode")?
                        .to_owned();
                    emitted = full;
                    Ok(Some(piece))
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

    fn decode(&self, ids: &[u32]) -> String {
        match self {
            ServerTokenizer::Real(t) => t.decode(ids).unwrap_or_else(|e| {
                eprintln!("server: tokenizer decode error: {e}");
                String::new()
            }),
            ServerTokenizer::Placeholder(t) => t.decode(ids),
        }
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> String {
        match self {
            // add_generation_prompt=true (we always want a completion to
            // follow); enable_thinking=true (Qwen3.6's own default).
            ServerTokenizer::Real(t) => {
                t.apply_chat_template(messages, true, true)
                    .unwrap_or_else(|e| {
                        eprintln!("server: chat template render error: {e}");
                        String::new()
                    })
            }
            ServerTokenizer::Placeholder(t) => t.apply_chat_template(messages),
        }
    }

    fn stop_ids(&self) -> HashSet<u32> {
        match self {
            ServerTokenizer::Real(t) => t.stop_ids(),
            ServerTokenizer::Placeholder(t) => t.eos_id().into_iter().collect(),
        }
    }
}

// --------------------------------------------------------------------------
// Server state / entry points
// --------------------------------------------------------------------------

struct ServerState {
    backend: Mutex<Box<dyn Backend + Send>>,
    tokenizer: ServerTokenizer,
    cfg: ServerConfig,
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
    let tokenizer = ServerTokenizer::build(&cfg)?;
    let backend_name = cfg.backend_name.clone();
    let model_id = cfg.model_id.clone();
    let state = Arc::new(ServerState {
        backend: Mutex::new(backend),
        tokenizer,
        cfg,
    });

    println!(
        "phlegm server on http://{}:{}",
        state.cfg.host, state.cfg.port
    );
    println!("  backend: {backend_name}   model: {model_id}");
    println!("  POST /v1/chat/completions | GET /v1/models | GET /health");

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

/// Bind + run in one call — the entry point `phlegm serve` calls.
pub fn serve(backend: Box<dyn Backend + Send>, cfg: ServerConfig) -> Result<(), String> {
    let server = bind(&cfg)?;
    run(server, backend, cfg)
}

// --------------------------------------------------------------------------
// Routing
// --------------------------------------------------------------------------

fn handle_request(request: Request, state: &Arc<ServerState>) {
    let method = request.method().clone();
    let url = request.url().to_string();
    match (method, url.as_str()) {
        (Method::Get, "/health") => {
            respond_json(
                request,
                200,
                &json!({"status": "ok", "backend": state.cfg.backend_name}),
            );
        }
        (Method::Get, "/v1/models") => {
            respond_json(request, 200, &models_payload(&state.cfg));
        }
        (Method::Post, "/v1/chat/completions") => {
            handle_chat_completions(request, state);
        }
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

fn models_payload(cfg: &ServerConfig) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": cfg.model_id,
            "object": "model",
            "created": now_unix(),
            "owned_by": "fastflowlm-open-npu",
        }],
    })
}

// --------------------------------------------------------------------------
// /v1/chat/completions
// --------------------------------------------------------------------------

fn handle_chat_completions(mut request: Request, state: &Arc<ServerState>) {
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
    let req_json: Value = if body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
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

    let messages_val = req_json.get("messages").and_then(Value::as_array);
    let messages_val = match messages_val {
        Some(arr) if !arr.is_empty() => arr,
        _ => {
            respond_json(
                request,
                400,
                &json!({"error": {"message": "'messages' must be a non-empty array", "type": "invalid_request_error"}}),
            );
            return;
        }
    };
    let messages: Vec<ChatMessage> = messages_val.iter().map(parse_message).collect();

    let model = req_json
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&state.cfg.model_id)
        .to_string();
    let stream = req_json
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max_tokens = req_json
        .get("max_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(state.cfg.default_max_tokens);
    let max_tokens = max_tokens.max(1).min(state.cfg.max_tokens_cap);
    let temperature = req_json
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let top_p = req_json
        .get("top_p")
        .and_then(Value::as_f64)
        .map(|v| v as f32);
    let seed = req_json.get("seed").and_then(Value::as_u64);

    if temperature.is_some_and(|v| !v.is_finite() || v < 0.0)
        || top_p.is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
    {
        respond_json(
            request,
            400,
            &json!({"error": {"message": "temperature must be nonnegative and top_p must be in (0, 1]", "type": "invalid_request_error"}}),
        );
        return;
    }

    let prompt = state.tokenizer.apply_chat_template(&messages);
    let prompt_ids = state.tokenizer.encode(&prompt);
    let stop_ids = state.tokenizer.stop_ids();

    let sampler = Sampler::new(
        temperature.unwrap_or(0.0),
        0,
        top_p.unwrap_or(1.0),
        1.0,
        seed,
    );
    let params = GenParams {
        max_tokens,
        sampler,
        stop_ids,
    };

    if stream {
        stream_response(request, state, prompt_ids, params, model);
    } else {
        full_response(request, state, prompt_ids, params, model);
    }
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
    ChatMessage { role, content }
}

fn full_response(
    request: Request,
    state: &Arc<ServerState>,
    prompt_ids: Vec<u32>,
    mut params: GenParams,
    model: String,
) {
    let max_tokens = params.max_tokens;
    let mut out_ids: Vec<u32> = Vec::new();
    let gen_result = {
        let mut backend = state.backend.lock().unwrap();
        let mut on_token = |tid: u32| out_ids.push(tid);
        backend.generate(&prompt_ids, &mut params, &mut on_token)
    };
    if let Err(e) = gen_result {
        let msg = format!("generation error: {e}");
        respond_json(
            request,
            500,
            &json!({"error": {"message": msg, "type": "internal_error"}}),
        );
        return;
    }

    let text = state.tokenizer.decode(&out_ids);
    let finish = if out_ids.len() >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let resp = json!({
        "id": cmpl_id(),
        "object": "chat.completion",
        "created": now_unix(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_ids.len(),
            "completion_tokens": out_ids.len(),
            "total_tokens": prompt_ids.len() + out_ids.len(),
        },
    });
    respond_json(request, 200, &resp);
}

fn stream_response(
    request: Request,
    state: &Arc<ServerState>,
    prompt_ids: Vec<u32>,
    mut params: GenParams,
    model: String,
) {
    let max_tokens = params.max_tokens;
    let cid = cmpl_id();
    let created = now_unix();

    let mut writer = request.into_writer();
    if write_sse_headers(&mut *writer).is_err() {
        return;
    }
    let _ = sse_write(
        &mut *writer,
        &sse_frame(&sse_chunk(
            &cid,
            created,
            &model,
            json!({"role": "assistant"}),
            None,
        )),
    );

    let mut out_ids: Vec<u32> = Vec::new();
    let mut emitted = String::new();
    let mut decode_error = None;
    let mut decoder = state.tokenizer.decode_stream();
    let gen_result = {
        let mut backend = state.backend.lock().unwrap();
        let mut on_token = |tid: u32| {
            out_ids.push(tid);
            if decode_error.is_some() {
                return;
            }
            let piece = match decoder(tid) {
                Ok(Some(piece)) => piece,
                Ok(None) => return,
                Err(e) => {
                    decode_error = Some(e);
                    return;
                }
            };
            emitted.push_str(&piece);
            if !piece.is_empty() {
                let _ = sse_write(
                    &mut *writer,
                    &sse_frame(&sse_chunk(
                        &cid,
                        created,
                        &model,
                        json!({"content": piece}),
                        None,
                    )),
                );
            }
        };
        backend.generate(&prompt_ids, &mut params, &mut on_token)
    };

    let result = gen_result.and_then(|_| {
        if let Some(error) = decode_error {
            return Err(error);
        }
        let full = state.tokenizer.final_decode(&out_ids)?;
        let tail = final_stream_tail(&full, &emitted)?;
        if !tail.is_empty() {
            sse_write(
                &mut *writer,
                &sse_frame(&sse_chunk(
                    &cid,
                    created,
                    &model,
                    json!({"content": tail}),
                    None,
                )),
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    });
    if let Err(e) = result {
        let _ = sse_write(
            &mut *writer,
            &sse_frame(&json!({"error": {"message": e, "type": "internal_error"}})),
        );
        let _ = sse_write(&mut *writer, "data: [DONE]\n\n");
        let _ = sse_end(&mut *writer);
        let _ = writer.flush();
        return;
    }

    let finish = if out_ids.len() >= max_tokens {
        "length"
    } else {
        "stop"
    };
    let _ = sse_write(
        &mut *writer,
        &sse_frame(&sse_chunk(&cid, created, &model, json!({}), Some(finish))),
    );
    let _ = sse_write(&mut *writer, "data: [DONE]\n\n");
    let _ = sse_end(&mut *writer);
    let _ = writer.flush();
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

fn sse_chunk(cid: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
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

fn respond_json(request: Request, status: u16, value: &Value) {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    let response = Response::from_data(body)
        .with_status_code(status)
        .with_header(json_content_type());
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
                emit: &mut dyn FnMut(u32),
            ) -> Result<(), String> {
                for &id in &self.0 {
                    emit(id);
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
            };
            let server = bind(&cfg).unwrap();
            let addr = server.server_addr().to_ip().unwrap().to_string();
            let state = Arc::new(ServerState {
                backend: Mutex::new(Box::new(Tokens(ids))),
                tokenizer: ServerTokenizer::Real(tokenizer),
                cfg,
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
    fn http_request(
        addr: &str,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> (u16, Vec<String>, String) {
        let mut stream = TcpStream::connect(addr).expect("connect");
        let body_bytes = body.unwrap_or("").as_bytes();
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
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
        for extra in [r#""top_p":0"#, r#""temperature":-1"#] {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"hi"}}],{extra}}}"#);
            let (status, _, response) =
                http_request(&addr, "POST", "/v1/chat/completions", Some(&body));
            assert_eq!(status, 400, "{response}");
        }
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
