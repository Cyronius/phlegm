use super::super::tests::{http_request, http_request_headers};
use super::*;
use crate::backend::{GenParams, TokenEvent};
use crate::embeddings::{EmbeddingBackend, EmbeddingError, EmbeddingInput, EmbeddingOutput};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc;
use std::thread::JoinHandle;

#[derive(Default)]
struct Events {
    held: AtomicBool,
    log: Mutex<Vec<String>>,
}
struct Resource {
    events: Arc<Events>,
    id: String,
}
impl Resource {
    fn load(events: &Arc<Events>, id: &str) -> Self {
        assert!(
            !events.held.swap(true, Ordering::SeqCst),
            "overlapping fake NPU locks"
        );
        events.log.lock().unwrap().push(format!("load:{id}"));
        Self {
            events: events.clone(),
            id: id.into(),
        }
    }
}
impl Drop for Resource {
    fn drop(&mut self) {
        self.events
            .log
            .lock()
            .unwrap()
            .push(format!("unload:{}", self.id));
        assert!(self.events.held.swap(false, Ordering::SeqCst));
    }
}
struct Chat {
    _resource: Resource,
    gate: Option<Arc<Gate>>,
}
struct Gate {
    started: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}
impl Backend for Chat {
    fn generate(
        &mut self,
        _: &[u32],
        _: &mut GenParams,
        emit: &mut dyn FnMut(TokenEvent<'_>) -> ControlFlow<()>,
    ) -> Result<(), String> {
        if emit(TokenEvent {
            id: b'A' as u32,
            logits: None,
        })
        .is_break()
        {
            return Ok(());
        }
        if let Some(gate) = self.gate.take() {
            gate.started.send(()).unwrap();
            gate.resume
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
        }
        let _ = emit(TokenEvent {
            id: b'B' as u32,
            logits: None,
        });
        Ok(())
    }
}
struct Embed(Resource);
impl EmbeddingBackend for Embed {
    fn model_id(&self) -> &str {
        &self.0.id
    }
    fn embed(&mut self, input: EmbeddingInput) -> Result<EmbeddingOutput, EmbeddingError> {
        Ok(EmbeddingOutput {
            vectors: vec![vec![0.25, 0.75]; input.len()],
            tokens: input.len(),
        })
    }
}
fn config() -> ServerConfig {
    let mut cfg = ServerConfig::from_env();
    cfg.host = "127.0.0.1".into();
    cfg.port = 0;
    cfg.model_dir = None;
    cfg.backend_name = "mock".into();
    cfg.api_key = None;
    cfg.allowed_origins = "*".into();
    cfg.default_max_tokens = 16;
    cfg.max_tokens_cap = 32;
    cfg
}
fn model(id: &str, kind: ModelKind) -> InstalledModel {
    InstalledModel {
        id: id.into(),
        path: PathBuf::from(id),
        kind,
    }
}
fn manager(events: Arc<Events>, gate: Option<Arc<Gate>>, timeout: Duration) -> ModelServer {
    let cfg = config();
    let base = cfg.clone();
    ModelServer::new(
        vec![
            model("a", ModelKind::Chat),
            model("b", ModelKind::Chat),
            model("embed", ModelKind::Embedding),
            model("bad", ModelKind::Chat),
        ],
        cfg,
        timeout,
        None,
        move |m| {
            let resource = Resource::load(&events, &m.id);
            if m.id == "bad" {
                return Err("planted load failure".into());
            }
            let mut cfg = base.clone();
            cfg.model_id = m.id.clone();
            match m.kind {
                ModelKind::Embedding => Ok(LoadedModel::embedding(Box::new(Embed(resource)), cfg)),
                _ => LoadedModel::chat(
                    Box::new(Chat {
                        _resource: resource,
                        gate: if m.id == "a" { gate.clone() } else { None },
                    }),
                    cfg,
                ),
            }
        },
    )
    .unwrap()
}
struct Http {
    addr: String,
    socket: Arc<tiny_http::Server>,
    state: Arc<ModelServer>,
    thread: Option<JoinHandle<()>>,
}
impl Http {
    fn new(state: ModelServer) -> Self {
        let socket = Arc::new(bind(&state.cfg).unwrap());
        let addr = socket.server_addr().to_ip().unwrap().to_string();
        let state = Arc::new(state);
        let worker = state.clone();
        let accept = socket.clone();
        let thread = std::thread::spawn(move || {
            let mut threads = Vec::new();
            while let Ok(request) = accept.recv() {
                let worker = worker.clone();
                threads.push(std::thread::spawn(move || worker.handle(request)));
            }
            for t in threads {
                t.join().unwrap();
            }
        });
        Self {
            addr,
            socket,
            state,
            thread: Some(thread),
        }
    }
    fn chat(&self, id: &str, stream: bool) -> (u16, Vec<String>, String) {
        http_request(
            &self.addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body(id, stream)),
        )
    }
}
impl Drop for Http {
    fn drop(&mut self) {
        self.socket.unblock();
        self.thread.take().unwrap().join().unwrap();
    }
}
fn chat_body(id: &str, stream: bool) -> String {
    json!({"model":id,"messages":[{"role":"user","content":"hi"}],"stream":stream}).to_string()
}
fn logs(events: &Events) -> Vec<String> {
    events.log.lock().unwrap().clone()
}
fn payload(http: &Http) -> Value {
    let (status, _, body) = http_request(&http.addr, "GET", "/v1/models", None);
    assert_eq!(status, 200);
    serde_json::from_str(&body).unwrap()
}

#[test]
fn switches_two_chats_and_embedder_reuses_and_unloads_before_loading() {
    let events = Arc::new(Events::default());
    {
        let http = Http::new(manager(events.clone(), None, Duration::from_secs(2)));
        assert!(payload(&http)["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["status"] == "unloaded"));
        assert!(logs(&events).is_empty());
        assert_eq!(http.chat("a", false).0, 200);
        assert_eq!(http.chat("a", true).0, 200);
        let p = payload(&http);
        assert_eq!(p["data"].as_array().unwrap().len(), 4);
        assert_eq!(p["data"][0]["id"], "a");
        assert_eq!(p["data"][0]["loaded"], true);
        assert_eq!(http.chat("b", false).0, 200);
        let (status, _, body) = http_request(
            &http.addr,
            "POST",
            "/v1/embeddings",
            Some(r#"{"model":"embed","input":["one","two"]}"#),
        );
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["data"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(http.chat("a", false).0, 200);
        assert_eq!(
            logs(&events),
            [
                "load:a",
                "unload:a",
                "load:b",
                "unload:b",
                "load:embed",
                "unload:embed",
                "load:a"
            ]
        );
    }
    assert!(!events.held.load(Ordering::SeqCst));
    assert_eq!(logs(&events).last().unwrap(), "unload:a");
}

#[test]
fn failed_load_frees_lock_and_unknown_or_wrong_kind_do_not_swap() {
    let events = Arc::new(Events::default());
    let http = Http::new(manager(events.clone(), None, Duration::from_secs(2)));
    assert_eq!(http.chat("a", false).0, 200);
    assert_eq!(http.chat("absent", false).0, 404);
    assert_eq!(http.chat("embed", false).0, 400);
    assert_eq!(logs(&events), ["load:a"]);
    let failed = http.chat("bad", false);
    assert_eq!(failed.0, 503);
    assert!(failed.2.contains("planted load failure"));
    assert!(!events.held.load(Ordering::SeqCst));
    assert!(payload(&http)["data"]
        .as_array()
        .unwrap()
        .iter()
        .all(|m| m["status"] == "unloaded"));
    assert_eq!(http.chat("b", false).0, 200);
    assert_eq!(
        logs(&events),
        ["load:a", "unload:a", "load:bad", "unload:bad", "load:b"]
    );
}

#[test]
fn stream_keeps_lease_while_different_model_waits_and_discovery_stays_live() {
    let events = Arc::new(Events::default());
    let (started, rx) = mpsc::channel();
    let (resume, released) = mpsc::channel();
    let gate = Arc::new(Gate {
        started,
        resume: Mutex::new(released),
    });
    let http = Http::new(manager(events.clone(), Some(gate), Duration::from_secs(5)));
    let addr = http.addr.clone();
    let first = std::thread::spawn(move || {
        http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body("a", true)),
        )
    });
    rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let addr = http.addr.clone();
    let (done, result) = mpsc::channel();
    let second = std::thread::spawn(move || {
        done.send(http_request(
            &addr,
            "POST",
            "/v1/embeddings",
            Some(r#"{"model":"embed","input":"hi"}"#),
        ))
        .unwrap();
    });
    assert!(result.recv_timeout(Duration::from_millis(100)).is_err());
    assert_eq!(logs(&events), ["load:a"]);
    assert_eq!(payload(&http)["data"][0]["status"], "loaded");
    assert_eq!(http_request(&http.addr, "GET", "/health", None).0, 200);
    resume.send(()).unwrap();
    let stream = first.join().unwrap();
    assert_eq!(stream.0, 200);
    assert!(stream.2.contains("data: [DONE]"));
    assert!(stream.2.contains("\"content\":\"A\""));
    assert!(stream.2.contains("\"content\":\"B\""));
    assert_eq!(result.recv_timeout(Duration::from_secs(3)).unwrap().0, 200);
    second.join().unwrap();
    assert_eq!(logs(&events), ["load:a", "unload:a", "load:embed"]);
}

#[test]
fn queue_timeout_does_not_interrupt_current_request() {
    let events = Arc::new(Events::default());
    let (started, rx) = mpsc::channel();
    let (resume, released) = mpsc::channel();
    let http = Http::new(manager(
        events.clone(),
        Some(Arc::new(Gate {
            started,
            resume: Mutex::new(released),
        })),
        Duration::from_millis(20),
    ));
    let addr = http.addr.clone();
    let active = std::thread::spawn(move || {
        http_request(
            &addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body("a", true)),
        )
    });
    rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let failed = http.chat("b", false);
    assert_eq!(failed.0, 503);
    assert!(failed.2.contains("model queue timeout"));
    assert_eq!(logs(&events), ["load:a"]);
    resume.send(()).unwrap();
    assert_eq!(active.join().unwrap().0, 200);
    assert_eq!(http.chat("b", false).0, 200);
}

#[test]
fn idle_unload_is_optional_and_authentication_precedes_loading() {
    let events = Arc::new(Events::default());
    let mut manager = manager(events.clone(), None, Duration::from_secs(1));
    manager.cfg.api_key = Some("secret".into());
    let http = Http::new(manager);
    assert_eq!(http.chat("a", false).0, 401);
    assert!(logs(&events).is_empty());
    assert_eq!(http_request(&http.addr, "GET", "/health", None).0, 200);
    let (status, headers, _) = http_request_headers(
        &http.addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat_body("a", false)),
        "Authorization: Bearer secret\r\n",
    );
    assert_eq!(status, 200);
    assert!(headers.iter().any(|h| h
        .to_ascii_lowercase()
        .contains("access-control-allow-origin: *")));
    http.state.slot.expire(Duration::from_secs(3600));
    assert!(events.held.load(Ordering::SeqCst));
    http.state.slot.expire(Duration::ZERO);
    assert!(!events.held.load(Ordering::SeqCst));
    assert_eq!(logs(&events), ["load:a", "unload:a"]);
}

#[test]
fn loading_status_is_observable_and_failure_is_retryable() {
    let slot = model_slot::Slot::<Resource>::new();
    let events = Arc::new(Events::default());
    let mut lease = slot.acquire(Duration::ZERO).unwrap();
    lease
        .select("a", || {
            assert_eq!(slot.status(), Some(("a".into(), "loading")));
            Ok(Resource::load(&events, "a"))
        })
        .unwrap();
    assert_eq!(slot.status(), Some(("a".into(), "loaded")));
    // Expiry never unloads a leased model.
    slot.expire(Duration::ZERO);
    assert!(events.held.load(Ordering::SeqCst));
    drop(lease);
    slot.expire(Duration::ZERO);
    assert!(!events.held.load(Ordering::SeqCst));
}

#[test]
fn discovers_registry_and_unregistered_chat_embedder_and_image_configs_without_loading() {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "phlegm-u7-catalog-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    for (id, files) in [
        ("KAT-Coder-V2.5-Dev-Q4K-NPU2", vec![("model.q4nx", "")]),
        ("unregistered", vec![("published-Q4_K_M.gguf", "")]),
        (
            "embed",
            vec![
                ("config.json", r#"{"architecture":"TritEmbed"}"#),
                ("model.safetensors", ""),
            ],
        ),
        ("not-installed", vec![("tokenizer.json", "{}")]),
        (
            "pictures",
            vec![(
                "image_config.json",
                r#"{"backend":"flux2","model_id":"black-forest-labs/FLUX.2-klein-4B"}"#,
            )],
        ),
    ] {
        std::fs::create_dir_all(root.join(id)).unwrap();
        for (name, contents) in files {
            std::fs::write(root.join(id).join(name), contents).unwrap();
        }
    }
    let registry = crate::settings::Config::with_user(json!({}), root.join("settings.json"));
    let catalog = discover(&root, &registry, &[root.join("pictures/image_config.json")]).unwrap();
    assert_eq!(catalog.len(), 4);
    assert!(catalog
        .iter()
        .any(|m| m.id == "embed" && m.kind == ModelKind::Embedding));
    assert!(catalog
        .iter()
        .any(|m| m.id == "unregistered" && m.kind == ModelKind::Chat));
    assert!(catalog
        .iter()
        .any(|m| m.id == crate::image::FLUX2_KLEIN_IDS[0] && m.kind == ModelKind::Image));
    std::fs::write(
        root.join("phlegm-duplicate-image.json"),
        r#"{"backend":"flux2","model_id":"black-forest-labs/FLUX.2-klein-4B"}"#,
    )
    .unwrap();
    assert!(discover(&root, &registry, &[])
        .unwrap_err()
        .contains("duplicate installed model"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn image_routes_reuse_native_validation_and_support_multipart_model_selection() {
    let cfg = config();
    let image = ModelServer::new(
        vec![model(crate::image::FLUX2_KLEIN_IDS[0], ModelKind::Image)],
        cfg,
        Duration::from_secs(1),
        None,
        |_| {
            Ok(LoadedModel::image(ImageConfig::Flux2(
                crate::image::Flux2ImageConfig {
                    backend: "flux2".into(),
                    model_id: crate::image::FLUX2_KLEIN_IDS[0].into(),
                    dibs: None,
                    python: "/missing".into(),
                    repo_root: "/missing".into(),
                    env_script: None,
                    model: "/missing".into(),
                    kernels: "/missing".into(),
                    output_root: "/missing".into(),
                },
            )))
        },
    )
    .unwrap();
    let http = Http::new(image);
    let body = json!({"model":crate::image::FLUX2_KLEIN_IDS[0],"prompt":"fox","n":2}).to_string();
    assert_eq!(
        http_request(&http.addr, "POST", "/v1/images/generations", Some(&body)).0,
        400
    );
    let body =
        json!({"model":crate::image::FLUX2_KLEIN_IDS[0],"prompt":"fox","image":"not base64"})
            .to_string();
    assert_eq!(
        http_request(&http.addr, "POST", "/v1/images/edits", Some(&body)).0,
        400
    );
    let body = format!(
        "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{}\r\n--b--\r\n",
        crate::image::FLUX2_KLEIN_IDS[0]
    );
    assert_eq!(
        crate::image::request_model("multipart/form-data; boundary=b", body.as_bytes())
            .unwrap()
            .unwrap(),
        crate::image::FLUX2_KLEIN_IDS[0]
    );
    assert!(payload(&http)["data"][0]["capabilities"] == json!(["image"]));
}
