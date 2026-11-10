//! Installed-model catalog and lazy, exclusive serving. No device is opened by discovery.
use super::*;
use crate::image::ImageConfig;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Chat,
    Embedding,
    Image,
}
impl ModelKind {
    fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Embedding => "embedding",
            Self::Image => "image",
        }
    }
}
#[derive(Clone, Debug)]
pub struct InstalledModel {
    pub id: String,
    /// Model directory for chat/embedding; configuration JSON for images.
    pub path: PathBuf,
    pub kind: ModelKind,
}

/// Registry entries supply model names/settings, not filesystem locations. Search the
/// explicitly selected root (or ~/models) and its immediate child directories only.
pub fn default_models_dir() -> PathBuf {
    std::env::var_os("PHLEGM_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                std::env::var_os("HOME")
                    .or_else(|| std::env::var_os("USERPROFILE"))
                    .unwrap_or_default(),
            )
            .join("models")
        })
}
fn image_id(path: &Path) -> Result<String, String> {
    let v: Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let id = match v.get("backend").and_then(Value::as_str) {
        Some("flux2") => v
            .get("model_id")
            .and_then(Value::as_str)
            .ok_or("flux2 config needs model_id")?,
        Some("flux3") => crate::image::FLUX3_ID,
        Some("qwen21") => crate::image::QWEN21_ID,
        None if v.get("dit_weights").is_some() => crate::image::MOSS_ID,
        _ => {
            return Err(format!(
                "unsupported image configuration {}",
                path.display()
            ))
        }
    };
    Ok(id.to_owned())
}
fn insert(
    models: &mut BTreeMap<String, InstalledModel>,
    model: InstalledModel,
) -> Result<(), String> {
    if let Some(old) = models.get(&model.id) {
        if old.path == model.path && old.kind == model.kind {
            return Ok(());
        }
        return Err(format!(
            "duplicate installed model {}: {} and {}",
            model.id,
            old.path.display(),
            model.path.display()
        ));
    }
    models.insert(model.id.clone(), model);
    Ok(())
}
pub fn discover(
    root: &Path,
    registry: &crate::settings::Config,
    image_configs: &[PathBuf],
) -> Result<Vec<InstalledModel>, String> {
    let mut models = BTreeMap::new();
    let root = std::fs::canonicalize(root)
        .map_err(|e| format!("models directory {}: {e}", root.display()))?;
    let mut dirs = vec![root.clone()];
    for entry in std::fs::read_dir(&root).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    let registry_names: HashSet<String> = registry
        .tags()
        .into_iter()
        .filter_map(|(_, entry)| entry["name"].as_str().map(str::to_owned))
        .collect();
    for dir in &dirs {
        let id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("non-UTF8 model directory")?
            .to_owned();
        let config = std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
        let embed = config
            .as_ref()
            .is_some_and(|c| c["architecture"] == "TritEmbed");
        let gguf = std::fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .any(|entry| {
                entry.is_ok_and(|e| {
                    e.path().is_file()
                        && e.path()
                            .extension()
                            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
                })
            });
        let weights = gguf || dir.join("model.q4nx").is_file();
        if embed && dir.join("model.safetensors").is_file() {
            insert(
                &mut models,
                InstalledModel {
                    id,
                    path: dir.clone(),
                    kind: ModelKind::Embedding,
                },
            )?;
        } else if weights
            && (registry_names.contains(&id) || dir.join("tokenizer.json").is_file() || gguf)
        {
            insert(
                &mut models,
                InstalledModel {
                    id,
                    path: dir.clone(),
                    kind: ModelKind::Chat,
                },
            )?;
        }
        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if path.is_file()
                && (name == "image_config.json"
                    || (name.starts_with("phlegm-") && name.ends_with("-image.json")))
            {
                let path = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
                insert(
                    &mut models,
                    InstalledModel {
                        id: image_id(&path)?,
                        path,
                        kind: ModelKind::Image,
                    },
                )?;
            }
        }
    }
    for path in image_configs {
        let path = std::fs::canonicalize(path)
            .map_err(|e| format!("image config {}: {e}", path.display()))?;
        insert(
            &mut models,
            InstalledModel {
                id: image_id(&path)?,
                path,
                kind: ModelKind::Image,
            },
        )?;
    }
    Ok(models.into_values().collect())
}

pub struct LoadedModel(Loaded);
enum Loaded {
    Inference(Arc<ServerState>),
    Image(ImageConfig),
}
impl LoadedModel {
    pub fn chat(backend: Box<dyn Backend + Send>, cfg: ServerConfig) -> Result<Self, String> {
        let tokenizer = ServerTokenizer::build(&cfg)?;
        let context_capacity = backend.context_capacity();
        Ok(Self(Loaded::Inference(Arc::new(ServerState {
            backend: Mutex::new(backend),
            tokenizer,
            context_capacity,
            cfg,
            embedder: None,
            embedding_model: None,
            embedding_only: false,
        }))))
    }
    pub fn embedding(
        embedder: Box<dyn crate::embeddings::EmbeddingBackend>,
        cfg: ServerConfig,
    ) -> Self {
        let embedding_model = Some(embedder.model_id().to_owned());
        Self(Loaded::Inference(Arc::new(ServerState {
            backend: Mutex::new(Box::new(crate::mock_backend::MockBackend::new())),
            tokenizer: ServerTokenizer::Placeholder(PlaceholderTokenizer::new()),
            context_capacity: None,
            cfg,
            embedder: Some(Mutex::new(embedder)),
            embedding_model,
            embedding_only: true,
        })))
    }
    pub fn image(cfg: ImageConfig) -> Self {
        Self(Loaded::Image(cfg))
    }
}

pub struct ModelServer {
    models: BTreeMap<String, InstalledModel>,
    loader: Box<dyn Fn(&InstalledModel) -> Result<LoadedModel, String> + Send + Sync>,
    slot: model_slot::Slot<LoadedModel>,
    cfg: ServerConfig,
    queue_timeout: Duration,
    keep_alive: Option<Duration>,
}
impl ModelServer {
    pub fn new(
        models: Vec<InstalledModel>,
        cfg: ServerConfig,
        queue_timeout: Duration,
        keep_alive: Option<Duration>,
        loader: impl Fn(&InstalledModel) -> Result<LoadedModel, String> + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let mut catalog = BTreeMap::new();
        for model in models {
            insert(&mut catalog, model)?;
        }
        Ok(Self {
            models: catalog,
            loader: Box::new(loader),
            slot: model_slot::Slot::new(),
            cfg,
            queue_timeout,
            keep_alive,
        })
    }
    fn payload(&self) -> Value {
        let active = self.slot.status();
        json!({"object":"list", "data": self.models.values().map(|m| {
            let status = active.as_ref().filter(|(id, _)| id == &m.id).map_or("unloaded", |(_, status)| *status);
            json!({"id":m.id,"object":"model","created":0,"owned_by":"phlegm","capabilities":[m.kind.name()],"loaded":status == "loaded","status":status})
        }).collect::<Vec<_>>()})
    }
    pub fn handle(&self, mut request: Request) {
        if !authorize(&request, &self.cfg) {
            return respond_json(
                request,
                401,
                &json!({"error":{"message":"Invalid API key","type":"authentication_error"}}),
            );
        }
        let path = request.url().to_owned();
        match (request.method(), path.as_str()) {
            (Method::Options, _) => return respond_json(request, 200, &json!({})),
            (Method::Get, "/health") => {
                return respond_json(
                    request,
                    200,
                    &json!({"status":"ok","backend":"multi-model"}),
                )
            }
            (Method::Get, "/v1/models") => return respond_json(request, 200, &self.payload()),
            (Method::Get, "/") => return crate::server::respond_chat_page(request),
            _ => {}
        }
        let kind = match (request.method(), path.as_str()) {
            (Method::Post, "/v1/chat/completions" | "/v1/completions") => ModelKind::Chat,
            (Method::Post, "/v1/embeddings") => ModelKind::Embedding,
            (Method::Post, "/v1/images/generations" | "/v1/images/edits") => ModelKind::Image,
            _ => return error(request, 404, "unknown path"),
        };
        let content_type = request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Content-Type"))
            .map(|h| h.value.as_str().to_owned())
            .unwrap_or_default();
        let limit = match (kind, path.as_str()) {
            (ModelKind::Image, "/v1/images/edits") => 32 * 1024 * 1024,
            (ModelKind::Image, _) => 64 * 1024,
            _ => 16_000_000,
        };
        if request.body_length().is_some_and(|n| n as u64 > limit) {
            return error(request, 413, "request body is too large");
        }
        let mut body = Vec::new();
        if request
            .as_reader()
            .take(limit + 1)
            .read_to_end(&mut body)
            .is_err()
            || body.len() as u64 > limit
        {
            return error(request, 413, "invalid or oversized request body");
        }
        let model = match crate::image::request_model(&content_type, &body) {
            Ok(Some(id)) => id,
            Ok(None) if kind == ModelKind::Image && self.models.len() == 1 => {
                self.models.keys().next().unwrap().clone()
            }
            Ok(None) => return error(request, 400, "model is required in multi-model mode"),
            Err(e) => return error(request, 400, &e),
        };
        let Some(installed) = self.models.get(&model) else {
            return error(request, 404, "model not found");
        };
        if installed.kind != kind {
            return error(request, 400, "model does not support this endpoint");
        }
        let mut lease = match self.slot.acquire(self.queue_timeout) {
            Ok(l) => l,
            Err(e) => return error(request, 503, &e),
        };
        if let Err(e) = lease.select(&model, || (self.loader)(installed)) {
            return error(request, 503, &format!("model load failed: {e}"));
        }
        let (_, loaded) = lease.resident.as_ref().unwrap();
        match &loaded.0 {
            Loaded::Inference(state) if kind == ModelKind::Chat => {
                handle_chat_body(request, state, path == "/v1/completions", &body)
            }
            Loaded::Inference(state) if kind == ModelKind::Embedding => {
                embeddings_route::handle_body(request, state, &body)
            }
            Loaded::Image(cfg) if kind == ModelKind::Image => crate::image::handle_body(
                request,
                cfg,
                &body,
                &content_type,
                path == "/v1/images/edits",
            ),
            _ => error(request, 500, "loader returned an incompatible model"),
        }
        // lease deliberately spans the final SSE frame / image child exit / worker reply.
    }
    /// Bounded concurrent handlers let discovery/health remain responsive during a load
    /// or generation. Queued handlers wait on the same lease, with no second backend.
    pub fn run(self, server: tiny_http::Server) -> Result<(), String> {
        let state = Arc::new(self);
        let pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        loop {
            if let Some(keep_alive) = state.keep_alive {
                state.slot.expire(keep_alive);
            }
            let request = match server.recv_timeout(Duration::from_millis(250)) {
                Ok(Some(r)) => r,
                Ok(None) => continue,
                Err(e) => return Err(e.to_string()),
            };
            if request.method() != &Method::Post {
                state.handle(request);
                continue;
            }
            if pending.load(Ordering::SeqCst) >= 64 {
                // Set CORS and authenticate even on overload responses.
                if !authorize(&request, &state.cfg) {
                    error(request, 401, "Invalid API key");
                } else {
                    error(request, 503, "model request queue is full");
                }
                continue;
            }
            pending.fetch_add(1, Ordering::SeqCst);
            let state = state.clone();
            let pending = pending.clone();
            std::thread::spawn(move || {
                struct Pending(Arc<std::sync::atomic::AtomicUsize>);
                impl Drop for Pending {
                    fn drop(&mut self) {
                        self.0.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                let _pending = Pending(pending);
                state.handle(request);
            });
        }
    }
}
fn error(request: Request, status: u16, message: &str) {
    let kind = match status {
        401 => "authentication_error",
        404 => "not_found",
        503 => "service_unavailable",
        500 => "internal_error",
        _ => "invalid_request_error",
    };
    respond_json(
        request,
        status,
        &json!({"error":{"message":message,"type":kind}}),
    );
}

#[cfg(test)]
#[path = "server_models_tests.rs"]
mod tests;
