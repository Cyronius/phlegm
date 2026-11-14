//! CLI wiring for lazy multimodal serving; settings and bridge selection stay per model.
use super::*;
use phlegm_core::server::models::{self, InstalledModel, LoadedModel, ModelKind, ModelServer};

fn embedding_options(opts: &HashMap<String, String>) -> (PathBuf, PathBuf) {
    let python = opts
        .get("--embed-python")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("PHLEGM_EMBED_PYTHON").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("python3"));
    let source = opts
        .get("--embed-source")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("PHLEGM_EMBED_SOURCE").map(PathBuf::from))
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../open_kernels/designs/trit_embed")
        });
    (python, source)
}

pub(super) fn serve(opts: HashMap<String, String>, cfg: ServerConfig) -> Result<(), String> {
    if ["--kernels", "--config-model"]
        .iter()
        .any(|key| opts.contains_key(*key))
    {
        return Err("--kernels and --config-model require --model-dir; discovered models select their own kernels and settings".into());
    }
    let root = opts
        .get("--models-dir")
        .map(PathBuf::from)
        .unwrap_or_else(models::default_models_dir);
    let images: Vec<PathBuf> = opts
        .get("--image-config")
        .map(|v| v.split(',').map(PathBuf::from).collect())
        .unwrap_or_default();
    let registry = Config::load()?;
    for warning in &registry.warnings {
        eprintln!("phlegm: {warning}");
    }
    let catalog = models::discover(&root, &registry, &images)?;
    start(catalog, opts, cfg)
}

pub(super) fn start(
    catalog: Vec<InstalledModel>,
    opts: HashMap<String, String>,
    mut cfg: ServerConfig,
) -> Result<(), String> {
    let seconds = |key: &str| -> Result<Option<Duration>, String> {
        opts.get(key)
            .map(|v| {
                v.parse::<u64>()
                    .map(Duration::from_secs)
                    .map_err(|_| format!("{key} must be a nonnegative number of seconds"))
            })
            .transpose()
    };
    let timeout = seconds("--queue-timeout")?.unwrap_or(Duration::from_secs(600));
    let keep_alive = seconds("--keep-alive")?;
    cfg.backend_name = "multi-model".into();
    cfg.model_dir = None;
    let base = cfg.clone();
    let (python, source) = embedding_options(&opts);
    let server = ModelServer::new(catalog, cfg.clone(), timeout, keep_alive, move |model| {
        let mut cfg = base.clone();
        cfg.model_id = model.id.clone();
        match model.kind {
            ModelKind::Chat => {
                let mut opts = opts.clone();
                opts.insert(
                    "--model-dir".into(),
                    model.path.to_string_lossy().into_owned(),
                );
                let settings = model_settings(&opts)?;
                cfg.model_dir = Some(model.path.clone());
                cfg.backend_name = "npu".into();
                cfg.default_max_tokens = settings.usize("max_tokens");
                cfg.defaults = GenDefaults::from_settings(&settings)?;
                // Validate the tokenizer before acquiring a lock or starting a child.
                Tokenizer::load(&model.path)?;
                let backend = NpuBackend::load(npu_options(&opts, &settings)?)?;
                LoadedModel::chat(Box::new(backend), cfg)
            }
            ModelKind::Embedding => {
                cfg.backend_name = "trit-embed".into();
                let embedder = phlegm_core::embeddings::TritEmbeddingBackend::load(
                    model.id.clone(),
                    &python,
                    &source,
                    &model.path,
                )?;
                Ok(LoadedModel::embedding(Box::new(embedder), cfg))
            }
            ModelKind::Image => Ok(LoadedModel::image(ImageConfig::load(&model.path)?)),
        }
    })?;
    println!(
        "phlegm multi-model server on http://{}:{} (models load on request)",
        cfg.host, cfg.port
    );
    server.run(server::bind(&cfg)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_multi_model_options_fail_before_binding_or_loading() {
        for key in ["--keep-alive", "--queue-timeout"] {
            let opts = HashMap::from([(key.into(), "invalid".into())]);
            assert!(start(vec![], opts, ServerConfig::from_env())
                .unwrap_err()
                .contains(key));
        }
        let opts = HashMap::from([("--kernels".into(), "unused".into())]);
        assert!(serve(opts, ServerConfig::from_env())
            .unwrap_err()
            .contains("require --model-dir"));
    }
}
