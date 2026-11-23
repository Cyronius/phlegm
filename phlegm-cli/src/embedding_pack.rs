use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub const FLAGS: &[&str] = &[
    "--embed-packer",
    "--embed-gguf-sha256",
    "--embed-pack-script",
    "--embed-pack-cache",
];

pub struct PreparedPack {
    pub path: PathBuf,
    pub source_sha256: Option<String>,
}

pub fn is_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
}

pub fn prepare<'a>(
    model: &Path,
    design: &Path,
    option: impl Fn(&str) -> Option<&'a str>,
) -> Result<PreparedPack, String> {
    if !is_gguf(model) {
        if FLAGS.iter().any(|name| option(name).is_some()) || option("--embed-python").is_some() {
            return Err("NPUE packing options require a GGUF embedding model".into());
        }
        return Ok(PreparedPack {
            path: model.to_path_buf(),
            source_sha256: None,
        });
    }
    let expected = option("--embed-gguf-sha256")
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or("GGUF embedding models require a valid --embed-gguf-sha256")?
        .to_ascii_lowercase();
    let packer = option("--embed-packer").ok_or("GGUF embedding models require --embed-packer")?;
    let script = option("--embed-pack-script")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../open-kernels/utilities/bert-embeddings/pack_gguf.py")
        });
    if !script.is_file() {
        return Err("embedding packing script is missing; provide --embed-pack-script".into());
    }
    let cache = option("--embed-pack-cache")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            model
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(".npue-cache")
        });
    let python = option("--embed-python")
        .map(std::ffi::OsString::from)
        .or_else(|| std::env::var_os("PHLEGM_EMBED_PYTHON"))
        .unwrap_or_else(|| {
            if cfg!(windows) {
                "python".into()
            } else {
                "python3".into()
            }
        });
    let result = Command::new(python)
        .arg(script)
        .arg(model)
        .arg(design)
        .arg("--packer")
        .arg(packer)
        .arg("--gguf-sha256")
        .arg(&expected)
        .arg("--cache")
        .arg(&cache)
        .output()
        .map_err(|e| format!("starting the embedding packer: {e}"))?;
    if !result.status.success() {
        return Err(format!(
            "embedding packer {}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    let receipt: Value = serde_json::from_slice(&result.stdout)
        .map_err(|e| format!("reading embedding packer receipt: {e}"))?;
    if receipt.get("gguf_sha256").and_then(Value::as_str) != Some(expected.as_str()) {
        return Err("embedding packer did not confirm the requested GGUF hash".into());
    }
    let path = receipt
        .get("pack_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or("embedding packer omitted the packed model path")?;
    if !path.is_absolute()
        || !path.is_file()
        || !path
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(cache.canonicalize().map_err(|e| e.to_string())?)
    {
        return Err("embedding packer returned a path outside its cache".into());
    }
    Ok(PreparedPack {
        path,
        source_sha256: Some(expected),
    })
}
