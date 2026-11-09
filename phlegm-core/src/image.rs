//! Local image generation through Phlegm's native AMD NPU sampler.

use base64::encode;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::BTreeMap;
use tiny_http::{Header, Method, Request, Response, Server};

pub const MOSS_ID: &str = "OpenMOSS-Team/MOSS-Image-SFT";
pub const FLUX3_ID: &str = "black-forest-labs/flux-3-image-klein";
pub const QWEN21_ID: &str = "Qwen/Qwen-Image-2.1";
pub const FLUX2_KLEIN_IDS: [&str; 2] = ["black-forest-labs/FLUX.2-klein-4B",
                                         "black-forest-labs/FLUX.2-klein-9B"];
static REQUEST_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MossImageConfig {
    #[serde(default)]
    pub dibs: Option<PathBuf>,
    pub python: PathBuf,
    pub repo_root: PathBuf,
    pub env_script: PathBuf,
    pub model: PathBuf,
    pub dit_weights: PathBuf,
    pub vae_weights: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub text_model: PathBuf,
    pub text_kernels: PathBuf,
    pub text_binary: PathBuf,
    pub refine_kernels: String,
    pub condition_kernels: PathBuf,
    pub block_kernels: String,
    pub edge_kernels: PathBuf,
    pub vae_kernels: PathBuf,
    pub harness: PathBuf,
    pub output_root: PathBuf,
}

impl MossImageConfig {
    pub fn validate(&self) -> Result<(), String> {
        let cfg = self;
        if let Some(dibs) = &cfg.dibs {
            if !dibs.is_absolute() || !dibs.is_file() {
                return Err(format!("invalid Dibs launcher: {}", dibs.display()));
            }
        }
        for asset in [&cfg.python, &cfg.repo_root, &cfg.env_script, &cfg.model,
                      &cfg.dit_weights, &cfg.vae_weights, &cfg.tokenizer_dir,
                      &cfg.text_model, &cfg.text_kernels, &cfg.text_binary,
                      &cfg.condition_kernels, &cfg.edge_kernels,
                      &cfg.vae_kernels, &cfg.harness, &cfg.output_root] {
            if !asset.is_absolute() {
                return Err(format!("image config path must be absolute: {}", asset.display()));
            }
        }
        if !Path::new(&cfg.refine_kernels).is_absolute()
            || !Path::new(&cfg.block_kernels).is_absolute()
        {
            return Err("image kernel templates must be absolute paths".into());
        }
        for file in [&cfg.python, &cfg.env_script, &cfg.text_binary, &cfg.harness,
                     &cfg.model.join("pipeline_moss_image.py"),
                     &cfg.dit_weights.join("packed_weights.json"),
                     &cfg.vae_weights.join("manifest.json"),
                     &cfg.tokenizer_dir.join("tokenizer.json"),
                     &cfg.text_model.join("model.q4nx"),
                     &cfg.text_kernels.join("manifest.json"),
                     &cfg.condition_kernels.join("manifest.json"),
                     &cfg.edge_kernels.join("manifest.json"),
                     &cfg.vae_kernels.join("manifest.json")]
        {
            if !file.is_file() {
                return Err(format!("missing image asset: {}", file.display()));
            }
        }
        if !cfg.repo_root.join("open_kernels/moss_native_sample.py").is_file() {
            return Err("image repo_root has no native MOSS sampler".into());
        }
        if !cfg.refine_kernels.contains("{length}")
            || !cfg.block_kernels.contains("{length}")
            || !cfg.block_kernels.contains("{resolution}")
        {
            return Err("image kernel templates must name token length and resolution".into());
        }
        fs::create_dir_all(&cfg.output_root)
            .map_err(|e| format!("creating image output root: {e}"))?;
        Ok(())
    }

    pub fn supports_resolution(&self, resolution: u32) -> Result<(), String> {
        let edge: Value = serde_json::from_slice(&fs::read(
            self.edge_kernels.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let vae: Value = serde_json::from_slice(&fs::read(
            self.vae_kernels.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if edge["resolution"].as_u64() != Some(resolution as u64)
            || !vae["resolutions"].as_array().is_some_and(|values|
                values.iter().any(|v| v.as_u64() == Some(resolution as u64)))
        {
            return Err(format!("configured NPU image/VAE kernels do not cover {resolution} pixels"));
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flux3ImageConfig {
    pub backend: String,
    #[serde(default)]
    pub dibs: Option<PathBuf>,
    pub python: PathBuf,
    pub repo_root: PathBuf,
    pub env_script: PathBuf,
    pub model: PathBuf,
    pub dit_weights: PathBuf,
    pub vae_weights: PathBuf,
    pub pup_model: PathBuf,
    pub pup_kernels: PathBuf,
    pub text_model: PathBuf,
    pub text_kernels: PathBuf,
    pub text_binary: PathBuf,
    pub condition_kernels: PathBuf,
    pub edge_kernels: BTreeMap<String, PathBuf>,
    pub image_block_kernels: PathBuf,
    pub text_block_kernels: BTreeMap<String, PathBuf>,
    pub shared_block_kernels: BTreeMap<String, PathBuf>,
    pub vae_kernels: PathBuf,
    pub harness: PathBuf,
    pub output_root: PathBuf,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_pup_max_tokens")]
    pub pup_max_tokens: u32,
    /// One-reference editing: native VAE encoder, image_cond per-mode blocks,
    /// and shared blocks compiled for text + image + one reference.
    #[serde(default)]
    pub vae_encoder_kernels: Option<PathBuf>,
    #[serde(default)]
    pub vae_encoder_weights: Option<PathBuf>,
    #[serde(default)]
    pub cond_block_kernels: Option<PathBuf>,
    #[serde(default)]
    pub edit_shared_block_kernels: BTreeMap<String, PathBuf>,
    /// Text-to-image kernel sets for square sizes other than 512 (key "1024").
    /// The text encoder, upsampler, conditioning and weights are shared.
    #[serde(default)]
    pub resolutions: BTreeMap<String, Flux3ResolutionKernels>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flux3ResolutionKernels {
    pub edge_kernels: BTreeMap<String, PathBuf>,
    pub image_block_kernels: PathBuf,
    pub text_block_kernels: BTreeMap<String, PathBuf>,
    pub shared_block_kernels: BTreeMap<String, PathBuf>,
    pub vae_kernels: PathBuf,
}

impl Flux3ResolutionKernels {
    fn validate(&self, resolution: u64) -> Result<(), String> {
        let keys = |m: &BTreeMap<String, PathBuf>| m.keys().cloned().collect::<Vec<_>>();
        if self.edge_kernels.is_empty() || keys(&self.edge_kernels) != keys(&self.text_block_kernels)
            || keys(&self.edge_kernels) != keys(&self.shared_block_kernels) {
            return Err(format!("FLUX 3 {resolution}-pixel bucket maps must have the same nonempty keys"));
        }
        for (bucket, _) in &self.edge_kernels {
            let length: u64 = bucket.parse().map_err(|_| format!("invalid FLUX 3 text bucket {bucket}"))?;
            if length < 80 || length % 80 != 0 {
                return Err(format!("invalid FLUX 3 text bucket {bucket}"));
            }
            for map in [&self.edge_kernels, &self.text_block_kernels, &self.shared_block_kernels] {
                let path = &map[bucket];
                if !path.is_absolute() { return Err(format!("FLUX 3 kernel path must be absolute: {}", path.display())); }
                let manifest = read_manifest(&path.join("manifest.json"))?;
                if manifest["text_length"].as_u64() != Some(length)
                    || manifest["resolution"].as_u64() != Some(resolution) {
                    return Err(format!("FLUX 3 {resolution}-pixel kernel bucket mismatch: {}", path.display()));
                }
            }
        }
        let image = read_manifest(&self.image_block_kernels.join("manifest.json"))?;
        let vae = read_manifest(&self.vae_kernels.join("manifest.json"))?;
        if image["resolution"].as_u64() != Some(resolution) || image["mode"].as_str() != Some("image")
            || !vae["resolutions"].as_array().is_some_and(|v| v.iter().any(|r| r.as_u64() == Some(resolution))) {
            return Err(format!("FLUX 3 image block / VAE kernels do not cover {resolution} pixels"));
        }
        Ok(())
    }
}

fn default_alpha() -> f32 { 16.0 }
fn default_pup_max_tokens() -> u32 { 512 }

impl Flux3ImageConfig {
    fn validate(&self) -> Result<(), String> {
        if self.backend != "flux3" {
            return Err("FLUX 3 image config requires backend=flux3".into());
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || !(1..=3584).contains(&self.pup_max_tokens) {
            return Err("invalid FLUX 3 alpha or pup_max_tokens".into());
        }
        if let Some(dibs) = &self.dibs {
            if !dibs.is_absolute() || !dibs.is_file() {
                return Err(format!("invalid Dibs launcher: {}", dibs.display()));
            }
        }
        for path in [&self.python, &self.repo_root, &self.env_script, &self.model,
                     &self.dit_weights, &self.vae_weights, &self.pup_model,
                     &self.pup_kernels, &self.text_model, &self.text_kernels,
                     &self.text_binary, &self.condition_kernels, &self.image_block_kernels,
                     &self.vae_kernels, &self.harness, &self.output_root] {
            if !path.is_absolute() {
                return Err(format!("FLUX 3 image config path must be absolute: {}", path.display()));
            }
        }
        let mut files = vec![self.python.clone(), self.env_script.clone(), self.text_binary.clone(),
            self.harness.clone(), self.model.join("model.safetensors"),
            self.dit_weights.join("packed_weights.json"), self.vae_weights.join("manifest.json"),
            self.pup_model.join("model.q4nx"), self.pup_kernels.join("manifest.json"),
            self.text_model.join("model.q4nx"), self.text_kernels.join("manifest.json"),
            self.condition_kernels.join("manifest.json"),
            self.image_block_kernels.join("manifest.json"), self.vae_kernels.join("manifest.json"),
            self.repo_root.join("open_kernels/flux3_native_generate.py")];
        if self.edge_kernels.is_empty() || self.edge_kernels.keys().collect::<Vec<_>>()
            != self.text_block_kernels.keys().collect::<Vec<_>>()
            || self.edge_kernels.keys().collect::<Vec<_>>()
            != self.shared_block_kernels.keys().collect::<Vec<_>>() {
            return Err("FLUX 3 kernel bucket maps must have the same nonempty keys".into());
        }
        for bucket in self.edge_kernels.keys() {
            let length: usize = bucket.parse().map_err(|_| format!("invalid FLUX 3 text bucket {bucket}"))?;
            if length < 80 || length % 80 != 0 {
                return Err(format!("invalid FLUX 3 text bucket {bucket}"));
            }
            for map in [&self.edge_kernels, &self.text_block_kernels, &self.shared_block_kernels] {
                let path = &map[bucket];
                if !path.is_absolute() { return Err(format!("FLUX 3 kernel path must be absolute: {}", path.display())); }
                let manifest = path.join("manifest.json");
                files.push(manifest.clone());
                let metadata: Value = serde_json::from_slice(&fs::read(&manifest)
                    .map_err(|e| format!("reading {}: {e}", manifest.display()))?)
                    .map_err(|e| format!("parsing {}: {e}", manifest.display()))?;
                if metadata["text_length"].as_u64() != Some(length as u64)
                    || metadata["resolution"].as_u64().is_some_and(|r| r != 512) {
                    return Err(format!("FLUX 3 kernel bucket mismatch: {}", manifest.display()));
                }
            }
        }
        let edit = [&self.vae_encoder_kernels, &self.vae_encoder_weights, &self.cond_block_kernels];
        if edit.iter().any(|p| p.is_some()) || !self.edit_shared_block_kernels.is_empty() {
            if edit.iter().any(|p| p.is_none()) || self.edit_shared_block_kernels.is_empty() {
                return Err("FLUX 3 editing needs vae_encoder_kernels, vae_encoder_weights, \
                            cond_block_kernels and edit_shared_block_kernels together".into());
            }
            for path in edit.into_iter().flatten() {
                if !path.is_absolute() {
                    return Err(format!("FLUX 3 image config path must be absolute: {}", path.display()));
                }
            }
            files.push(self.vae_encoder_weights.as_ref().unwrap().join("manifest.json"));
            for dir in [&self.vae_encoder_kernels, &self.cond_block_kernels] {
                files.push(dir.as_ref().unwrap().join("manifest.json"));
            }
            let cond: Value = read_manifest(&self.cond_block_kernels.as_ref().unwrap().join("manifest.json"))?;
            if cond["mode"].as_str() != Some("image_cond") || cond["resolution"].as_u64() != Some(512) {
                return Err("FLUX 3 cond_block_kernels must be 512-pixel image_cond kernels".into());
            }
            for (bucket, path) in &self.edit_shared_block_kernels {
                if !self.edge_kernels.contains_key(bucket) || !path.is_absolute() {
                    return Err(format!("invalid FLUX 3 edit shared kernel bucket {bucket}"));
                }
                let manifest = read_manifest(&path.join("manifest.json"))?;
                if manifest["text_length"].as_u64() != bucket.parse::<u64>().ok()
                    || manifest["references"].as_u64() != Some(1)
                    || manifest["resolution"].as_u64() != Some(512) {
                    return Err(format!("FLUX 3 edit shared kernel bucket mismatch: {}", path.display()));
                }
            }
        }
        for file in files {
            if !file.is_file() { return Err(format!("missing FLUX 3 image asset: {}", file.display())); }
        }
        for (size, kernels) in &self.resolutions {
            if size != "1024" {
                return Err(format!("FLUX 3 native kernels cover 512 and 1024 pixels, not {size}"));
            }
            kernels.validate(1024)?;
        }
        fs::create_dir_all(&self.output_root).map_err(|e| format!("creating image output root: {e}"))?;
        Ok(())
    }

    pub fn supports_resolution(&self, resolution: u32) -> bool {
        resolution == 512 || self.resolutions.contains_key(&resolution.to_string())
    }

    pub fn supports_edit(&self) -> bool {
        self.cond_block_kernels.is_some()
    }
}

fn read_manifest(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?)
        .map_err(|e| format!("parsing {}: {e}", path.display()))
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen21ImageConfig {
    pub backend: String,
    #[serde(default)]
    pub dibs: Option<PathBuf>,
    pub python: PathBuf,
    pub repo_root: PathBuf,
    pub env_script: PathBuf,
    pub model: PathBuf,
    pub dit_weights: PathBuf,
    pub text_model: PathBuf,
    pub text_kernels: PathBuf,
    pub text_binary: PathBuf,
    pub edge_kernels: PathBuf,
    pub block_kernels: PathBuf,
    pub kernel_probes: PathBuf,
    pub vae_weights: PathBuf,
    pub vae_conv_kernels: PathBuf,
    pub vae_ew_kernels: PathBuf,
    pub vae_attention_kernel: PathBuf,
    pub harness: PathBuf,
    pub output_root: PathBuf,
}

impl Qwen21ImageConfig {
    fn validate(&self) -> Result<(), String> {
        if self.backend != "qwen21" {
            return Err("Qwen Image 2.1 config requires backend=qwen21".into());
        }
        if let Some(dibs) = &self.dibs {
            if !dibs.is_absolute() || !dibs.is_file() {
                return Err(format!("invalid Dibs launcher: {}", dibs.display()));
            }
        }
        for path in [&self.python, &self.repo_root, &self.env_script, &self.model,
                     &self.dit_weights, &self.text_model, &self.text_kernels,
                     &self.text_binary, &self.edge_kernels, &self.block_kernels,
                     &self.kernel_probes, &self.vae_weights, &self.vae_conv_kernels,
                     &self.vae_ew_kernels, &self.vae_attention_kernel, &self.harness,
                     &self.output_root] {
            if !path.is_absolute() {
                return Err(format!("Qwen image config path must be absolute: {}", path.display()));
            }
        }
        for file in [&self.python, &self.env_script, &self.text_binary, &self.harness,
                     &self.model.join("scheduler/scheduler_config.json"),
                     &self.dit_weights.join("packed_weights.json"),
                     &self.text_model.join("model.q4nx"),
                     &self.text_model.join("tokenizer.json"),
                     &self.text_kernels.join("manifest.json"),
                     &self.edge_kernels.join("manifest.json"),
                     &self.block_kernels, &self.kernel_probes,
                     &self.vae_weights.join("manifest.json"),
                     &self.vae_conv_kernels.join("manifest.json"),
                     &self.vae_ew_kernels.join("manifest.json"),
                     &self.vae_attention_kernel.join("final.xclbin"),
                     &self.vae_attention_kernel.join("insts.bin"),
                     &self.repo_root.join("open_kernels/qwen21_native_generate.py")] {
            if !file.is_file() {
                return Err(format!("missing Qwen image asset: {}", file.display()));
            }
        }
        fs::create_dir_all(&self.output_root)
            .map_err(|e| format!("creating Qwen image output root: {e}"))?;
        Ok(())
    }
}

/// FLUX.2 [klein] (distilled) through `open_kernels/flux2_native_generate.py`: the klein
/// schedule on the pyxrt host, weights read from the published diffusers checkpoint.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flux2ImageConfig {
    pub backend: String,
    /// One of FLUX2_KLEIN_IDS; the runner checks the checkpoint's geometry against the
    /// compiled schedule.
    pub model_id: String,
    #[serde(default)]
    pub dibs: Option<PathBuf>,
    pub python: PathBuf,
    pub repo_root: PathBuf,
    #[serde(default)]
    pub env_script: Option<PathBuf>,
    pub model: PathBuf,
    pub kernels: PathBuf,
    pub output_root: PathBuf,
}

impl Flux2ImageConfig {
    fn validate(&self) -> Result<(), String> {
        if self.backend != "flux2" {
            return Err("FLUX.2 klein config requires backend=flux2".into());
        }
        if !FLUX2_KLEIN_IDS.contains(&self.model_id.as_str()) {
            return Err(format!("unsupported FLUX.2 model_id {}; expected one of {FLUX2_KLEIN_IDS:?}",
                               self.model_id));
        }
        if let Some(dibs) = &self.dibs {
            if !dibs.is_absolute() || !dibs.is_file() {
                return Err(format!("invalid Dibs launcher: {}", dibs.display()));
            }
        }
        for path in [Some(&self.python), Some(&self.repo_root), self.env_script.as_ref(),
                     Some(&self.model), Some(&self.kernels), Some(&self.output_root)]
            .into_iter().flatten() {
            if !path.is_absolute() {
                return Err(format!("FLUX.2 image config path must be absolute: {}", path.display()));
            }
        }
        for file in [Some(&self.python), self.env_script.as_ref(),
                     Some(&self.model.join("model_index.json")),
                     Some(&self.model.join("transformer/config.json")),
                     Some(&self.model.join("tokenizer/tokenizer.json")),
                     Some(&self.kernels.join("final.xclbin")),
                     Some(&self.kernels.join("fa/dit_fa.json")),
                     Some(&self.kernels.join("ew/dit_ew.json")),
                     Some(&self.repo_root.join("open_kernels/flux2_native_generate.py"))]
            .into_iter().flatten() {
            if !file.is_file() {
                return Err(format!("missing FLUX.2 image asset: {}", file.display()));
            }
        }
        let index = read_manifest(&self.model.join("model_index.json"))?;
        if index["is_distilled"].as_bool() != Some(true) {
            return Err("only distilled FLUX.2 klein checkpoints are supported (Base needs CFG)".into());
        }
        let (resolutions, _) = self.kernel_sizes()?;
        if resolutions.is_empty() {
            return Err("FLUX.2 kernel set lists no resolutions".into());
        }
        fs::create_dir_all(&self.output_root)
            .map_err(|e| format!("creating FLUX.2 image output root: {e}"))?;
        Ok(())
    }

    /// (text-to-image resolutions, edit resolutions) from the dit_gemm set's marker.
    fn kernel_sizes(&self) -> Result<(Vec<u64>, Vec<u64>), String> {
        let marker = read_manifest(&self.kernels.join("dit_kernels.json"))?;
        if marker["complete"].as_bool() != Some(true) {
            return Err(format!("incomplete FLUX.2 kernel set: {}", self.kernels.display()));
        }
        let list = |key: &str| marker[key].as_array().map(|v| v.iter()
            .filter_map(Value::as_u64).collect::<Vec<_>>()).unwrap_or_default();
        Ok((list("resolutions"), list("edits")))
    }

    pub fn supports_resolution(&self, resolution: u32) -> bool {
        self.kernel_sizes().is_ok_and(|(r, _)| r.contains(&(resolution as u64)))
    }

    pub fn supports_edit(&self, resolution: u32) -> bool {
        self.kernel_sizes().is_ok_and(|(_, e)| e.contains(&(resolution as u64)))
    }

    fn model_id(&self) -> &'static str {
        FLUX2_KLEIN_IDS.iter().copied().find(|id| *id == self.model_id).unwrap_or(FLUX2_KLEIN_IDS[0])
    }
}

#[derive(Clone)]
pub enum ImageConfig {
    Moss(MossImageConfig),
    Flux3(Flux3ImageConfig),
    Qwen21(Qwen21ImageConfig),
    Flux2(Flux2ImageConfig),
}

impl ImageConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let value: Value = serde_json::from_slice(&data)
            .map_err(|e| format!("parsing {}: {e}", path.display()))?;
        let cfg = match value.get("backend").and_then(Value::as_str) {
            Some("flux3") => {
                let flux: Flux3ImageConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
                flux.validate()?;
                Self::Flux3(flux)
            }
            Some("qwen21") => {
                let qwen: Qwen21ImageConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
                qwen.validate()?;
                Self::Qwen21(qwen)
            }
            Some("flux2") => {
                let flux: Flux2ImageConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
                flux.validate()?;
                Self::Flux2(flux)
            }
            None => {
                let moss: MossImageConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
                moss.validate()?;
                Self::Moss(moss)
            }
            Some(other) => return Err(format!("unsupported image backend {other}")),
        };
        Ok(cfg)
    }

    pub fn model_id(&self) -> &'static str {
        match self { Self::Moss(_) => MOSS_ID, Self::Flux3(_) => FLUX3_ID,
            Self::Qwen21(_) => QWEN21_ID, Self::Flux2(c) => c.model_id() }
    }

    pub fn output_root(&self) -> &Path {
        match self { Self::Moss(c) => &c.output_root, Self::Flux3(c) => &c.output_root,
            Self::Qwen21(c) => &c.output_root, Self::Flux2(c) => &c.output_root }
    }

    pub fn supports_resolution(&self, resolution: u32) -> Result<(), String> {
        match self {
            Self::Moss(c) => c.supports_resolution(resolution),
            Self::Flux3(c) if c.supports_resolution(resolution) => Ok(()),
            Self::Flux3(_) => Err(format!("this FLUX 3 configuration has no native {resolution}-pixel kernels")),
            Self::Qwen21(_) if resolution == 512 => Ok(()),
            Self::Qwen21(_) => Err("Qwen Image 2.1 currently supports 512x512 on the native NPU".into()),
            Self::Flux2(c) if c.supports_resolution(resolution) => Ok(()),
            Self::Flux2(_) => Err(format!("this FLUX.2 klein configuration has no native {resolution}-pixel kernels")),
        }
    }

    pub fn validate_request(&self, request: &ImageRequest) -> Result<u32, String> {
        let resolution = request.validate()?;
        if request.model.as_deref().is_some_and(|id| id != self.model_id()) {
            return Err(format!("unsupported image model; expected {}", self.model_id()));
        }
        self.supports_resolution(resolution)?;
        if let Some(image) = &request.image {
            match self {
                Self::Flux3(_) if resolution != 512 => {
                    return Err("FLUX 3 native editing covers 512x512 references".into())
                }
                Self::Flux3(flux) if flux.supports_edit() => {}
                Self::Flux3(_) => return Err("this FLUX 3 configuration has no native editing kernels".into()),
                // The published MOSS pipeline takes no reference image; editing is not
                // part of the model (user decision 2026-11-08: reject, no img2img stand-in).
                Self::Moss(_) => return Err(format!(
                    "{} does not support editing: its published pipeline takes no reference image",
                    self.model_id())),
                Self::Qwen21(_) => return Err(format!("{} editing has no native NPU route yet",
                                                      self.model_id())),
                Self::Flux2(flux) if flux.supports_edit(resolution) => {}
                Self::Flux2(_) => return Err(format!(
                    "this FLUX.2 klein configuration has no native {resolution}-pixel editing kernels")),
            }
            if !image.is_absolute() || !image.is_file() {
                return Err("reference image must be an existing absolute path".into());
            }
        }
        if matches!(self, Self::Flux2(_)) {
            if request.steps.is_some_and(|steps| steps > 50) {
                return Err("FLUX.2 klein steps must be between 1 and 50".into());
            }
            if request.guidance_scale.is_some() {
                return Err("FLUX.2 klein is step-distilled and does not use guidance_scale".into());
            }
            if !request.negative_prompt.is_empty() {
                return Err("FLUX.2 klein does not support negative_prompt".into());
            }
        }
        if matches!(self, Self::Flux3(_) | Self::Qwen21(_)) {
            let (name, default_steps) = if matches!(self, Self::Qwen21(_)) {
                ("Qwen Image 2.1", 40)
            } else { ("FLUX 3", 16) };
            if request.steps.unwrap_or(default_steps) > 50
                || (matches!(self, Self::Qwen21(_)) && request.steps == Some(1)) {
                let minimum = if matches!(self, Self::Qwen21(_)) { 2 } else { 1 };
                return Err(format!("{name} steps must be between {minimum} and 50"));
            }
            if request.guidance_scale.is_some() {
                return Err(format!("{name} does not use guidance_scale"));
            }
            if !request.negative_prompt.is_empty() {
                return Err(format!("{name} does not support negative_prompt"));
            }
        }
        Ok(resolution)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub guidance_scale: Option<f32>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    /// Reference image for an edit; set by the CLI or the edits endpoint.
    #[serde(skip)]
    pub image: Option<PathBuf>,
}

fn default_size() -> String { "512x512".into() }
fn default_seed() -> u64 { 41 }
fn default_n() -> u32 { 1 }
fn default_response_format() -> String { "b64_json".into() }

impl ImageRequest {
    pub fn validate(&self) -> Result<u32, String> {
        if self.prompt.trim().is_empty() {
            return Err("prompt must be nonempty".into());
        }
        let resolution = match self.size.as_str() {
            "512x512" => 512,
            "1024x1024" => 1024,
            _ => return Err("size must be 512x512 or 1024x1024".into()),
        };
        if self.steps.is_some_and(|steps| !(1..=100).contains(&steps)) {
            return Err("steps must be between 1 and 100".into());
        }
        if self.guidance_scale.is_some_and(|scale| !scale.is_finite() || scale < 0.0) {
            return Err("guidance_scale must be finite and nonnegative".into());
        }
        if self.n != 1 {
            return Err("only n=1 is supported".into());
        }
        if self.response_format != "b64_json" {
            return Err("only response_format=b64_json is supported".into());
        }
        Ok(resolution)
    }
}

pub fn generate(cfg: &ImageConfig, request: &ImageRequest, out: &Path) -> Result<PathBuf, String> {
    let resolution = cfg.validate_request(request)?;
    if out.exists() {
        return Err(format!("image output directory already exists: {}", out.display()));
    }
    match cfg {
        ImageConfig::Moss(moss) => generate_moss(moss, request, out, resolution),
        ImageConfig::Flux3(flux) => generate_flux3(flux, request, out, resolution),
        ImageConfig::Qwen21(qwen) => generate_qwen21(qwen, request, out, resolution),
        ImageConfig::Flux2(flux) => generate_flux2(flux, request, out, resolution),
    }
}

fn generate_moss(cfg: &MossImageConfig, request: &ImageRequest,
                 out: &Path, resolution: u32) -> Result<PathBuf, String> {
    let steps = request.steps.unwrap_or(50);
    let guidance = request.guidance_scale.unwrap_or(3.0);
    let mut process = if let Some(dibs) = &cfg.dibs {
        let mut wrapped = Command::new(dibs);
        wrapped.args(["run", "npu", "cpu", "--label", "Phlegm native image generation",
                      "--eta"])
            .arg(format!("{}m", (steps * 2 + 10).min(240)))
            .arg("--").arg("bash");
        wrapped
    } else {
        Command::new("bash")
    };
    process.arg("-c").arg("source \"$1\" && shift && exec \"$@\"")
        .arg("phlegm-image")
        .arg(&cfg.env_script)
        .arg(&cfg.python).arg("-m").arg("open_kernels.moss_native_sample")
        .arg("--prompt").arg(&request.prompt)
        .arg("--negative-prompt").arg(&request.negative_prompt)
        .arg("--model").arg(&cfg.model)
        .arg("--dit-weights").arg(&cfg.dit_weights)
        .arg("--vae-weights").arg(&cfg.vae_weights)
        .arg("--tokenizer-dir").arg(&cfg.tokenizer_dir)
        .arg("--text-model").arg(&cfg.text_model)
        .arg("--text-kernels").arg(&cfg.text_kernels)
        .arg("--text-binary").arg(&cfg.text_binary)
        .arg("--refine-kernels").arg(&cfg.refine_kernels)
        .arg("--condition-kernels").arg(&cfg.condition_kernels)
        .arg("--block-kernels").arg(&cfg.block_kernels)
        .arg("--edge-kernels").arg(&cfg.edge_kernels)
        .arg("--vae-kernels").arg(&cfg.vae_kernels)
        .arg("--harness").arg(&cfg.harness)
        .arg("--out").arg(out)
        .arg("--resolution").arg(resolution.to_string())
        .arg("--steps").arg(steps.to_string())
        .arg("--guidance").arg(guidance.to_string())
        .arg("--seed").arg(request.seed.to_string())
        .current_dir(&cfg.repo_root);
    let status = process.status().map_err(|e| format!("starting native image sampler: {e}"))?;
    if !status.success() {
        return Err(format!("native image sampler exited with {status}"));
    }
    let image = out.join("decode/native-vae.png");
    if !image.is_file() {
        return Err("native image sampler did not produce a PNG".into());
    }
    Ok(image)
}

fn generate_flux3(cfg: &Flux3ImageConfig, request: &ImageRequest,
                  out: &Path, resolution: u32) -> Result<PathBuf, String> {
    let steps = request.steps.unwrap_or(16);
    let (edge, image_block, text_block, shared_block, vae) = match cfg.resolutions.get(&resolution.to_string()) {
        Some(k) => (&k.edge_kernels, &k.image_block_kernels, &k.text_block_kernels,
                    &k.shared_block_kernels, &k.vae_kernels),
        None => (&cfg.edge_kernels, &cfg.image_block_kernels, &cfg.text_block_kernels,
                 &cfg.shared_block_kernels, &cfg.vae_kernels),
    };
    let mut process = if let Some(dibs) = &cfg.dibs {
        let mut wrapped = Command::new(dibs);
        wrapped.args(["run", "npu", "cpu", "--label", "Phlegm native FLUX 3 image generation",
                      "--eta"])
            .arg(format!("{}m", (steps * if resolution > 512 { 8 } else { 2 } + 20).min(480)))
            .arg("--").arg("bash");
        wrapped
    } else {
        Command::new("bash")
    };
    process.arg("-c").arg("source \"$1\" && shift && exec \"$@\"")
        .arg("phlegm-flux3-image")
        .arg(&cfg.env_script)
        .arg(&cfg.python).arg("-m").arg("open_kernels.flux3_native_generate")
        .arg("--prompt").arg(&request.prompt)
        .arg("--model").arg(&cfg.model)
        .arg("--dit-weights").arg(&cfg.dit_weights)
        .arg("--vae-weights").arg(&cfg.vae_weights)
        .arg("--pup-model").arg(&cfg.pup_model)
        .arg("--pup-kernels").arg(&cfg.pup_kernels)
        .arg("--text-model").arg(&cfg.text_model)
        .arg("--text-kernels").arg(&cfg.text_kernels)
        .arg("--text-binary").arg(&cfg.text_binary)
        .arg("--condition-kernels").arg(&cfg.condition_kernels)
        .arg("--edge-kernels").arg(serde_json::to_string(edge).map_err(|e| e.to_string())?)
        .arg("--image-block-kernels").arg(image_block)
        .arg("--text-block-kernels").arg(serde_json::to_string(text_block).map_err(|e| e.to_string())?)
        .arg("--shared-block-kernels").arg(serde_json::to_string(shared_block).map_err(|e| e.to_string())?)
        .arg("--vae-kernels").arg(vae)
        .arg("--harness").arg(&cfg.harness)
        .arg("--out").arg(out)
        .arg("--resolution").arg(resolution.to_string())
        .arg("--steps").arg(steps.to_string())
        .arg("--alpha").arg(cfg.alpha.to_string())
        .arg("--seed").arg(request.seed.to_string())
        .arg("--pup-max-tokens").arg(cfg.pup_max_tokens.to_string())
        .current_dir(&cfg.repo_root);
    if let Some(image) = &request.image {
        process.arg("--image").arg(image)
            .arg("--vae-encoder-kernels").arg(cfg.vae_encoder_kernels.as_ref().unwrap())
            .arg("--vae-encoder-weights").arg(cfg.vae_encoder_weights.as_ref().unwrap())
            .arg("--cond-block-kernels").arg(cfg.cond_block_kernels.as_ref().unwrap())
            .arg("--edit-shared-block-kernels")
            .arg(serde_json::to_string(&cfg.edit_shared_block_kernels).map_err(|e| e.to_string())?)
            .arg("--discard-steps");
    }
    let status = process.status().map_err(|e| format!("starting native FLUX 3 sampler: {e}"))?;
    if !status.success() { return Err(format!("native FLUX 3 sampler exited with {status}")); }
    let summary = read_manifest(&out.join("summary.json"))?;
    let image = summary["image"].as_str().map(PathBuf::from)
        .unwrap_or_else(|| out.join("image/decode/native-vae.png"));
    if !image.is_file() { return Err("native FLUX 3 sampler did not produce a PNG".into()); }
    Ok(image)
}

fn generate_qwen21(cfg: &Qwen21ImageConfig, request: &ImageRequest,
                   out: &Path, _resolution: u32) -> Result<PathBuf, String> {
    let steps = request.steps.unwrap_or(40);
    let mut process = if let Some(dibs) = &cfg.dibs {
        let mut wrapped = Command::new(dibs);
        wrapped.args(["run", "npu", "cpu", "--label", "Phlegm native Qwen Image 2.1 generation",
                      "--eta"])
            .arg(format!("{}m", (steps * 2 + 20).min(240)))
            .arg("--").arg("bash");
        wrapped
    } else {
        Command::new("bash")
    };
    process.arg("-c").arg("source \"$1\" && shift && exec \"$@\"")
        .arg("phlegm-qwen21-image")
        .arg(&cfg.env_script)
        .arg(&cfg.python).arg("-m").arg("open_kernels.qwen21_native_generate")
        .arg("--prompt").arg(&request.prompt)
        .arg("--model").arg(&cfg.model)
        .arg("--dit-weights").arg(&cfg.dit_weights)
        .arg("--text-model").arg(&cfg.text_model)
        .arg("--text-kernels").arg(&cfg.text_kernels)
        .arg("--text-binary").arg(&cfg.text_binary)
        .arg("--edge-kernels").arg(&cfg.edge_kernels)
        .arg("--block-kernels").arg(&cfg.block_kernels)
        .arg("--kernel-probes").arg(&cfg.kernel_probes)
        .arg("--vae-weights").arg(&cfg.vae_weights)
        .arg("--vae-conv-kernels").arg(&cfg.vae_conv_kernels)
        .arg("--vae-ew-kernels").arg(&cfg.vae_ew_kernels)
        .arg("--vae-attention-kernel").arg(&cfg.vae_attention_kernel)
        .arg("--harness").arg(&cfg.harness)
        .arg("--out").arg(out)
        .arg("--steps").arg(steps.to_string())
        .arg("--seed").arg(request.seed.to_string())
        .current_dir(&cfg.repo_root);
    let status = process.status().map_err(|e| format!("starting native Qwen sampler: {e}"))?;
    if !status.success() { return Err(format!("native Qwen sampler exited with {status}")); }
    let image = out.join("decode/native-qwen21.png");
    if !image.is_file() { return Err("native Qwen sampler did not produce a PNG".into()); }
    Ok(image)
}

fn generate_flux2(cfg: &Flux2ImageConfig, request: &ImageRequest,
                  out: &Path, resolution: u32) -> Result<PathBuf, String> {
    let steps = request.steps.unwrap_or(4);
    let mut process = if let Some(dibs) = &cfg.dibs {
        let mut wrapped = Command::new(dibs);
        wrapped.args(["run", "npu", "cpu", "--label", "Phlegm native FLUX.2 klein generation",
                      "--eta"])
            .arg(format!("{}m", (steps * if resolution > 512 { 2 } else { 1 } + 15).min(120)))
            .arg("--").arg("bash");
        wrapped
    } else {
        Command::new("bash")
    };
    let script = if cfg.env_script.is_some() { "source \"$1\" && shift && exec \"$@\"" }
                 else { "shift && exec \"$@\"" };
    process.arg("-c").arg(script).arg("phlegm-flux2-image")
        .arg(cfg.env_script.clone().unwrap_or_default())
        .arg(&cfg.python).arg("-m").arg("open_kernels.flux2_native_generate")
        .arg("--prompt").arg(&request.prompt)
        .arg("--model").arg(&cfg.model)
        .arg("--kernels").arg(&cfg.kernels)
        .arg("--size").arg(resolution.to_string())
        .arg("--steps").arg(steps.to_string())
        .arg("--seed").arg(request.seed.to_string())
        .arg("--out").arg(out)
        .current_dir(&cfg.repo_root);
    if let Some(image) = &request.image {
        process.arg("--image").arg(image);
    }
    let status = process.status().map_err(|e| format!("starting native FLUX.2 sampler: {e}"))?;
    if !status.success() { return Err(format!("native FLUX.2 sampler exited with {status}")); }
    let summary = read_manifest(&out.join("summary.json"))?;
    let image = summary["image"].as_str().map(PathBuf::from)
        .unwrap_or_else(|| out.join("image.png"));
    if !image.is_file() { return Err("native FLUX.2 sampler did not produce a PNG".into()); }
    Ok(image)
}

fn respond(request: Request, status: u16, value: Value) {
    let bytes = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let _ = request.respond(Response::from_data(bytes).with_status_code(status).with_header(header));
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn handle(mut request: Request, cfg: &ImageConfig) {
    if request.method() == &Method::Get && request.url() == "/health" {
        respond(request, 200, json!({"status":"ok","backend":"native-npu-image"}));
        return;
    }
    if request.method() == &Method::Get && request.url() == "/v1/models" {
        let owner = match cfg { ImageConfig::Moss(_) => "openmoss",
            ImageConfig::Flux3(_) | ImageConfig::Flux2(_) => "black-forest-labs",
            ImageConfig::Qwen21(_) => "Qwen" };
        respond(request, 200, json!({"object":"list","data":[{"id":cfg.model_id(),
            "object":"model","created":now_unix(),"owned_by":owner}]}));
        return;
    }
    let edit = request.url() == "/v1/images/edits";
    if request.method() != &Method::Post || !(edit || request.url() == "/v1/images/generations") {
        respond(request, 404, json!({"error":{"message":"unknown path","type":"not_found"}}));
        return;
    }
    let limit = if edit { EDIT_BODY_LIMIT } else { 64 * 1024 };
    if request.body_length().is_some_and(|n| n > limit) {
        respond(request, 413, json!({"error":{"message":"request too large","type":"invalid_request_error"}}));
        return;
    }
    let content_type = request.headers().iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    let mut body = Vec::new();
    if request.as_reader().take(limit as u64 + 1).read_to_end(&mut body).is_err()
        || body.len() > limit
    {
        respond(request, 413, json!({"error":{"message":"request too large","type":"invalid_request_error"}}));
        return;
    }
    let serial = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let prefix = match cfg { ImageConfig::Moss(_) => "moss", ImageConfig::Flux3(_) => "flux3",
        ImageConfig::Qwen21(_) => "qwen21", ImageConfig::Flux2(_) => "flux2" };
    let name = format!("{prefix}{}-{}-{serial}", if edit { "-edit" } else { "" }, now_unix());
    let stem = cfg.output_root().join(format!("{name}-reference"));
    let parsed = if edit {
        parse_edit(&content_type, &body, &stem)
    } else {
        serde_json::from_slice::<ImageRequest>(&body).map_err(|e| e.to_string())
    }.and_then(|r| cfg.validate_request(&r).map(|_| r));
    let image_request = match parsed {
        Ok(r) => r,
        Err(message) => {
            for ext in ["png", "jpg", "webp"] {
                let _ = fs::remove_file(stem.with_extension(ext));
            }
            respond(request, 400, json!({"error":{"message":message,"type":"invalid_request_error"}}));
            return;
        }
    };
    let out = cfg.output_root().join(&name);
    match generate(cfg, &image_request, &out).and_then(|path| fs::read(&path)
         .map_err(|e| format!("reading generated image: {e}"))) {
        Ok(png) => respond(request, 200, json!({"created":now_unix(),
            "data":[{"b64_json":encode(png)}]})),
        Err(message) => respond(request, 500, json!({"error":{"message":message,
            "type":"generation_error"}})),
    }
}

/// Largest accepted `/v1/images/edits` body: one reference image plus fields.
const EDIT_BODY_LIMIT: usize = 32 * 1024 * 1024;

/// Image bytes are accepted only as PNG, JPEG or WebP; the file keeps that extension.
fn image_extension(bytes: &[u8]) -> Result<&'static str, String> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") { return Ok("png"); }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) { return Ok("jpg"); }
    if bytes.len() > 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" { return Ok("webp"); }
    Err("reference image must be PNG, JPEG or WebP".into())
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    haystack.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|i| i + from)
}

/// `multipart/form-data` fields as (name, has filename, bytes).
fn multipart(content_type: &str, body: &[u8]) -> Result<Vec<(String, bool, Vec<u8>)>, String> {
    let boundary = content_type.split(';').map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))
        .map(|b| b.trim_matches('"'))
        .filter(|b| !b.is_empty())
        .ok_or("multipart request has no boundary")?;
    let delimiter = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let mut at = find(body, &delimiter, 0).ok_or("multipart body has no parts")? + delimiter.len();
    loop {
        if body.get(at..at + 2) == Some(b"--") { break; }
        at = find(body, b"\r\n", at).ok_or("malformed multipart part")? + 2;
        let head_end = find(body, b"\r\n\r\n", at).ok_or("malformed multipart headers")?;
        let headers = String::from_utf8_lossy(&body[at..head_end]).to_string();
        let start = head_end + 4;
        let mut next_delim = b"\r\n".to_vec();
        next_delim.extend_from_slice(&delimiter);
        let end = find(body, &next_delim, start).ok_or("unterminated multipart part")?;
        let disposition = headers.lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-disposition:"))
            .ok_or("multipart part lacks Content-Disposition")?;
        let param = |key: &str| disposition.split(';').map(str::trim)
            .find_map(|p| p.strip_prefix(key)).map(|v| v.trim_matches('"').to_string());
        let name = param("name=").ok_or("multipart part has no name")?;
        parts.push((name, param("filename=").is_some(), body[start..end].to_vec()));
        at = end + next_delim.len();
    }
    Ok(parts)
}

/// An OpenAI-style edit request: multipart with an `image` file, or JSON with
/// a base64 (optionally data-URL) `image`. The reference is written beside the
/// output so the native pipeline can read it.
fn parse_edit(content_type: &str, body: &[u8], stem: &Path) -> Result<ImageRequest, String> {
    let (mut fields, image) = if content_type.to_ascii_lowercase().starts_with("multipart/form-data") {
        let mut fields = serde_json::Map::new();
        let mut image = None;
        for (name, _file, bytes) in multipart(content_type, body)? {
            if name == "image" || name == "image[]" {
                if image.replace(bytes).is_some() {
                    return Err("native editing accepts exactly one reference image".into());
                }
                continue;
            }
            let text = String::from_utf8(bytes).map_err(|_| format!("field {name} is not UTF-8"))?;
            let value = match name.as_str() {
                "steps" | "seed" | "n" => Value::from(text.trim().parse::<u64>()
                    .map_err(|_| format!("invalid {name}"))?),
                "guidance_scale" => Value::from(text.trim().parse::<f64>()
                    .map_err(|_| format!("invalid {name}"))?),
                _ => Value::from(text),
            };
            fields.insert(name, value);
        }
        (fields, image.ok_or("edit request needs an image file")?)
    } else {
        let mut value: Value = serde_json::from_slice(body).map_err(|e| e.to_string())?;
        let fields = value.as_object_mut().ok_or("edit request must be a JSON object")?;
        let data = fields.remove("image").and_then(|v| v.as_str().map(str::to_string))
            .ok_or("edit request needs a base64 image")?;
        let data = data.split_once(";base64,").map(|(_, d)| d.to_string()).unwrap_or(data);
        let image = base64::decode(data.trim()).map_err(|_| "image is not valid base64".to_string())?;
        (fields.clone(), image)
    };
    fields.remove("mask").map_or(Ok(()), |_| Err("masks are not supported".to_string()))?;
    let mut request: ImageRequest = serde_json::from_value(Value::Object(std::mem::take(&mut fields)))
        .map_err(|e| e.to_string())?;
    let path = stem.with_extension(image_extension(&image)?);
    fs::write(&path, &image).map_err(|e| format!("writing reference image: {e}"))?;
    request.image = Some(path);
    Ok(request)
}

pub fn serve(cfg: ImageConfig, host: &str, port: u16) -> Result<(), String> {
    let addr = format!("{host}:{port}");
    let server = Server::http(&addr).map_err(|e| format!("binding {addr}: {e}"))?;
    println!("phlegm image server on http://{addr}");
    println!("  POST /v1/images/generations | POST /v1/images/edits | GET /v1/models | GET /health");
    for request in server.incoming_requests() {
        handle(request, &cfg);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsupported_image_shapes_and_parameters() {
        let base = ImageRequest { prompt: "cube".into(), model: Some(MOSS_ID.into()),
            negative_prompt: String::new(), size: default_size(), steps: Some(50),
            guidance_scale: Some(3.0), seed: 41, n: 1, response_format: default_response_format(),
            image: None };
        assert_eq!(base.validate().unwrap(), 512);
        let mut wrong = base.clone();
        wrong.size = "768x512".into();
        assert!(wrong.validate().is_err());
        wrong = base.clone();
        wrong.model = Some("Qwen/Qwen-Image-2.1".into());
        assert!(wrong.validate().is_ok());
        wrong = base.clone();
        wrong.guidance_scale = Some(f32::NAN);
        assert!(wrong.validate().is_err());
    }

    #[test]
    fn qwen_image_request_contract() {
        let path = PathBuf::from("/unused");
        let cfg = ImageConfig::Qwen21(Qwen21ImageConfig {
            backend: "qwen21".into(), dibs: None, python: path.clone(),
            repo_root: path.clone(), env_script: path.clone(), model: path.clone(),
            dit_weights: path.clone(), text_model: path.clone(), text_kernels: path.clone(),
            text_binary: path.clone(), edge_kernels: path.clone(),
            block_kernels: path.clone(), kernel_probes: path.clone(),
            vae_weights: path.clone(), vae_conv_kernels: path.clone(),
            vae_ew_kernels: path.clone(), vae_attention_kernel: path.clone(),
            harness: path.clone(), output_root: path,
        });
        let mut request = ImageRequest { prompt: "red cube".into(), model: Some(QWEN21_ID.into()),
            negative_prompt: String::new(), size: default_size(), steps: Some(40),
            guidance_scale: None, seed: 41, n: 1, response_format: default_response_format(),
            image: None };
        assert_eq!(cfg.validate_request(&request).unwrap(), 512);
        request.steps = Some(1);
        assert!(cfg.validate_request(&request).is_err());
        request.steps = Some(51);
        assert!(cfg.validate_request(&request).is_err());
        request.steps = Some(40);
        request.guidance_scale = Some(3.0);
        assert!(cfg.validate_request(&request).is_err());
        request.guidance_scale = None;
        request.size = "1024x1024".into();
        assert!(cfg.validate_request(&request).is_err());
    }

    #[test]
    fn flux3_size_follows_configured_kernel_sets() {
        let path = PathBuf::from("/unused");
        let mut flux = Flux3ImageConfig {
            backend: "flux3".into(), dibs: None, python: path.clone(), repo_root: path.clone(),
            env_script: path.clone(), model: path.clone(), dit_weights: path.clone(),
            vae_weights: path.clone(), pup_model: path.clone(), pup_kernels: path.clone(),
            text_model: path.clone(), text_kernels: path.clone(), text_binary: path.clone(),
            condition_kernels: path.clone(), edge_kernels: BTreeMap::new(),
            image_block_kernels: path.clone(), text_block_kernels: BTreeMap::new(),
            shared_block_kernels: BTreeMap::new(), vae_kernels: path.clone(),
            harness: path.clone(), output_root: path.clone(), alpha: 16.0, pup_max_tokens: 512,
            vae_encoder_kernels: None, vae_encoder_weights: None, cond_block_kernels: Some(path.clone()),
            edit_shared_block_kernels: BTreeMap::new(), resolutions: BTreeMap::new(),
        };
        let mut request = ImageRequest { prompt: "red cube".into(), model: Some(FLUX3_ID.into()),
            negative_prompt: String::new(), size: "1024x1024".into(), steps: Some(16),
            guidance_scale: None, seed: 7, n: 1, response_format: default_response_format(),
            image: None };
        assert!(ImageConfig::Flux3(flux.clone()).validate_request(&request).is_err());
        flux.resolutions.insert("1024".into(), Flux3ResolutionKernels {
            edge_kernels: BTreeMap::new(), image_block_kernels: path.clone(),
            text_block_kernels: BTreeMap::new(), shared_block_kernels: BTreeMap::new(),
            vae_kernels: path.clone() });
        let cfg = ImageConfig::Flux3(flux);
        assert_eq!(cfg.validate_request(&request).unwrap(), 1024);
        let file = std::env::current_exe().unwrap();
        request.image = Some(file.clone());
        assert!(cfg.validate_request(&request).unwrap_err().contains("512x512"));
        request.size = default_size();
        assert_eq!(cfg.validate_request(&request).unwrap(), 512);
    }

    #[test]
    fn edit_requests_parse_multipart_and_json_and_gate_by_backend() {
        let root = std::env::temp_dir().join(format!("phlegm-edit-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let png = b"\x89PNG\r\n\x1a\nrest".to_vec();
        let mut body = Vec::new();
        for (name, file, value) in [("prompt", false, b"make it blue".to_vec()),
                                    ("steps", false, b"4".to_vec()),
                                    ("image", true, png.clone())] {
            body.extend_from_slice(b"--XyZ\r\nContent-Disposition: form-data; name=\"");
            body.extend_from_slice(name.as_bytes());
            body.extend_from_slice(if file { b"\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n" as &[u8] }
                                   else { b"\"\r\n\r\n" });
            body.extend_from_slice(&value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--XyZ--\r\n");
        let request = parse_edit("multipart/form-data; boundary=XyZ", &body, &root.join("m")).unwrap();
        assert_eq!(request.prompt, "make it blue");
        assert_eq!(request.steps, Some(4));
        let image = request.image.clone().unwrap();
        assert_eq!(fs::read(&image).unwrap(), png);
        let json = serde_json::to_vec(&json!({"prompt": "p", "image":
            format!("data:image/png;base64,{}", encode(&png))})).unwrap();
        assert!(parse_edit("application/json", &json, &root.join("j")).unwrap().image.is_some());
        let bad = serde_json::to_vec(&json!({"prompt": "p", "image": encode(b"GIF89a")})).unwrap();
        assert!(parse_edit("application/json", &bad, &root.join("b")).is_err());
        let path = PathBuf::from("/unused");
        let qwen = ImageConfig::Qwen21(Qwen21ImageConfig {
            backend: "qwen21".into(), dibs: None, python: path.clone(),
            repo_root: path.clone(), env_script: path.clone(), model: path.clone(),
            dit_weights: path.clone(), text_model: path.clone(), text_kernels: path.clone(),
            text_binary: path.clone(), edge_kernels: path.clone(),
            block_kernels: path.clone(), kernel_probes: path.clone(),
            vae_weights: path.clone(), vae_conv_kernels: path.clone(),
            vae_ew_kernels: path.clone(), vae_attention_kernel: path.clone(),
            harness: path.clone(), output_root: path.clone(),
        });
        let mut edit = request;
        edit.model = None;
        edit.steps = Some(40);
        assert!(qwen.validate_request(&edit).unwrap_err().contains("editing"));
        let kernels = root.join("moss-kernels");
        fs::create_dir_all(&kernels).unwrap();
        fs::write(kernels.join("manifest.json"), br#"{"resolution":512,"resolutions":[512]}"#).unwrap();
        let moss = ImageConfig::Moss(MossImageConfig {
            dibs: None, python: path.clone(), repo_root: path.clone(), env_script: path.clone(),
            model: path.clone(), dit_weights: path.clone(), vae_weights: path.clone(),
            tokenizer_dir: path.clone(), text_model: path.clone(), text_kernels: path.clone(),
            text_binary: path.clone(), refine_kernels: String::new(),
            condition_kernels: path.clone(), block_kernels: String::new(),
            edge_kernels: kernels.clone(), vae_kernels: kernels, harness: path.clone(),
            output_root: path.clone(),
        });
        let mut moss_edit = edit.clone();
        moss_edit.steps = Some(50);
        assert!(moss.validate_request(&moss_edit).unwrap_err().contains("does not support editing"));
        edit.image = None;
        assert!(qwen.validate_request(&edit).is_ok());
        moss_edit.image = None;
        assert!(moss.validate_request(&moss_edit).is_ok());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn flux2_sizes_and_edits_follow_the_kernel_marker() {
        let root = std::env::temp_dir().join(format!("phlegm-flux2-test-{}", std::process::id()));
        let kernels = root.join("kernels");
        fs::create_dir_all(&kernels).unwrap();
        fs::write(kernels.join("dit_kernels.json"),
                  br#"{"complete":true,"resolutions":[512,1024],"edits":[512]}"#).unwrap();
        let path = PathBuf::from("/unused");
        let flux = Flux2ImageConfig {
            backend: "flux2".into(), model_id: FLUX2_KLEIN_IDS[0].into(), dibs: None,
            python: path.clone(), repo_root: path.clone(), env_script: None,
            model: path.clone(), kernels, output_root: root.clone(),
        };
        let cfg = ImageConfig::Flux2(flux);
        assert_eq!(cfg.model_id(), "black-forest-labs/FLUX.2-klein-4B");
        let reference = root.join("ref.png");
        fs::write(&reference, b"\x89PNG").unwrap();
        let mut request = ImageRequest { prompt: "fox".into(), model: None,
            negative_prompt: String::new(), size: "1024x1024".into(), steps: None,
            guidance_scale: None, seed: 1, n: 1, response_format: default_response_format(),
            image: None };
        assert_eq!(cfg.validate_request(&request).unwrap(), 1024);
        request.image = Some(reference.clone());
        assert!(cfg.validate_request(&request).unwrap_err().contains("1024-pixel editing"));
        request.size = default_size();
        assert_eq!(cfg.validate_request(&request).unwrap(), 512);
        request.guidance_scale = Some(4.0);
        assert!(cfg.validate_request(&request).unwrap_err().contains("guidance_scale"));
        request.guidance_scale = None;
        request.steps = Some(51);
        assert!(cfg.validate_request(&request).is_err());
        request.steps = Some(4);
        request.model = Some(FLUX2_KLEIN_IDS[1].into());
        assert!(cfg.validate_request(&request).unwrap_err().contains("unsupported image model"));
        fs::remove_dir_all(&root).unwrap();
    }
}
