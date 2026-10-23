//! Temporary, resident C++ reference-engine adapter. Timings originate in the
//! child process, not in its JSON transport. The source checkout is never copied.
use super::{is_hex, BenchBackend, ModelInfo, RuntimeInfo, Sample};
use crate::npu_lock::NpuLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    time::Duration,
};

pub const BACKEND_NAME: &str = "openflowlm-cpp-reference";
const BRIDGE_SOURCE: &[u8] = include_bytes!("../../../benchmarks/reference/bridge.cpp");

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceInfo {
    pub engine: String,
    pub source_commit: String,
    pub source_dirty: bool,
    pub source_tree_sha256: String,
    pub binary_sha256: String,
    pub compiler_flags: String,
    pub prefill_route: String,
    pub prefill_mode: String,
    pub decode_route: String,
    pub npu_power_mode: String,
    pub environment: BTreeMap<String, String>,
    pub weights_file: String,
    pub model_config_sha256: Option<String>,
}

impl ReferenceInfo {
    pub fn validate(&self) -> Result<(), String> {
        if self.engine != BACKEND_NAME
            || !is_hex(&self.source_commit, 40)
            || !is_hex(&self.source_tree_sha256, 64)
            || !is_hex(&self.binary_sha256, 64)
            || self.compiler_flags.trim().is_empty()
            || !["sequential", "gemm-block", "layer-major"].contains(&self.prefill_route.as_str())
            || !["fast", "lean"].contains(&self.prefill_mode.as_str())
            || self.decode_route != "npu"
            || self.npu_power_mode.trim().is_empty()
            || !["model.gguf", "model.q4nx"].contains(&self.weights_file.as_str())
            || self.environment.get("OMP_WAIT_POLICY").map(String::as_str) != Some("PASSIVE")
            || self
                .model_config_sha256
                .as_ref()
                .is_some_and(|s| !is_hex(s, 64))
        {
            return Err("incomplete or unsupported C++ reference-engine provenance".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildInfo {
    protocol: u32,
    engine: String,
    source_commit: String,
    source_dirty: bool,
    source_tree_sha256: String,
    adapter_sha256: String,
    binary_sha256: String,
    compiler_flags: String,
    source_files: BTreeMap<String, String>,
}

pub struct ReferenceOptions {
    pub bridge: PathBuf,
    pub model_dir: PathBuf,
    pub kernel_dir: PathBuf,
    pub model_id: String,
    pub model_revision: String,
    pub quantization: String,
    pub driver_version: String,
    pub context_capacity: usize,
    pub prefill_route: String,
    pub prefill_mode: String,
    pub threads: usize,
    pub xrt_smi: PathBuf,
    pub request_timeout: Duration,
    pub dispatch_timeout_ms: u32,
}

pub fn hash_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn manifest_hash(files: &BTreeMap<String, String>) -> String {
    let mut hash = Sha256::new();
    for (name, digest) in files {
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(digest.as_bytes());
        hash.update(b"\n");
    }
    format!("{:x}", hash.finalize())
}

fn build_info(bridge: &Path) -> Result<BuildInfo, String> {
    let path = bridge.with_extension("build.json");
    let data = fs::read(&path)
        .map_err(|e| format!("{}: {e}; build the reference bridge first", path.display()))?;
    if data.len() > 8_000_000 {
        return Err("reference build metadata exceeds 8 MB".into());
    }
    let build: BuildInfo =
        serde_json::from_slice(&data).map_err(|e| format!("reference build metadata: {e}"))?;
    if build.protocol != 1
        || build.engine != BACKEND_NAME
        || !is_hex(&build.source_commit, 40)
        || build.compiler_flags.trim().is_empty()
        || build.source_files.values().any(|s| !is_hex(s, 64))
        || build.source_files.get("bridge/bridge.cpp") != Some(&build.adapter_sha256)
        || manifest_hash(&build.source_files) != build.source_tree_sha256
        || format!("{:x}", Sha256::digest(BRIDGE_SOURCE)) != build.adapter_sha256
        || hash_file(bridge)? != build.binary_sha256
    {
        return Err("reference bridge provenance does not match this Phlegm build or its binary; rebuild the bridge".into());
    }
    Ok(build)
}

fn add_kernel_files(
    value: &Value,
    root: &Path,
    files: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let mut names = Vec::new();
    if let Some(contexts) = value.get("contexts").and_then(Value::as_object) {
        for path in contexts.values() {
            names.push(path.as_str().ok_or("invalid kernel context path")?);
        }
    }
    if let Some(kernels) = value.get("kernels").and_then(Value::as_object) {
        for kernel in kernels.values() {
            names.push(
                kernel
                    .get("insts")
                    .and_then(Value::as_str)
                    .ok_or("missing instruction path")?,
            );
        }
    }
    for name in names {
        let path = root
            .join(name)
            .canonicalize()
            .map_err(|e| format!("kernel artifact {name}: {e}"))?;
        if !path.starts_with(root) {
            return Err("kernel artifact path escapes the kernel directory".into());
        }
        files.insert(name.replace('\\', "/"), hash_file(&path)?);
    }
    if let Some(gguf) = value.get("gguf") {
        add_kernel_files(gguf, root, files)?;
    }
    Ok(())
}

pub fn kernel_hash(root: &Path) -> Result<String, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    let path = root.join("manifest.json");
    let value: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut files = BTreeMap::from([("manifest.json".into(), hash_file(&path)?)]);
    add_kernel_files(&value, &root, &mut files)?;
    if files.len() < 2 {
        return Err("kernel manifest names no artifacts".into());
    }
    Ok(manifest_hash(&files))
}

fn power_mode(tool: &Path) -> Result<String, String> {
    let output = Command::new(tool)
        .args(["examine", "-r", "platform", "--batch"])
        .output()
        .map_err(|e| format!("reading NPU power mode: {e}"))?;
    if !output.status.success() {
        return Err("xrt-smi could not read NPU power mode".into());
    }
    parse_power_mode(&String::from_utf8_lossy(&output.stdout))
}

fn parse_power_mode(report: &str) -> Result<String, String> {
    report
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim().eq_ignore_ascii_case("Power Mode") && !value.trim().is_empty())
                .then(|| value.trim().to_owned())
        })
        .ok_or_else(|| "xrt-smi platform report has no power mode".into())
}

struct Process {
    child: Child,
    input: ChildStdin,
    replies: Receiver<Result<Value, String>>,
    timeout: Duration,
}

impl Process {
    fn spawn(bridge: &Path, threads: usize, timeout: Duration) -> Result<Self, String> {
        let mut command = Command::new(bridge);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // Inherited tuning/probe switches must not silently alter a measurement.
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy().to_ascii_uppercase();
            if name.starts_with("OFLM_") || name.starts_with("OMP_") {
                command.env_remove(key);
            }
        }
        command
            .env("OMP_WAIT_POLICY", "PASSIVE")
            .env("OMP_NUM_THREADS", threads.to_string());
        let mut child = command
            .spawn()
            .map_err(|e| format!("starting reference bridge: {e}"))?;
        let input = child.stdin.take().ok_or("missing bridge stdin")?;
        let output = child.stdout.take().ok_or("missing bridge stdout")?;
        let (sender, replies) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                let result = (&mut reader).take(1_000_001).read_until(b'\n', &mut line);
                let value = match result {
                    Ok(0) => Err("reference bridge exited before replying".into()),
                    Ok(_) if line.len() > 1_000_000 => Err("reference reply exceeds 1 MB".into()),
                    Ok(_) => serde_json::from_slice::<Value>(&line)
                        .map_err(|e| format!("invalid reference reply: {e}")),
                    Err(e) => Err(format!("reading reference reply: {e}")),
                };
                let failed = value.is_err();
                if sender.send(value).is_err() || failed {
                    break;
                }
            }
        });
        let mut process = Self {
            child,
            input,
            replies,
            timeout,
        };
        let hello = process.receive()?;
        if hello.get("protocol").and_then(Value::as_u64) != Some(1)
            || hello.get("engine").and_then(Value::as_str) != Some(BACKEND_NAME)
        {
            return Err("unsupported reference bridge protocol".into());
        }
        Ok(process)
    }

    fn receive(&mut self) -> Result<Value, String> {
        let value = self
            .replies
            .recv_timeout(self.timeout)
            .map_err(|e| format!("reference bridge reply failed or timed out: {e}"))??;
        if value.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("reference bridge failed")
                .to_owned());
        }
        Ok(value)
    }

    fn request(&mut self, value: Value) -> Result<Value, String> {
        serde_json::to_writer(&mut self.input, &value).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("writing reference request: {e}"))?;
        self.receive()
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Also handles timeouts/device failures; never wait indefinitely for a
        // hung driver's graceful shutdown. The child owns every XRT handle.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ReferenceBackend {
    process: Process,
    model: ModelInfo,
    runtime: RuntimeInfo,
    tokenizer: tokenizers::Tokenizer,
    options: ReferenceOptions,
    _lock: NpuLock,
}

impl ReferenceBackend {
    pub fn load(options: ReferenceOptions) -> Result<Self, String> {
        if options.threads == 0
            || options.request_timeout.is_zero()
            || options.dispatch_timeout_ms == 0
            || !["auto", "sequential", "block"].contains(&options.prefill_route.as_str())
            || !["fast", "lean"].contains(&options.prefill_mode.as_str())
        {
            return Err("invalid reference-engine settings".into());
        }
        let build = build_info(&options.bridge)?;
        let tokenizer_path = options.model_dir.join("tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        let weight_name = if options.model_dir.join("model.q4nx").is_file() {
            "model.q4nx"
        } else {
            "model.gguf"
        };
        eprintln!("Hashing model, tokenizer, and kernels (outside benchmark timers)...");
        let model = ModelInfo {
            id: options.model_id.clone(),
            revision: options.model_revision.clone(),
            quantization: options.quantization.clone(),
            weights_sha256: hash_file(&options.model_dir.join(weight_name))?,
            tokenizer_sha256: hash_file(&tokenizer_path)?,
        };
        let kernels = kernel_hash(&options.kernel_dir)?;
        let model_config_sha256 = optional_hash(&options.model_dir.join("config.json"))?;
        let initial_power = power_mode(&options.xrt_smi)?;
        let lock = NpuLock::acquire("phlegm bench reference")?;
        let mut process =
            Process::spawn(&options.bridge, options.threads, options.request_timeout)?;
        let hello = process.request(json!({"command":"init", "model_dir":options.model_dir,
            "kernel_dir":options.kernel_dir, "max_ctx":options.context_capacity,
            "dispatch_timeout_ms":options.dispatch_timeout_ms, "prefill_route":options.prefill_route,
            "prefill_mode":options.prefill_mode}))?;
        if hello.get("weights_file").and_then(Value::as_str) != Some(weight_name)
            || hello.get("context_capacity").and_then(Value::as_u64)
                != Some(options.context_capacity as u64)
        {
            return Err(
                "reference engine loaded different weights or context capacity than requested"
                    .into(),
            );
        }
        let reference = ReferenceInfo {
            engine: BACKEND_NAME.into(),
            source_commit: build.source_commit,
            source_dirty: build.source_dirty,
            source_tree_sha256: build.source_tree_sha256,
            binary_sha256: build.binary_sha256,
            compiler_flags: build.compiler_flags,
            prefill_route: hello
                .get("prefill_route")
                .and_then(Value::as_str)
                .ok_or("missing prefill route")?
                .into(),
            prefill_mode: options.prefill_mode.clone(),
            decode_route: "npu".into(),
            npu_power_mode: initial_power,
            environment: BTreeMap::from([
                ("OMP_WAIT_POLICY".into(), "PASSIVE".into()),
                ("OMP_NUM_THREADS".into(), options.threads.to_string()),
            ]),
            weights_file: weight_name.into(),
            model_config_sha256,
        };
        reference.validate()?;
        let runtime = RuntimeInfo {
            backend: BACKEND_NAME.into(),
            device_kind: super::DeviceKind::Npu,
            device: hello
                .get("device")
                .and_then(Value::as_str)
                .ok_or("missing device name")?
                .into(),
            driver_version: options.driver_version.clone(),
            kernel_sha256: kernels,
            context_capacity: options.context_capacity,
            reference: Some(reference),
        };
        Ok(Self {
            process,
            model,
            runtime,
            tokenizer,
            options,
            _lock: lock,
        })
    }

    pub fn finish(&mut self) -> Result<(), String> {
        self.process.request(json!({"command":"quit"}))?;
        let weights = self
            .runtime
            .reference
            .as_ref()
            .ok_or("missing reference provenance")?;
        if hash_file(&self.options.model_dir.join(&weights.weights_file))?
            != self.model.weights_sha256
            || hash_file(&self.options.model_dir.join("tokenizer.json"))?
                != self.model.tokenizer_sha256
            || kernel_hash(&self.options.kernel_dir)? != self.runtime.kernel_sha256
            || optional_hash(&self.options.model_dir.join("config.json"))?
                != weights.model_config_sha256
        {
            return Err(
                "model, tokenizer, or kernels changed during the run; discard results".into(),
            );
        }
        Ok(())
    }

    fn check_power(&self) -> Result<(), String> {
        self._lock.ensure_held()?;
        let current = power_mode(&self.options.xrt_smi)?;
        if self.runtime.reference.as_ref().map(|r| &r.npu_power_mode) != Some(&current) {
            return Err("NPU power mode changed during the benchmark; discard this run".into());
        }
        Ok(())
    }
}

fn optional_hash(path: &Path) -> Result<Option<String>, String> {
    if path.is_file() {
        hash_file(path).map(Some)
    } else {
        Ok(None)
    }
}

fn token(reply: Value) -> Result<u32, String> {
    reply
        .get("token")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| "missing output token".into())
}

fn trial_reply(reply: Value, prompt: usize, outputs: usize) -> Result<Sample, String> {
    if reply.get("prompt_tokens").and_then(Value::as_u64) != Some(prompt as u64)
        || reply.get("output_tokens").and_then(Value::as_u64) != Some(outputs as u64)
        || reply.get("decode_steps").and_then(Value::as_u64) != Some((outputs - 1) as u64)
    {
        return Err("reference engine returned incomplete token counts".into());
    }
    let sample: Sample =
        serde_json::from_value(reply.get("sample").cloned().ok_or("missing timings")?)
            .map_err(|e| e.to_string())?;
    if sample.decode_token_ms.len() != outputs - 1
        || sample.prefill_ms <= 0.0
        || sample.decode_ms <= 0.0
        || sample
            .decode_token_ms
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid reference-engine timings".into());
    }
    Ok(sample)
}

impl BenchBackend for ReferenceBackend {
    fn model_info(&self) -> Result<ModelInfo, String> {
        Ok(self.model.clone())
    }
    fn runtime_info(&self) -> Result<RuntimeInfo, String> {
        Ok(self.runtime.clone())
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }
    fn reset(&mut self) -> Result<(), String> {
        self.process.request(json!({"command":"reset"}))?;
        Ok(())
    }
    fn prefill(&mut self, prompt: &[u32]) -> Result<u32, String> {
        token(
            self.process
                .request(json!({"command":"prefill", "prompt":prompt}))?,
        )
    }
    fn decode(&mut self, previous_token: u32) -> Result<u32, String> {
        token(
            self.process
                .request(json!({"command":"decode", "token":previous_token}))?,
        )
    }
    fn measure_trial(&mut self, prompt: &[u32], output_tokens: usize) -> Result<Sample, String> {
        if !(2..=4096).contains(&output_tokens) {
            return Err("invalid output count".into());
        }
        self.check_power()?;
        eprintln!(
            "Reference trial: {} input tokens, {} output tokens",
            prompt.len(),
            output_tokens
        );
        let result = trial_reply(
            self.process
                .request(json!({"command":"trial", "prompt":prompt,
            "output_tokens":output_tokens}))?,
            prompt.len(),
            output_tokens,
        )?;
        self.check_power()?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the compiled C++ reference bridge; no NPU model is loaded"]
    fn compiled_bridge_handshake_and_provenance() {
        let bridge = PathBuf::from(
            std::env::var_os("PHLEGM_REFERENCE_BRIDGE").expect("set PHLEGM_REFERENCE_BRIDGE"),
        );
        build_info(&bridge).unwrap();
        let mut process = Process::spawn(&bridge, 2, Duration::from_secs(10)).unwrap();
        process.request(json!({"command":"quit"})).unwrap();
        let mut process = Process::spawn(&bridge, 2, Duration::from_secs(10)).unwrap();
        assert!(process
            .request(json!({"command":"trial", "prompt":[1], "output_tokens":2}))
            .unwrap_err()
            .contains("init is required"));
    }

    #[test]
    #[ignore = "requires the compiled bridge, local model/kernels and an available NPU; exports no benchmark"]
    fn reference_model_smoke() {
        let path =
            |name| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")));
        let options = ReferenceOptions {
            bridge: path("PHLEGM_REFERENCE_BRIDGE"),
            model_dir: path("PHLEGM_REFERENCE_MODEL"),
            kernel_dir: path("PHLEGM_REFERENCE_KERNELS"),
            model_id: "smoke-check-only".into(),
            model_revision: "smoke-check-only".into(),
            quantization: "smoke-check-only".into(),
            driver_version: "smoke-check-only".into(),
            context_capacity: 64,
            prefill_route: "auto".into(),
            prefill_mode: "fast".into(),
            threads: 2,
            xrt_smi: PathBuf::from("C:/Windows/System32/AMD/xrt-smi.exe"),
            request_timeout: Duration::from_secs(300),
            dispatch_timeout_ms: 60000,
        };
        let mut backend = ReferenceBackend::load(options).unwrap();
        let ids = backend.encode("A short test prompt.").unwrap();
        assert!(!ids.is_empty() && ids.len() < 32);
        backend.reset().unwrap();
        let first = backend.prefill(&ids).unwrap();
        let sample = backend.measure_trial(&ids, 8).unwrap();
        assert_eq!(sample.decode_token_ms.len(), 7);
        backend.reset().unwrap();
        assert_eq!(backend.prefill(&ids).unwrap(), first);
        backend.finish().unwrap();
    }

    #[test]
    fn reference_trial_checks_exact_counts_and_keeps_engine_timings() {
        let reply = json!({"prompt_tokens":512,"output_tokens":4,"decode_steps":3,
            "sample":{"prefill_ms":123.0,"decode_ms":9.0,"decode_token_ms":[2.0,3.0,4.0]}});
        let sample = trial_reply(reply.clone(), 512, 4).unwrap();
        assert_eq!(sample.prefill_ms, 123.0);
        assert_eq!(sample.decode_ms, 9.0);
        assert!(trial_reply(reply.clone(), 1024, 4).is_err());
        assert!(trial_reply(reply, 512, 128).is_err());
    }

    #[test]
    fn power_mode_parser_requires_an_explicit_mode() {
        assert_eq!(
            parse_power_mode("Platform\n  Power Mode : Turbo\n").unwrap(),
            "Turbo"
        );
        assert!(parse_power_mode("Power Mode : \n").is_err());
        assert!(parse_power_mode("Platform unavailable").is_err());
    }

    #[test]
    fn sha256_uses_standard_vectors_and_manifest_order_is_stable() {
        assert_eq!(
            format!("{:x}", Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let a = BTreeMap::from([("b".into(), "2".into()), ("a".into(), "1".into())]);
        assert_eq!(
            manifest_hash(&a),
            format!("{:x}", Sha256::digest(b"a\x001\nb\x002\n"))
        );
    }
}
