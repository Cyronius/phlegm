//! Temporary, resident C++ reference-engine adapter. Timings originate in the
//! child process, not in its JSON transport. The source checkout is never copied.
mod turbo;

use super::{is_hex, BenchBackend, ModelInfo, RuntimeInfo, Sample};
use crate::backend::{Backend, GenParams};
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

/// `accurate` is the lean prefill route by what it buys (OPEN-PREFILL-MODE); reports record `lean`.
fn canonical_prefill_mode(mode: &str) -> String {
    if mode == "accurate" {
        "lean".into()
    } else {
        mode.into()
    }
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

/// The bridge must not outlive phlegm. `Drop` kills it on a normal exit, but a SIGTERM or
/// SIGKILL ends phlegm without running it, and the bridge would hold the NPU until its current
/// request finished. On Linux the kernel sends it SIGKILL when the thread that spawned it ends
/// (PR_SET_PDEATHSIG), so spawn it from a thread that lives as long as the backend (`load` runs
/// on the caller's thread: main, for `run` and `serve`).
#[cfg(target_os = "linux")]
pub(crate) fn die_with_parent(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    extern "C" {
        fn prctl(option: std::ffi::c_int, ...) -> std::ffi::c_int;
        fn getppid() -> std::ffi::c_int;
    }
    const PR_SET_PDEATHSIG: std::ffi::c_int = 1;
    const SIGKILL: std::ffi::c_ulong = 9;
    let parent = std::process::id() as std::ffi::c_int;
    // SAFETY: between fork and exec; prctl and getppid are async-signal-safe and allocate nothing.
    unsafe {
        command.pre_exec(move || {
            if prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // The parent may have died before prctl took effect.
            if getppid() != parent {
                return Err(std::io::Error::other("phlegm exited while starting the bridge"));
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn die_with_parent(_: &mut Command) {}

struct Process {
    child: Child,
    input: ChildStdin,
    /// Each reply line, with the raw bytes that follow it when it announces
    /// `logits_f32_le_bytes` (the logits, binary32 little endian).
    replies: Receiver<Result<(Value, Vec<u8>), String>>,
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
        die_with_parent(&mut command);
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
                let result = (&mut reader).take(8_000_001).read_until(b'\n', &mut line);
                let value = match result {
                    Ok(0) => Err("reference bridge exited before replying".into()),
                    Ok(_) if line.len() > 8_000_000 => Err("reference reply exceeds 8 MB".into()),
                    Ok(_) => serde_json::from_slice::<Value>(&line)
                        .map_err(|e| format!("invalid reference reply: {e}"))
                        .and_then(|value| {
                            let n = value.get("logits_f32_le_bytes").and_then(Value::as_u64).unwrap_or(0);
                            if n > 8_000_000 {
                                return Err("reference logits exceed 8 MB".to_string());
                            }
                            let mut payload = vec![0u8; n as usize];
                            reader
                                .read_exact(&mut payload)
                                .map_err(|e| format!("reading reference logits: {e}"))?;
                            Ok((value, payload))
                        }),
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
        self.receive_with_payload().map(|(value, _)| value)
    }

    fn receive_with_payload(&mut self) -> Result<(Value, Vec<u8>), String> {
        let (value, payload) = self
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
        Ok((value, payload))
    }

    fn request(&mut self, value: Value) -> Result<Value, String> {
        self.request_with_payload(value).map(|(value, _)| value)
    }

    fn request_with_payload(&mut self, value: Value) -> Result<(Value, Vec<u8>), String> {
        serde_json::to_writer(&mut self.input, &value).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("writing reference request: {e}"))?;
        self.receive_with_payload()
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
    pub fn load(mut options: ReferenceOptions) -> Result<Self, String> {
        options.prefill_mode = canonical_prefill_mode(&options.prefill_mode);
        if options.threads == 0
            || options.request_timeout.is_zero()
            || options.dispatch_timeout_ms == 0
            || !["auto", "sequential", "block"].contains(&options.prefill_route.as_str())
            || !["fast", "lean"].contains(&options.prefill_mode.as_str())
        {
            return Err("invalid reference-engine settings".into());
        }
        let build = build_info(&options.bridge)?;
        crate::gguf::preflight_direct_route(&options.model_dir, &options.kernel_dir)?;
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
    fn accurate_is_the_lean_prefill_mode() {
        assert_eq!(canonical_prefill_mode("accurate"), "lean");
        assert_eq!(canonical_prefill_mode("lean"), "lean");
        assert_eq!(canonical_prefill_mode("fast"), "fast");
        assert_eq!(canonical_prefill_mode("other"), "other");
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

/// Resident generation backend using the same verified C++ bridge as Phlegm Bench.
/// Rust owns sampling and the device lock; the C++ child owns XRT resources.
pub struct NpuBackend {
    process: Process,
    context_capacity: usize,
    real_vocab: usize,
    _lock: NpuLock,
    /// SERVER-PROMPT-REUSE: whether the model has recurrent layers (None: reuse is off, or the
    /// bridge predates it), the tokens at the bridge's positions [0, len), its mark, and the
    /// last request's reused prompt tokens.
    recurrent: Option<bool>,
    context: Vec<u32>,
    mark: Option<usize>,
    reused: usize,
}

/// How a request's prompt starts on the bridge (SERVER-PROMPT-REUSE).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Start {
    Reset,
    /// The prompt extends everything the bridge holds.
    Append,
    /// Back to a position: the mark, or anywhere on a model without recurrent layers.
    Rewind(usize),
}

/// The longest reuse of `context` (the bridge's tokens) for `prompt`, leaving at least one token
/// to prefill (the logits come from it). Returns the start and how to get there.
fn reuse_start(context: &[u32], mark: Option<usize>, recurrent: Option<bool>, prompt: &[u32]) -> (usize, Start) {
    let Some(recurrent) = recurrent else { return (0, Start::Reset) };
    let shared = context.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let limit = shared.min(prompt.len().saturating_sub(1));
    let mut best = (0, Start::Reset);
    if shared == context.len() && shared == limit && shared > 0 {
        best = (shared, Start::Append);
    }
    if let Some(m) = mark.filter(|&m| m <= limit && m > best.0) {
        best = (m, Start::Rewind(m));
    }
    if !recurrent && limit > best.0 {
        best = (limit, Start::Rewind(limit));
    }
    best
}

pub struct NpuOptions {
    pub bridge: PathBuf,
    pub model_dir: PathBuf,
    pub kernel_dir: PathBuf,
    pub turbo: bool,
    pub context_capacity: usize,
    pub prefill_route: String,
    pub prefill_mode: String,
    pub threads: usize,
    pub request_timeout: Duration,
    pub dispatch_timeout_ms: u32,
}

impl NpuBackend {
    pub fn load(mut options: NpuOptions) -> Result<Self, String> {
        options.prefill_mode = canonical_prefill_mode(&options.prefill_mode);
        if options.threads == 0
            || options.request_timeout.is_zero()
            || options.dispatch_timeout_ms == 0
            || options.context_capacity < 2
            || !["auto", "sequential", "block"].contains(&options.prefill_route.as_str())
            || !["fast", "lean"].contains(&options.prefill_mode.as_str())
        {
            return Err("invalid NPU engine settings".into());
        }
        if options.turbo {
            options.kernel_dir = turbo::select(&options.kernel_dir, &options.prefill_route)?;
            eprintln!("phlegm: turbo enabled: experimental bfp16-hh precision, batched prefill ({})",
                      options.kernel_dir.display());
        }
        crate::gguf::preflight_direct_route(&options.model_dir, &options.kernel_dir)?;
        build_info(&options.bridge)?;
        let lock = NpuLock::acquire("phlegm NPU")?;
        let mut process =
            Process::spawn(&options.bridge, options.threads, options.request_timeout)?;
        let hello = process.request(json!({"command":"init", "model_dir":options.model_dir,
            "kernel_dir":options.kernel_dir, "max_ctx":options.context_capacity,
            "dispatch_timeout_ms":options.dispatch_timeout_ms, "prefill_route":options.prefill_route,
            "prefill_mode":options.prefill_mode}))?;
        if options.turbo {
            turbo::validate_loaded_route(hello.get("prefill_route").and_then(Value::as_str))?;
        }
        let context_capacity = hello
            .get("context_capacity")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("missing context capacity")?;
        let real_vocab = hello
            .get("real_vocab")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("missing real vocabulary")?;
        if context_capacity != options.context_capacity || real_vocab == 0 {
            return Err("NPU engine loaded unexpected capacity or vocabulary".into());
        }
        let reuse = std::env::var("PHLEGM_PROMPT_REUSE").map_or(true, |v| v != "0");
        let recurrent = hello
            .get("recurrent_layers")
            .and_then(Value::as_u64)
            .filter(|_| reuse)
            .map(|n| n > 0);
        Ok(Self {
            process,
            context_capacity,
            real_vocab,
            _lock: lock,
            recurrent,
            context: Vec::new(),
            mark: None,
            reused: 0,
        })
    }

    fn logits(&mut self, command: Value) -> Result<Vec<f32>, String> {
        let (_, bytes) = self.process.request_with_payload(command)?;
        self.logits_of(&bytes)
    }

    fn logits_of(&self, bytes: &[u8]) -> Result<Vec<f32>, String> {
        if bytes.len() != self.real_vocab * 4 {
            return Err(format!(
                "NPU engine returned {} logit bytes, expected {}",
                bytes.len(),
                self.real_vocab * 4
            ));
        }
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let f = f32::from_le_bytes(chunk.try_into().unwrap());
                if !f.is_finite() {
                    return Err("nonfinite logit".into());
                }
                Ok(f)
            })
            .collect()
    }
}

impl Backend for NpuBackend {
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        params: &mut GenParams,
        on_token: &mut dyn FnMut(crate::backend::TokenEvent<'_>) -> std::ops::ControlFlow<()>,
    ) -> Result<(), String> {
        self._lock.ensure_held()?;
        if prompt_ids.is_empty()
            || params.max_tokens == 0
            || prompt_ids
                .len()
                .checked_add(params.max_tokens - 1)
                .is_none_or(|n| n > self.context_capacity)
            || prompt_ids.iter().any(|&id| id as usize >= self.real_vocab)
        {
            return Err("prompt or output count exceeds the NPU model's context/vocabulary".into());
        }
        // Until this request's prefill succeeds the bridge's context is unknown: a failed
        // request leaves nothing to reuse.
        let context = std::mem::take(&mut self.context);
        let old_mark = self.mark.take();
        self.reused = 0;
        let (start, how) = reuse_start(&context, old_mark, self.recurrent, prompt_ids);
        let kept_mark = match how {
            Start::Reset => {
                self.process.request(json!({"command":"reset"}))?;
                None
            }
            Start::Append => old_mark,
            Start::Rewind(pos) => {
                self.process.request(json!({"command":"rewind", "pos":pos}))?;
                old_mark.filter(|&m| m <= pos)
            }
        };
        let mut prefill = json!({"command":"prefill_logits", "prompt":&prompt_ids[start..]});
        // A model with recurrent layers saves their state at the conversation's end (or the
        // prompt's), where the next turn's prompt will part from this one.
        let want_mark = (self.recurrent == Some(true)).then(|| {
            params.reuse_mark.filter(|&k| k > start && k <= prompt_ids.len()).unwrap_or(prompt_ids.len())
        });
        if let Some(k) = want_mark {
            prefill["mark"] = json!(k);
        }
        let (reply, bytes) = self.process.request_with_payload(prefill)?;
        let mut logits = self.logits_of(&bytes)?;
        self.mark = match want_mark {
            Some(_) => Some(
                reply
                    .get("mark")
                    .and_then(Value::as_u64)
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&m| m >= start && m <= prompt_ids.len())
                    .ok_or("NPU engine returned no mark")?,
            ),
            None => kept_mark,
        };
        self.context = prompt_ids.to_vec();
        self.reused = start;
        if self.recurrent.is_some() {
            eprintln!("phlegm: prompt {} tokens, {} reused", prompt_ids.len(), start);
        }
        let mut history: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
        for index in 0..params.max_tokens {
            let next = params.sampler.try_sample(&logits, &history)? as u32;
            if params.stop_ids.contains(&next) {
                break;
            }
            if on_token(crate::backend::TokenEvent { id: next, logits: Some(&logits) }).is_break() { break; }
            history.push(i64::from(next));
            if index + 1 < params.max_tokens {
                let context = std::mem::take(&mut self.context);
                logits = self.logits(json!({"command":"decode_logits", "token":next}))?;
                self.context = context;
                self.context.push(next);
            }
        }
        Ok(())
    }

    fn context_capacity(&self) -> Option<usize> {
        Some(self.context_capacity)
    }

    fn cached_tokens(&self) -> usize {
        self.reused
    }
}

#[cfg(all(test, unix))]
mod generation_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn resident_bridge_resets_and_stops_across_requests() {
        let dir = std::env::temp_dir().join(format!("phlegm-bridge-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bridge = dir.join("fake-bridge");
        fs::write(
            &bridge,
            r#"#!/usr/bin/env python3
import json, struct, sys
print(json.dumps({'ok': True, 'protocol': 1, 'engine': 'openflowlm-cpp-reference'}), flush=True)
phase = 0
logits = b''
for line in sys.stdin:
    r = json.loads(line)
    cmd = r['command']
    if cmd == 'init':
        out = {'context_capacity': r['max_ctx'], 'real_vocab': 5}
    elif cmd == 'reset':
        phase = 0
        out = {}
    elif cmd == 'prefill_logits':
        assert phase == 0 and r['prompt'] == [0, 1]
        phase = 1
        out, logits = {}, struct.pack('<5f', 0, 0, 10, 0, 0)
    elif cmd == 'decode_logits':
        assert phase == 1 and r['token'] == 2
        phase = 2
        out, logits = {}, struct.pack('<5f', 0, 0, 0, 0, 10)
    elif cmd == 'quit':
        print(json.dumps({'ok': True}), flush=True)
        break
    else:
        raise RuntimeError(cmd)
    out['ok'] = True
    if logits:
        out['logits_f32_le_bytes'] = len(logits)
    sys.stdout.write(json.dumps(out) + '\n'); sys.stdout.flush()
    sys.stdout.buffer.write(logits); sys.stdout.buffer.flush()
    logits = b''
"#,
        )
        .unwrap();
        let mut mode = fs::metadata(&bridge).unwrap().permissions();
        mode.set_mode(0o700);
        fs::set_permissions(&bridge, mode).unwrap();
        let adapter_sha = format!("{:x}", Sha256::digest(BRIDGE_SOURCE));
        let files = BTreeMap::from([("bridge/bridge.cpp".into(), adapter_sha.clone())]);
        let sidecar = json!({"protocol":1, "engine":BACKEND_NAME, "source_commit":"0".repeat(40),
            "source_dirty":false, "source_tree_sha256":manifest_hash(&files),
            "adapter_sha256":adapter_sha, "binary_sha256":hash_file(&bridge).unwrap(),
            "compiler_flags":"fake test bridge", "source_files":files});
        fs::write(bridge.with_extension("build.json"), sidecar.to_string()).unwrap();
        let opts = NpuOptions {
            bridge,
            model_dir: dir.clone(),
            kernel_dir: dir.clone(),
            turbo: false,
            context_capacity: 8,
            prefill_route: "auto".into(),
            prefill_mode: "fast".into(),
            threads: 1,
            request_timeout: Duration::from_secs(5),
            dispatch_timeout_ms: 1000,
        };
        let mut backend = NpuBackend::load(opts).unwrap();
        for _ in 0..2 {
            let mut params = GenParams {
                max_tokens: 4,
                sampler: crate::sampler::Sampler::new(0.0, 0, 1.0, 1.0, None),
                stop_ids: [4u32].into_iter().collect(),
                reuse_mark: None,
            };
            let mut output = Vec::new();
            backend
                .generate(&[0, 1], &mut params, &mut |event| { output.push(event.id); std::ops::ControlFlow::Continue(()) })
                .unwrap();
            assert_eq!(output, vec![2]);
        }
        drop(backend);
        fs::remove_dir_all(dir).unwrap();
    }

    /// A fake bridge with a recurrent layer: it tracks its position and mark the way bridge.cpp
    /// does and logs every command, so the test sees what phlegm asks of it across turns.
    #[test]
    fn resident_bridge_reuses_the_shared_prompt() {
        let dir = std::env::temp_dir().join(format!("phlegm-reuse-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let bridge = dir.join("fake-bridge");
        let log = dir.join("log.jsonl");
        fs::write(
            &bridge,
            format!(
                r#"#!/usr/bin/env python3
import json, struct, sys
print(json.dumps({{'ok': True, 'protocol': 1, 'engine': 'openflowlm-cpp-reference'}}), flush=True)
log = open({log:?}, 'a')
pos, mark, T = 0, None, 4
LOGITS = struct.pack('<6f', 0, 0, 0, 0, 0, 10)
for line in sys.stdin:
    logits = b''
    r = json.loads(line)
    cmd = r['command']
    log.write(json.dumps({{k: v for k, v in r.items() if k != 'model_dir' and k != 'kernel_dir'}}) + '\n'); log.flush()
    if cmd == 'init':
        out = {{'context_capacity': r['max_ctx'], 'real_vocab': 6, 'recurrent_layers': 1}}
    elif cmd == 'reset':
        pos, mark, out = 0, None, {{}}
    elif cmd == 'rewind':
        assert r['pos'] == mark, (r, mark)
        pos, out = mark, {{'position': mark}}
    elif cmd == 'prefill_logits':
        at = r['mark'] - pos
        at -= at % T
        mark = pos + at
        pos += len(r['prompt'])
        out, logits = {{'mark': mark}}, LOGITS
    elif cmd == 'decode_logits':
        pos += 1
        out, logits = {{}}, LOGITS
    else:
        raise RuntimeError(cmd)
    out['ok'] = True
    if logits:
        out['logits_f32_le_bytes'] = len(logits)
    sys.stdout.write(json.dumps(out) + '\n'); sys.stdout.flush()
    sys.stdout.buffer.write(logits); sys.stdout.buffer.flush()
"#
            ),
        )
        .unwrap();
        let mut mode = fs::metadata(&bridge).unwrap().permissions();
        mode.set_mode(0o700);
        fs::set_permissions(&bridge, mode).unwrap();
        let adapter_sha = format!("{:x}", Sha256::digest(BRIDGE_SOURCE));
        let files = BTreeMap::from([("bridge/bridge.cpp".into(), adapter_sha.clone())]);
        let sidecar = json!({"protocol":1, "engine":BACKEND_NAME, "source_commit":"0".repeat(40),
            "source_dirty":false, "source_tree_sha256":manifest_hash(&files),
            "adapter_sha256":adapter_sha, "binary_sha256":hash_file(&bridge).unwrap(),
            "compiler_flags":"fake test bridge", "source_files":files});
        fs::write(bridge.with_extension("build.json"), sidecar.to_string()).unwrap();
        let opts = NpuOptions {
            bridge,
            model_dir: dir.clone(),
            kernel_dir: dir.clone(),
            turbo: false,
            context_capacity: 64,
            prefill_route: "auto".into(),
            prefill_mode: "fast".into(),
            threads: 1,
            request_timeout: Duration::from_secs(5),
            dispatch_timeout_ms: 1000,
        };
        let mut backend = NpuBackend::load(opts).unwrap();
        let mut turn = |prompt: &[u32], mark: usize| {
            let mut params = GenParams {
                max_tokens: 3,
                sampler: crate::sampler::Sampler::new(0.0, 0, 1.0, 1.0, None),
                stop_ids: std::collections::HashSet::new(),
                reuse_mark: Some(mark),
            };
            let mut output = Vec::new();
            backend.generate(prompt, &mut params, &mut |event| { output.push(event.id); std::ops::ControlFlow::Continue(()) }).unwrap();
            assert_eq!(output, vec![5, 5, 5]);
            backend.cached_tokens()
        };
        // Turn 1: 10 messages tokens + a 2-token generation prompt; mark asked at 10, lands on 8.
        let first: Vec<u32> = [vec![1; 10], vec![2, 3]].concat();
        assert_eq!(turn(&first, 10), 0);
        // Turn 2 re-renders turn 1's reply differently after token 11: back to the mark at 8.
        let second: Vec<u32> = [vec![1; 10], vec![2, 4, 4], vec![1; 7], vec![2, 3]].concat();
        assert_eq!(turn(&second, 20), 8);
        // Turn 3 extends everything the bridge holds (turn 2's prompt and its two fed tokens).
        let third: Vec<u32> = [second.clone(), vec![5, 5, 1, 2, 3]].concat();
        assert_eq!(turn(&third, third.len() - 2), second.len() + 2);
        // A different conversation starts over.
        assert_eq!(turn(&[3, 3, 3], 2), 0);
        let commands: Vec<Value> = fs::read_to_string(&log)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .filter(|v| v["command"] != "decode_logits" && v["command"] != "init")
            .map(|mut v| {
                if let Some(p) = v.get("prompt") {
                    v["prompt"] = json!(p.as_array().unwrap().len());
                }
                v
            })
            .collect();
        assert_eq!(
            commands,
            vec![
                json!({"command":"reset"}),
                json!({"command":"prefill_logits", "prompt":12, "mark":10}),
                json!({"command":"rewind", "pos":8}),
                json!({"command":"prefill_logits", "prompt":14, "mark":20}),
                json!({"command":"prefill_logits", "prompt":3, "mark":third.len() - 2}),
                json!({"command":"reset"}),
                json!({"command":"prefill_logits", "prompt":3, "mark":2}),
            ]
        );
        // Stop after the second callback: only the first token was run. The
        // callback's final verified token must not be added to the kept context.
        let before = fs::read_to_string(&log).unwrap().lines().count();
        let early = vec![3, 3, 3, 5, 5, 1, 2];
        let mut params = GenParams {
            max_tokens: 8,
            sampler: crate::sampler::Sampler::new(0.0, 0, 1.0, 1.0, None),
            stop_ids: Default::default(), reuse_mark: Some(6),
        };
        let mut seen = 0;
        backend.generate(&early, &mut params, &mut |event| {
            assert_eq!(event.id, 5);
            assert_eq!(event.logits.unwrap(), &[0., 0., 0., 0., 0., 10.]);
            seen += 1;
            if seen == 2 { std::ops::ControlFlow::Break(()) } else { std::ops::ControlFlow::Continue(()) }
        }).unwrap();
        assert_eq!(seen, 2);
        assert_eq!(backend.context, [early.clone(), vec![5]].concat());
        assert_eq!(backend.mark, Some(5)); // mark rounded by the bridge relative to reuse start
        let follow = [early.clone(), vec![5, 5, 1]].concat();
        params.max_tokens = 1;
        params.reuse_mark = Some(follow.len());
        backend.generate(&follow, &mut params, &mut |_| std::ops::ControlFlow::Break(())).unwrap();
        assert_eq!(backend.cached_tokens(), early.len() + 1);
        assert_eq!(backend.context, follow);
        let added: Vec<Value> = fs::read_to_string(&log).unwrap().lines().skip(before)
            .map(|s| serde_json::from_str(s).unwrap()).collect();
        assert_eq!(added, vec![
            json!({"command":"prefill_logits", "prompt":[1,2], "mark":6}),
            json!({"command":"decode_logits", "token":5}),
            json!({"command":"prefill_logits", "prompt":[5,1], "mark":10}),
        ]);
        drop(backend);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reuse_start_takes_the_longest_valid_start() {
        let ctx = [1, 2, 3, 4, 5, 6];
        // without reuse, always a reset
        assert_eq!(reuse_start(&ctx, Some(4), None, &[1, 2, 3, 4, 5, 6, 7]), (0, Start::Reset));
        // extends the context: append, even past the mark
        assert_eq!(reuse_start(&ctx, Some(4), Some(true), &[1, 2, 3, 4, 5, 6, 7]), (6, Start::Append));
        // parts after the mark: back to it; before it: nothing to reuse on a recurrent model
        assert_eq!(reuse_start(&ctx, Some(4), Some(true), &[1, 2, 3, 4, 5, 9]), (4, Start::Rewind(4)));
        assert_eq!(reuse_start(&ctx, Some(4), Some(true), &[1, 2, 3, 9]), (0, Start::Reset));
        // attention only: back to wherever the prompts part
        assert_eq!(reuse_start(&ctx, None, Some(false), &[1, 2, 3, 9]), (3, Start::Rewind(3)));
        // the same prompt again keeps one token to prefill
        assert_eq!(reuse_start(&ctx, None, Some(false), &ctx), (5, Start::Rewind(5)));
        assert_eq!(reuse_start(&ctx, Some(6), Some(true), &ctx), (0, Start::Reset));
        assert_eq!(reuse_start(&ctx, Some(4), Some(true), &ctx), (4, Start::Rewind(4)));
    }
}
