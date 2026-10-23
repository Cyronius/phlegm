//! Versioned, batch-one inference benchmark. No HTTP/chunk-arrival estimates:
//! an engine adapter must expose synchronous prefill and individual decode steps.
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub mod machine;
pub mod reference;

pub const INPUT_LENGTHS: [usize; 4] = [512, 1024, 2048, 4096];
pub const BENCHMARK_VERSION: &str = "phlegm-bench-v1";
const MAX_REPORT_BYTES: usize = 16_000_000;
pub const PROMPT: &str = "The research team records observations in a shared notebook. \
Each entry describes the experiment, the equipment, and the measured outcome. \
They repeat each measurement under the same conditions and compare the results. \
When an observation differs, they examine the evidence before drawing conclusions. \
Clear records allow another team to reproduce the work and explain the differences.\n";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseInfo {
    pub version: String,
    pub tag: String,
    pub commit: String,
    pub dirty: bool,
    pub opt_level: String,
    pub target: String,
    pub rustc: String,
    pub features: Vec<String>,
}

impl ReleaseInfo {
    pub fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").into(),
            tag: env!("PHLEGM_BUILD_TAG").into(),
            commit: env!("PHLEGM_BUILD_COMMIT").into(),
            dirty: env!("PHLEGM_BUILD_DIRTY") != "false",
            opt_level: env!("PHLEGM_BUILD_OPT_LEVEL").into(),
            target: env!("PHLEGM_BUILD_TARGET").into(),
            rustc: env!("PHLEGM_BUILD_RUSTC").into(),
            features: if cfg!(feature = "npu") {
                vec!["npu".into()]
            } else {
                vec![]
            },
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version.is_empty() || self.tag != format!("v{}", self.version) {
            return Err(
                "benchmarks require a build from the exact v<Cargo version> release tag".into(),
            );
        }
        if self.dirty {
            return Err("benchmarks require an unmodified release build".into());
        }
        if self.opt_level != "3" {
            return Err(
                "benchmarks require an optimized release build (cargo build --release)".into(),
            );
        }
        if !is_hex(&self.commit, 40) || self.target.is_empty() || self.rustc.is_empty() {
            return Err("missing build provenance".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineInfo {
    pub cpu: String,
    pub logical_cpus: usize,
    pub memory_bytes: u64,
    pub os: String,
    pub architecture: String,
    pub accelerators: Vec<AcceleratorInfo>,
    /// Record AC/battery and the selected power/TDP setting when known.
    pub power_profile: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorInfo {
    pub name: String,
    pub driver_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInfo {
    pub id: String,
    pub revision: String,
    /// SHA-256 of the actual weight artifact (or a documented artifact manifest).
    pub weights_sha256: String,
    pub tokenizer_sha256: String,
    pub quantization: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Npu,
    Cpu,
    Gpu,
    Mock,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeInfo {
    pub backend: String,
    pub device_kind: DeviceKind,
    pub device: String,
    pub driver_version: String,
    pub kernel_sha256: String,
    pub context_capacity: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<reference::ReferenceInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchConfig {
    pub warmup_runs: usize,
    pub measured_runs: usize,
    /// Includes the first output token produced by prefill.
    pub output_tokens: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 1,
            measured_runs: 3,
            output_tokens: 128,
        }
    }
}

impl BenchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=10).contains(&self.warmup_runs)
            || !(1..=20).contains(&self.measured_runs)
            || !(2..=4096).contains(&self.output_tokens)
        {
            return Err(
                "expected 1..10 warmups, 1..20 measured runs, and 2..4096 output tokens".into(),
            );
        }
        Ok(())
    }
}

/// An adapter for the real engine, deliberately separate from Backend::generate:
/// token callbacks alone cannot measure true prefill time.
///
/// All methods must finish device work before returning. The model stays loaded;
/// reset clears KV, recurrent, and prefix caches before EVERY warmup/trial.
/// Use greedy sampling, batch/concurrency one, no speculative decoding or EOS
/// stopping. prefill returns the first output token; decode consumes the previous
/// token and performs exactly one forward pass, returning the next output token.
pub trait BenchBackend {
    fn model_info(&self) -> Result<ModelInfo, String>;
    fn runtime_info(&self) -> Result<RuntimeInfo, String>;
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
    fn reset(&mut self) -> Result<(), String>;
    fn prefill(&mut self, prompt: &[u32]) -> Result<u32, String>;
    fn decode(&mut self, previous_token: u32) -> Result<u32, String>;
    /// External adapters time the complete trial inside their engine process,
    /// excluding IPC. In-process engines use the default synchronous timers.
    fn measure_trial(&mut self, prompt: &[u32], output_tokens: usize) -> Result<Sample, String> {
        measure(self, prompt, output_tokens)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    /// Prompt forward pass, lm_head, and first greedy sample, with device sync.
    /// Also engine time-to-first-token; excludes loading, tokenization and reset.
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_token_ms: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Stats {
    pub median: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseSummary {
    pub prefill_tokens_per_second: Stats,
    pub decode_tokens_per_second: Stats,
    pub time_to_first_token_ms: Stats,
    pub decode_token_ms_p50: f64,
    pub decode_token_ms_p95: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub input_tokens: usize,
    pub samples: Vec<Sample>,
    pub summary: CaseSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchReport {
    pub schema_version: u32,
    pub benchmark_version: String,
    /// UTC milliseconds since the Unix epoch, captured before the first warmup.
    pub started_at_unix_ms: u64,
    pub phlegm: ReleaseInfo,
    pub machine: MachineInfo,
    pub model: ModelInfo,
    pub runtime: RuntimeInfo,
    pub config: BenchConfig,
    /// The exact 4096 token IDs; shorter cases use prefixes of this sequence.
    pub prompt_token_ids: Vec<u32>,
    pub notes: Option<String>,
    pub cases: Vec<CaseResult>,
}

fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn validate_metadata(
    release: &ReleaseInfo,
    machine: &MachineInfo,
    model: &ModelInfo,
    runtime: &RuntimeInfo,
    config: &BenchConfig,
) -> Result<(), String> {
    release.validate()?;
    config.validate()?;
    if machine.cpu.trim().is_empty()
        || machine.os.trim().is_empty()
        || machine.architecture.is_empty()
        || machine.logical_cpus == 0
        || machine.memory_bytes == 0
    {
        return Err("missing CPU, RAM, OS, or architecture metadata".into());
    }
    if model.id.trim().is_empty()
        || model.revision.trim().is_empty()
        || model.quantization.trim().is_empty()
        || !is_hex(&model.weights_sha256, 64)
        || !is_hex(&model.tokenizer_sha256, 64)
    {
        return Err(
            "missing model identity, revision, quantization, or SHA-256 fingerprints".into(),
        );
    }
    if matches!(runtime.device_kind, DeviceKind::Mock) {
        return Err("mock backends cannot produce benchmark reports".into());
    }
    if runtime.backend.trim().is_empty()
        || runtime.device.trim().is_empty()
        || runtime.driver_version.trim().is_empty()
        || !is_hex(&runtime.kernel_sha256, 64)
    {
        return Err("missing backend, device, driver, or kernel fingerprint".into());
    }
    if let Some(reference) = &runtime.reference {
        reference.validate()?;
        if runtime.backend != reference::BACKEND_NAME {
            return Err("reference-engine provenance requires the reference backend label".into());
        }
    } else if runtime.backend == reference::BACKEND_NAME {
        return Err("reference-engine results require reference provenance".into());
    }
    // The last output token isn't fed back into the model.
    if runtime.context_capacity < INPUT_LENGTHS[3] + config.output_tokens - 1 {
        return Err("context capacity must fit 4096 input tokens plus the decode steps".into());
    }
    Ok(())
}

pub fn run(
    backend: &mut dyn BenchBackend,
    release: ReleaseInfo,
    machine: MachineInfo,
    config: BenchConfig,
    notes: Option<String>,
) -> Result<BenchReport, String> {
    // Fail before touching the backend on ineligible builds.
    release.validate()?;
    config.validate()?;
    let model = backend.model_info()?;
    let runtime = backend.runtime_info()?;
    validate_metadata(&release, &machine, &model, &runtime, &config)?;
    let mut prompt = backend.encode(&PROMPT.repeat(256))?;
    if prompt.len() < INPUT_LENGTHS[3] {
        return Err("benchmark corpus encodes to fewer than 4096 tokens".into());
    }
    prompt.truncate(INPUT_LENGTHS[3]);
    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64;
    let mut cases = Vec::new();
    for input_tokens in INPUT_LENGTHS {
        let prompt = &prompt[..input_tokens];
        for trial in 0..config.warmup_runs {
            backend
                .measure_trial(prompt, config.output_tokens)
                .map_err(|e| format!("{input_tokens} tokens, warmup {}: {e}", trial + 1))?;
        }
        let mut samples = Vec::new();
        for trial in 0..config.measured_runs {
            samples.push(
                backend
                    .measure_trial(prompt, config.output_tokens)
                    .map_err(|e| format!("{input_tokens} tokens, trial {}: {e}", trial + 1))?,
            );
        }
        let summary = summarize(input_tokens, config.output_tokens, &samples);
        cases.push(CaseResult {
            input_tokens,
            samples,
            summary,
        });
    }
    let report = BenchReport {
        schema_version: 1,
        benchmark_version: BENCHMARK_VERSION.into(),
        started_at_unix_ms,
        phlegm: release,
        machine,
        model,
        runtime,
        config,
        prompt_token_ids: prompt,
        notes,
        cases,
    };
    report.validate()?;
    Ok(report)
}

fn measure<B: BenchBackend + ?Sized>(
    backend: &mut B,
    prompt: &[u32],
    output_tokens: usize,
) -> Result<Sample, String> {
    backend.reset()?;
    let start = Instant::now();
    let mut previous = backend.prefill(prompt)?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut decode_token_ms = Vec::with_capacity(output_tokens - 1);
    let start = Instant::now();
    for _ in 1..output_tokens {
        let step = Instant::now();
        previous = backend.decode(previous)?;
        decode_token_ms.push(step.elapsed().as_secs_f64() * 1000.0);
    }
    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Sample {
        prefill_ms,
        decode_ms,
        decode_token_ms,
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (sorted.len() - 1) as f64 * p;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

fn stats(mut values: Vec<f64>) -> Stats {
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    Stats {
        median: percentile(&values, 0.5),
        mean,
        min: values[0],
        max: values[values.len() - 1],
        stddev: variance.sqrt(),
    }
}

fn summarize(input_tokens: usize, output_tokens: usize, samples: &[Sample]) -> CaseSummary {
    let mut latencies: Vec<f64> = samples
        .iter()
        .flat_map(|s| s.decode_token_ms.iter().copied())
        .collect();
    latencies.sort_by(f64::total_cmp);
    CaseSummary {
        prefill_tokens_per_second: stats(
            samples
                .iter()
                .map(|s| input_tokens as f64 * 1000.0 / s.prefill_ms)
                .collect(),
        ),
        decode_tokens_per_second: stats(
            samples
                .iter()
                .map(|s| (output_tokens - 1) as f64 * 1000.0 / s.decode_ms)
                .collect(),
        ),
        time_to_first_token_ms: stats(samples.iter().map(|s| s.prefill_ms).collect()),
        decode_token_ms_p50: percentile(&latencies, 0.5),
        decode_token_ms_p95: percentile(&latencies, 0.95),
    }
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 * b.abs().max(1.0)
}

fn stats_match(a: &Stats, b: &Stats) -> bool {
    [a.median, a.mean, a.min, a.max, a.stddev]
        .into_iter()
        .zip([b.median, b.mean, b.min, b.max, b.stddev])
        .all(|(a, b)| close(a, b))
}

impl BenchReport {
    /// Structural validation, not proof that a measurement is authentic.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.benchmark_version != BENCHMARK_VERSION {
            return Err("unsupported benchmark/schema version".into());
        }
        validate_metadata(
            &self.phlegm,
            &self.machine,
            &self.model,
            &self.runtime,
            &self.config,
        )?;
        if self.started_at_unix_ms == 0
            || self.prompt_token_ids.len() != INPUT_LENGTHS[3]
            || self.cases.len() != INPUT_LENGTHS.len()
        {
            return Err("missing run date, exact prompt IDs, or benchmark cases".into());
        }
        for (case, expected) in self.cases.iter().zip(INPUT_LENGTHS) {
            if case.input_tokens != expected || case.samples.len() != self.config.measured_runs {
                return Err(format!(
                    "expected {expected}-token case with all measured runs"
                ));
            }
            for s in &case.samples {
                if !s.prefill_ms.is_finite()
                    || s.prefill_ms <= 0.0
                    || !s.decode_ms.is_finite()
                    || s.decode_ms <= 0.0
                    || s.decode_token_ms.len() != self.config.output_tokens - 1
                    || s.decode_token_ms.iter().any(|v| !v.is_finite() || *v < 0.0)
                    || s.decode_token_ms.iter().sum::<f64>() > s.decode_ms + 1e-6
                {
                    return Err(format!(
                        "invalid or incomplete timings for {expected}-token case"
                    ));
                }
            }
            let expected = summarize(case.input_tokens, self.config.output_tokens, &case.samples);
            let got = &case.summary;
            if !stats_match(
                &got.prefill_tokens_per_second,
                &expected.prefill_tokens_per_second,
            ) || !stats_match(
                &got.decode_tokens_per_second,
                &expected.decode_tokens_per_second,
            ) || !stats_match(
                &got.time_to_first_token_ms,
                &expected.time_to_first_token_ms,
            ) || !close(got.decode_token_ms_p50, expected.decode_token_ms_p50)
                || !close(got.decode_token_ms_p95, expected.decode_token_ms_p95)
            {
                return Err("summary does not match raw timings".into());
            }
        }
        Ok(())
    }

    /// Validate before writing and never overwrite an existing result.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        use std::io::Write;
        self.validate()?;
        let data = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        if data.len() + 1 > MAX_REPORT_BYTES {
            return Err("benchmark JSON exceeds 16 MB".into());
        }
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        file.write_all(&data)
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| e.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        use std::io::Read;
        let mut data = Vec::new();
        fs::File::open(path)
            .map_err(|e| e.to_string())?
            .take((MAX_REPORT_BYTES + 1) as u64)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() > MAX_REPORT_BYTES {
            return Err("benchmark JSON exceeds 16 MB".into());
        }
        let report: Self = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        report.validate()?;
        Ok(report)
    }

    pub fn table(&self) -> String {
        let mut out = format!(
            "Phlegm {} | {} | {} | {}\ninput tokens   prefill tok/s   decode tok/s   first token ms\n",
            self.phlegm.tag, self.runtime.backend, self.model.id, self.runtime.device
        );
        for case in &self.cases {
            out.push_str(&format!(
                "{:>12}   {:>13.2}   {:>12.2}   {:>14.2}\n",
                case.input_tokens,
                case.summary.prefill_tokens_per_second.median,
                case.summary.decode_tokens_per_second.median,
                case.summary.time_to_first_token_ms.median
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release() -> ReleaseInfo {
        ReleaseInfo {
            version: "0.1.0".into(),
            tag: "v0.1.0".into(),
            commit: "a".repeat(40),
            dirty: false,
            opt_level: "3".into(),
            target: "test-target".into(),
            rustc: "test".into(),
            features: vec![],
        }
    }

    fn machine() -> MachineInfo {
        MachineInfo {
            cpu: "test CPU".into(),
            logical_cpus: 8,
            memory_bytes: 16 << 30,
            os: "test OS".into(),
            architecture: "test arch".into(),
            accelerators: vec![],
            power_profile: None,
        }
    }

    #[derive(Default)]
    struct Fixture {
        resets: usize,
        prefills: Vec<usize>,
        decodes: usize,
        fail: bool,
        engine_timing: bool,
        measurements: usize,
    }
    impl BenchBackend for Fixture {
        fn model_info(&self) -> Result<ModelInfo, String> {
            Ok(ModelInfo {
                id: "test-only".into(),
                revision: "test".into(),
                weights_sha256: "b".repeat(64),
                tokenizer_sha256: "c".repeat(64),
                quantization: "test".into(),
            })
        }
        fn runtime_info(&self) -> Result<RuntimeInfo, String> {
            Ok(RuntimeInfo {
                backend: "test-only".into(),
                device_kind: DeviceKind::Cpu,
                device: "test".into(),
                driver_version: "test".into(),
                kernel_sha256: "d".repeat(64),
                context_capacity: 8192,
                reference: None,
            })
        }
        fn encode(&self, _: &str) -> Result<Vec<u32>, String> {
            Ok((0..5000).collect())
        }
        fn reset(&mut self) -> Result<(), String> {
            self.resets += 1;
            Ok(())
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<u32, String> {
            if self.fail {
                return Err("device failed".into());
            }
            self.prefills.push(ids.len());
            std::thread::sleep(std::time::Duration::from_millis(1));
            Ok(10)
        }
        fn decode(&mut self, previous: u32) -> Result<u32, String> {
            assert_eq!(previous, 10 + (self.decodes % 3) as u32);
            self.decodes += 1;
            Ok(previous + 1)
        }
        fn measure_trial(&mut self, prompt: &[u32], outputs: usize) -> Result<Sample, String> {
            self.measurements += 1;
            if self.engine_timing {
                Ok(Sample {
                    prefill_ms: 500.0,
                    decode_ms: 1000.0,
                    decode_token_ms: vec![1000.0 / (outputs - 1) as f64; outputs - 1],
                })
            } else {
                measure(self, prompt, outputs)
            }
        }
    }

    fn fixture_report() -> BenchReport {
        run(
            &mut Fixture::default(),
            release(),
            machine(),
            BenchConfig {
                output_tokens: 4,
                ..BenchConfig::default()
            },
            None,
        )
        .unwrap()
    }

    #[test]
    fn resets_every_trial_uses_exact_lengths_and_excludes_warmups() {
        let mut backend = Fixture::default();
        let report = run(
            &mut backend,
            release(),
            machine(),
            BenchConfig {
                output_tokens: 4,
                ..BenchConfig::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(backend.resets, 16);
        assert_eq!(backend.decodes, 48);
        assert_eq!(
            backend.prefills,
            INPUT_LENGTHS
                .into_iter()
                .flat_map(|n| [n; 4])
                .collect::<Vec<_>>()
        );
        assert!(report.cases.iter().all(|c| c.samples.len() == 3));
        assert_eq!(report.prompt_token_ids, (0..4096).collect::<Vec<_>>());
        report.validate().unwrap();
    }

    #[test]
    fn external_engine_timings_are_not_replaced_with_transport_time() {
        let mut backend = Fixture {
            engine_timing: true,
            ..Fixture::default()
        };
        let report = run(
            &mut backend,
            release(),
            machine(),
            BenchConfig {
                output_tokens: 4,
                ..BenchConfig::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(backend.measurements, 16);
        assert_eq!(backend.resets, 0);
        assert_eq!(
            report.cases[0].summary.prefill_tokens_per_second.median,
            1024.0
        );
        assert_eq!(report.cases[0].summary.decode_tokens_per_second.median, 3.0);
        assert!(report
            .cases
            .iter()
            .all(|c| c.samples.iter().all(|s| s.prefill_ms == 500.0)));
    }

    #[test]
    fn counts_decode_steps_without_counting_first_token_twice() {
        let s = Sample {
            prefill_ms: 500.0,
            decode_ms: 1000.0,
            decode_token_ms: vec![100.0, 300.0, 600.0],
        };
        let summary = summarize(512, 4, &[s]);
        assert_eq!(summary.prefill_tokens_per_second.median, 1024.0);
        assert_eq!(summary.decode_tokens_per_second.median, 3.0);
        assert_eq!(summary.decode_token_ms_p50, 300.0);
        assert_eq!(summary.decode_token_ms_p95, 570.0);
        assert_eq!(stats(vec![1.0, 3.0]).median, 2.0);
        assert_eq!(stats(vec![1.0, 3.0]).stddev, 1.0);
    }

    #[test]
    fn rejects_ineligible_builds_before_inference() {
        for kind in 0..4 {
            let mut release = release();
            match kind {
                0 => release.tag.clear(),
                1 => release.dirty = true,
                2 => release.opt_level = "0".into(),
                _ => release.tag = "v9.9.9".into(),
            }
            let mut backend = Fixture::default();
            assert!(run(
                &mut backend,
                release,
                machine(),
                BenchConfig::default(),
                None
            )
            .is_err());
            assert_eq!(backend.resets, 0);
        }
    }

    #[test]
    fn rejects_mock_metadata_context_shortfalls_and_invalid_config() {
        let b = Fixture::default();
        let mut runtime = b.runtime_info().unwrap();
        runtime.device_kind = DeviceKind::Mock;
        assert!(validate_metadata(
            &release(),
            &machine(),
            &b.model_info().unwrap(),
            &runtime,
            &BenchConfig::default()
        )
        .is_err());
        runtime.device_kind = DeviceKind::Npu;
        runtime.context_capacity = 4096;
        assert!(validate_metadata(
            &release(),
            &machine(),
            &b.model_info().unwrap(),
            &runtime,
            &BenchConfig::default()
        )
        .is_err());
        assert!(BenchConfig {
            measured_runs: 0,
            ..BenchConfig::default()
        }
        .validate()
        .is_err());
        assert!(BenchConfig {
            output_tokens: 1,
            ..BenchConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn report_roundtrip_and_validation_detects_tampering() {
        let report = fixture_report();
        let bytes = serde_json::to_vec(&report).unwrap();
        let parsed: BenchReport = serde_json::from_slice(&bytes).unwrap();
        parsed.validate().unwrap();
        let mut bad = parsed.clone();
        bad.cases[0].summary.decode_tokens_per_second.median *= 2.0;
        assert!(bad.validate().is_err());
        let mut bad = parsed.clone();
        bad.cases[0].samples[0].decode_token_ms.pop();
        assert!(bad.validate().is_err());
        let mut bad = parsed.clone();
        bad.cases[0].samples[0].prefill_ms = f64::NAN;
        assert!(bad.validate().is_err());
        let mut bad = parsed;
        bad.cases.swap(0, 1);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn saves_reloadable_reports_and_refuses_overwrites() {
        let report = fixture_report();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/bench-report-tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "fixture-{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        report.save(&path).unwrap();
        let original = fs::read(&path).unwrap();
        let loaded = BenchReport::load(&path).unwrap();
        assert!(loaded.table().contains("4096"));
        assert!(report.save(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
        let bad_path = path.with_extension("invalid.json");
        let mut bad = report;
        bad.phlegm.dirty = true;
        assert!(bad.save(&bad_path).is_err());
        assert!(!bad_path.exists());
    }

    #[test]
    fn engine_failures_abort_instead_of_reporting_partial_results() {
        let result = run(
            &mut Fixture {
                fail: true,
                ..Fixture::default()
            },
            release(),
            machine(),
            BenchConfig::default(),
            None,
        );
        assert!(result
            .unwrap_err()
            .contains("512 tokens, warmup 1: device failed"));
    }
}
