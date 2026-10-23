use phlegm_core::bench::{
    self, machine,
    reference::{ReferenceBackend, ReferenceOptions},
    BenchConfig, BenchReport, MachineInfo, ReleaseInfo,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

const HELP: &str = "usage: phlegm bench [machine | validate <results.json> | --help]
       phlegm bench --reference-engine <bridge.exe> --model-dir <dir>
         --kernels <dir> --model-id <id> --model-revision <revision>
         --quantization <name> --output <results.json> [options]

Phlegm Bench v1: 512, 1024, 2048, 4096 input tokens; batch/concurrency 1.
Defaults: 1 warmup + 3 measured runs per length; 128 output tokens; greedy.
Records separate synchronous prefill/decode timings, raw samples and metadata.
Only clean optimized builds from an exact v<Cargo version> tag are eligible.

  machine                 print public machine metadata as JSON
  validate <results.json> validate a saved report and print its median speeds

Temporary backend: openflowlm-cpp-reference (not the future native Rust engine).
Build its resident process bridge with benchmarks/reference/build.ps1 first.

Options:
  --warmups N                 warmup trials per input length (default 1)
  --runs N                    measured trials per input length (default 3)
  --output-tokens N           output tokens, including the first (default 128)
  --context N                 context capacity (default 8192)
  --prefill-route MODE        auto | sequential | block (default auto)
  --prefill-mode MODE         fast | lean (default fast)
  --threads N                 OpenMP threads (default logical CPU count)
  --driver-version VERSION    override detected NPU driver version
  --machine-info FILE         use explicitly supplied machine JSON
  --xrt-smi FILE              XRT diagnostic executable
  --rpc-timeout-seconds N     per bridge request timeout (default 3600)
  --dispatch-timeout-ms N     per device dispatch timeout (default 60000)
  --notes TEXT                cooling, AC/battery, TDP and other run conditions

NPU power mode is read before/after each trial; changing it invalidates the run.
OMP_WAIT_POLICY=PASSIVE is enforced; inherited OFLM/OMP tuning is cleared.
See benchmarks/reference/README.md and specs/bench/spec.md.";

pub fn command(args: &[String]) -> Result<(), String> {
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--help"] | ["-h"] => {
            println!("{HELP}");
            Ok(())
        }
        ["machine"] => {
            let info = machine::collect()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&info).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        ["validate", path] => {
            let report = BenchReport::load(Path::new(path))?;
            print!("{}", report.table());
            Ok(())
        }
        [] => Err(HELP.into()),
        _ => run_reference(args),
    }
}

const FLAGS: &[&str] = &[
    "--reference-engine",
    "--model-dir",
    "--kernels",
    "--model-id",
    "--model-revision",
    "--quantization",
    "--output",
    "--warmups",
    "--runs",
    "--output-tokens",
    "--context",
    "--prefill-route",
    "--prefill-mode",
    "--threads",
    "--driver-version",
    "--machine-info",
    "--xrt-smi",
    "--rpc-timeout-seconds",
    "--dispatch-timeout-ms",
    "--notes",
];

fn parse(args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut flags = BTreeMap::new();
    let mut pairs = args.chunks_exact(2);
    for pair in &mut pairs {
        if !FLAGS.contains(&pair[0].as_str()) {
            return Err(format!("unknown benchmark option {}", pair[0]));
        }
        if pair[1].trim().is_empty() || pair[1].starts_with("--") {
            return Err(format!("missing value for {}", pair[0]));
        }
        if flags.insert(pair[0].clone(), pair[1].clone()).is_some() {
            return Err(format!("duplicate option {}", pair[0]));
        }
    }
    if !pairs.remainder().is_empty() {
        return Err("every benchmark option requires a value".into());
    }
    Ok(flags)
}

fn required<'a>(flags: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
    flags
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("{name} is required; see phlegm bench --help"))
}

fn number(flags: &BTreeMap<String, String>, name: &str, default: usize) -> Result<usize, String> {
    match flags.get(name) {
        Some(value) => value
            .parse()
            .map_err(|_| format!("{name} must be a positive integer")),
        None => Ok(default),
    }
}

fn run_reference(args: &[String]) -> Result<(), String> {
    let flags = parse(args)?;
    // Validate all required arguments before running hardware or creating files.
    for name in [
        "--reference-engine",
        "--model-dir",
        "--kernels",
        "--model-id",
        "--model-revision",
        "--quantization",
        "--output",
    ] {
        required(&flags, name)?;
    }
    let config = BenchConfig {
        warmup_runs: number(&flags, "--warmups", 1)?,
        measured_runs: number(&flags, "--runs", 3)?,
        output_tokens: number(&flags, "--output-tokens", 128)?,
    };
    config.validate()?;
    let release = ReleaseInfo::current();
    release.validate()?; // No development-build bypass; metadata/validator remain usable.
    let output = PathBuf::from(required(&flags, "--output")?);
    if output.exists() {
        return Err(format!(
            "{} already exists; refusing to overwrite",
            output.display()
        ));
    }
    let machine: MachineInfo = if let Some(path) = flags.get("--machine-info") {
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        machine::collect()?
    };
    let driver = flags
        .get("--driver-version")
        .cloned()
        .or_else(|| {
            machine
                .accelerators
                .iter()
                .find(|a| {
                    a.name.split(|c: char| !c.is_alphanumeric()).any(|s| {
                        s.eq_ignore_ascii_case("npu")
                            || s.eq_ignore_ascii_case("ipu")
                            || s.eq_ignore_ascii_case("neural")
                    })
                })
                .and_then(|a| a.driver_version.clone())
        })
        .ok_or("NPU driver not detected; provide --driver-version")?;
    let context = number(&flags, "--context", 8192)?;
    let threads = number(&flags, "--threads", machine.logical_cpus)?;
    let timeout = number(&flags, "--rpc-timeout-seconds", 3600)?;
    let dispatch_timeout = number(&flags, "--dispatch-timeout-ms", 60000)?;
    if !(4096 + config.output_tokens - 1..=131072).contains(&context)
        || threads == 0
        || !(1..=86400).contains(&timeout)
        || !(1..=3_600_000).contains(&dispatch_timeout)
    {
        return Err("invalid context capacity, thread count or timeout".into());
    }
    let xrt_smi = flags
        .get("--xrt-smi")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if cfg!(windows) {
                PathBuf::from(
                    std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into()),
                )
                .join("System32/AMD/xrt-smi.exe")
            } else {
                PathBuf::from("xrt-smi")
            }
        });
    let options = ReferenceOptions {
        bridge: Path::new(required(&flags, "--reference-engine")?)
            .canonicalize()
            .map_err(|e| e.to_string())?,
        model_dir: Path::new(required(&flags, "--model-dir")?)
            .canonicalize()
            .map_err(|e| e.to_string())?,
        kernel_dir: Path::new(required(&flags, "--kernels")?)
            .canonicalize()
            .map_err(|e| e.to_string())?,
        model_id: required(&flags, "--model-id")?.into(),
        model_revision: required(&flags, "--model-revision")?.into(),
        quantization: required(&flags, "--quantization")?.into(),
        driver_version: driver,
        context_capacity: context,
        prefill_route: flags
            .get("--prefill-route")
            .cloned()
            .unwrap_or_else(|| "auto".into()),
        prefill_mode: flags
            .get("--prefill-mode")
            .cloned()
            .unwrap_or_else(|| "fast".into()),
        threads,
        xrt_smi,
        request_timeout: Duration::from_secs(timeout as u64),
        dispatch_timeout_ms: dispatch_timeout as u32,
    };
    eprintln!(
        "Benchmarking the C++ reference engine; results will be labeled {}.",
        bench::reference::BACKEND_NAME
    );
    let mut backend = ReferenceBackend::load(options)?;
    let report = bench::run(
        &mut backend,
        release,
        machine,
        config,
        flags.get("--notes").cloned(),
    )?;
    backend.finish()?;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    report.save(&output)?;
    print!("{}", report.table());
    eprintln!("Saved {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_commands_and_missing_reports_fail() {
        assert!(command(&["validate".into()]).is_err());
        assert!(command(&["validate".into(), "does-not-exist.json".into()]).is_err());
        assert!(command(&["--allow-untagged".into()]).is_err());
    }

    #[test]
    fn rejects_duplicates_missing_values_and_unknown_options() {
        let args = |a: &[&str]| a.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!(parse(&args(&["--runs", "3", "--runs", "4"])).is_err());
        assert!(parse(&args(&["--runs"])).is_err());
        assert!(parse(&args(&["--runs", "--output"])).is_err());
        assert!(parse(&args(&["--allow-untagged", "true"])).is_err());
        let flags = parse(&args(&["--model-dir", "C:/models/a model", "--runs", "3"])).unwrap();
        assert_eq!(
            required(&flags, "--model-dir").unwrap(),
            "C:/models/a model"
        );
        assert_eq!(number(&flags, "--runs", 1).unwrap(), 3);
    }
}
