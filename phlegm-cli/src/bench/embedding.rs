use super::{absolute, machine, number, parse_flags, required, MachineInfo, ReleaseInfo};
use crate::embedding_pack;
use phlegm_core::{
    bench::reference::hash_file,
    embeddings::{EmbeddingInput, EmbeddingTimings, ResidentEmbeddingBackend},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

const HELP: &str =
    "usage: phlegm bench embed --embed-model <pack.npue|model.gguf> --embed-worker <exe>
  --embed-design <dir> [--embed-attention <dir>] --model-id <id> --model-revision <rev>
  --quantization <quant> --fixtures <cases.json> --output <report.json>
  [--embed-short-design <dir> [--embed-short-attention <dir>]]
  [--embed-packer EXE --embed-gguf-sha256 HASH] [--embed-pack-cache DIR]
  [--embed-pack-script FILE] [--embed-python PYTHON]
  [--warmups N] [--runs N] [--machine-info FILE] [--notes TEXT] [--development]

The fixture file is an array of cases with tokens_per_input and either texts
or token_ids. Use an optional prompt_name for text cases. Every synchronous run
validates the worker's input token count. Reports include raw timings and
artifact hashes. The measurement includes Phlegm worker IPC and excludes HTTP.";

const FLAGS: &[&str] = &[
    "--embed-model",
    "--embed-worker",
    "--embed-python",
    "--embed-design",
    "--embed-attention",
    "--embed-short-design",
    "--embed-short-attention",
    "--model-id",
    "--model-revision",
    "--quantization",
    "--fixtures",
    "--output",
    "--warmups",
    "--runs",
    "--machine-info",
    "--notes",
];

struct Case {
    input: EmbeddingInput,
    tokens_per_input: usize,
    prompt: Option<String>,
    samples: Vec<Sample>,
}

struct Sample {
    worker_ms: f64,
    encoder: Option<EmbeddingTimings>,
}

fn cases(data: &[u8]) -> Result<Vec<Case>, String> {
    let records: Vec<Value> = serde_json::from_slice(data).map_err(|e| e.to_string())?;
    if records.is_empty() {
        return Err("embedding benchmark requires at least one fixture".into());
    }
    records
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let invalid = |message: &str| format!("embedding fixture {index}: {message}");
            let tokens_per_input = row
                .get("tokens_per_input")
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| invalid("a positive tokens_per_input is required"))?;
            let input = match (row.get("texts"), row.get("token_ids")) {
                (Some(texts), None) => EmbeddingInput::Texts(
                    serde_json::from_value(texts.clone())
                        .map_err(|_| invalid("texts must be an array of strings"))?,
                ),
                (None, Some(ids)) => {
                    let rows: Vec<Vec<u32>> =
                        serde_json::from_value(ids.clone()).map_err(|_| {
                            invalid("token_ids must be an array of unsigned integer sequences")
                        })?;
                    if rows.iter().any(|ids| ids.len() != tokens_per_input) {
                        return Err(invalid(
                            "token sequence length differs from tokens_per_input",
                        ));
                    }
                    EmbeddingInput::TokenIds(rows)
                }
                _ => return Err(invalid("provide either texts or token_ids")),
            };
            if input.is_empty()
                || input.len() > 2048
                || tokens_per_input.checked_mul(input.len()).is_none()
            {
                return Err(invalid("invalid fixture batch or token count"));
            }
            let prompt = row
                .get("prompt_name")
                .map(|p| {
                    p.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("prompt_name must be a string"))
                })
                .transpose()?;
            if prompt.is_some() && matches!(input, EmbeddingInput::TokenIds(_)) {
                return Err(invalid("put task prefixes inside pretokenized inputs"));
            }
            Ok(Case {
                input,
                tokens_per_input,
                prompt,
                samples: Vec::new(),
            })
        })
        .collect()
}

fn artifact_hashes(directory: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut hashes = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|e| format!("{}: {e}", directory.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("bin" | "xclbin" | "json")
        ) {
            hashes.insert(
                entry.file_name().to_string_lossy().into_owned(),
                hash_file(&path)?,
            );
        }
    }
    Ok(hashes)
}

fn projection(design: &Path) -> Result<(usize, BTreeMap<String, String>), String> {
    let projector = artifact_hashes(&design.join("gemm_rtp"))?;
    if !projector.contains_key("design.json") || !projector.contains_key("final.xclbin") {
        return Err("embedding benchmark requires an exported GEMM RTP design".into());
    }
    let geometry: Value = serde_json::from_slice(
        &fs::read(design.join("gemm_rtp/design.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let context = geometry
        .get("seq")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or("projection design omitted its sequence capacity")?;
    Ok((context, projector))
}

fn measure(
    backend: &mut ResidentEmbeddingBackend,
    case: &Case,
    expected_context: usize,
) -> Result<Sample, String> {
    let input = case.input.clone();
    let start = Instant::now();
    let (result, encoder) = backend
        .embed_profiled(input, case.prompt.as_deref())
        .map_err(|e| format!("{e:?}"))?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let expected = case.tokens_per_input * case.input.len();
    if result.tokens != expected {
        return Err(format!(
            "fixture expected {expected} tokens, worker processed {}",
            result.tokens
        ));
    }
    if let Some(timing) = &encoder {
        if timing.context_tokens != expected_context {
            return Err(format!(
                "{}-token fixture expected a {expected_context}-token design, worker used {}",
                case.tokens_per_input, timing.context_tokens
            ));
        }
    }
    if elapsed <= 0.0 {
        return Err("timer could not resolve the embedding duration".into());
    }
    Ok(Sample {
        worker_ms: elapsed,
        encoder,
    })
}

pub(super) fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args == ["--help"] || args == ["-h"] {
        println!("{HELP}");
        return Ok(());
    }
    let mut release = ReleaseInfo::current();
    release.development = args.iter().any(|a| a == "--development");
    let args: Vec<String> = args
        .iter()
        .filter(|a| *a != "--development")
        .cloned()
        .collect();
    let allowed: Vec<_> = FLAGS.iter().chain(embedding_pack::FLAGS).copied().collect();
    let flags = parse_flags(&args, &allowed)?;
    let value = |name| required(&flags, name);
    let model = value("--model-id")?;
    let revision = value("--model-revision")?;
    let quantization = value("--quantization")?;
    let output = PathBuf::from(value("--output")?);
    if output.exists() {
        return Err(format!("{} already exists", output.display()));
    }
    let warmups = number(&flags, "--warmups", 1)?;
    let runs = number(&flags, "--runs", 3)?;
    if !(1..=10).contains(&warmups) || !(1..=20).contains(&runs) {
        return Err("expected 1..10 warmups and 1..20 measured runs".into());
    }
    release.validate()?;
    let model_path = absolute(value("--embed-model")?)?;
    let worker = absolute(value("--embed-worker")?)?;
    let design = absolute(value("--embed-design")?)?;
    let attention = flags
        .get("--embed-attention")
        .map(|p| absolute(p))
        .transpose()?;
    let short_design = flags
        .get("--embed-short-design")
        .map(|p| absolute(p))
        .transpose()?;
    let short_attention = flags
        .get("--embed-short-attention")
        .map(|p| absolute(p))
        .transpose()?;
    if short_attention.is_some() && short_design.is_none() {
        return Err("--embed-short-attention requires --embed-short-design".into());
    }
    let fixture = Path::new(value("--fixtures")?);
    let mut cases = cases(&fs::read(fixture).map_err(|e| e.to_string())?)?;
    let machine: MachineInfo = match flags.get("--machine-info") {
        Some(file) => serde_json::from_slice(&fs::read(file).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?,
        None => machine::collect()?,
    };
    let (context_capacity, projector) = projection(&design)?;
    if cases
        .iter()
        .any(|case| case.tokens_per_input > context_capacity)
    {
        return Err(format!(
            "benchmark fixtures exceed the {context_capacity}-token design"
        ));
    }
    let pack = embedding_pack::prepare(&model_path, &design, |name| {
        flags.get(name).map(String::as_str)
    })?;
    let artifacts = json!({
        "npue_sha256": hash_file(&pack.path)?,
        "gguf_sha256": pack.source_sha256,
        "worker_sha256": hash_file(&worker)?,
        "fixtures_sha256": hash_file(fixture)?,
        "projection_sha256": projector,
        "attention_sha256": attention.as_deref().map(artifact_hashes).transpose()?,
    });
    let mut short_context = None;
    let mut short_artifacts = None;
    if let Some(dir) = &short_design {
        let (context, hashes) = projection(dir)?;
        if context >= context_capacity {
            return Err(
                "short embedding design must have a smaller context than the full design".into(),
            );
        }
        short_context = Some(context);
        short_artifacts = Some(json!({
            "projection_sha256": hashes,
            "attention_sha256": short_attention.as_deref().map(artifact_hashes).transpose()?,
        }));
    }
    let mut backend = ResidentEmbeddingBackend::load_npue_routed(
        model.into(),
        &worker,
        &pack.path,
        &design,
        attention.as_deref(),
        short_design
            .as_deref()
            .map(|dir| (dir, short_attention.as_deref())),
    )?;
    let expected_context = |case: &Case| {
        short_context
            .filter(|&context| case.tokens_per_input <= context)
            .unwrap_or(context_capacity)
    };
    for case in &cases {
        for _ in 0..warmups {
            measure(&mut backend, case, expected_context(case))?;
        }
    }
    for round in 0..runs {
        let mut order: Vec<_> = (0..cases.len()).collect();
        if round % 2 != 0 {
            order.reverse();
        }
        for index in order {
            let case = &mut cases[index];
            let sample = measure(&mut backend, case, expected_context(case))?;
            case.samples.push(sample);
        }
    }
    drop(backend);
    let results: Vec<_> = cases
        .iter()
        .map(|case| {
            let batch = case.input.len();
            let worker_ms: Vec<_> = case.samples.iter().map(|sample| sample.worker_ms).collect();
            let encoder: Vec<_> = case
                .samples
                .iter()
                .map(|sample| sample.encoder.as_ref())
                .collect();
            let mut sorted = worker_ms.clone();
            sorted.sort_by(f64::total_cmp);
            let median = (sorted[(runs - 1) / 2] + sorted[runs / 2]) / 2.0;
            json!({
                "tokens_per_input": case.tokens_per_input,
                "batch": batch,
                "input_kind": match case.input {
                    EmbeddingInput::Texts(_) => "texts",
                    EmbeddingInput::TokenIds(_) => "token_ids",
                },
                "prompt_name": case.prompt,
                "samples_ms": worker_ms,
                "encoder_timings": encoder,
                "median_ms": median,
                "tokens_per_second": 1000.0 * (case.tokens_per_input * batch) as f64 / median,
            })
        })
        .collect();
    let report = json!({
        "benchmark": "phlegm-embedding-bench-v1",
        "release": release,
        "machine": machine,
        "model": {"id": model, "revision": revision, "quantization": quantization},
        "backend": "npue",
        "context_capacity": context_capacity,
        "short_context_capacity": short_context,
        "npu_attention": attention.is_some(),
        "artifacts": artifacts,
        "short_artifacts": short_artifacts,
        "notes": flags.get("--notes"),
        "warmups": warmups,
        "runs": runs,
        "results": results,
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let data = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)
        .and_then(|mut file| file.write_all(&data).and_then(|_| file.write_all(b"\n")))
        .map_err(|e| e.to_string())?;
    println!("Saved {}", output.display());
    Ok(())
}
