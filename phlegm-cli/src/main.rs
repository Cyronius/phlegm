use phlegm_core::backend::{Backend, GenParams};
use phlegm_core::bench::reference::{NpuBackend, NpuOptions};
use phlegm_core::mock_backend::MockBackend;
use phlegm_core::sampler::Sampler;
use phlegm_core::server::{self, ServerConfig};
use phlegm_core::tokenizer::{ChatMessage, Tokenizer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

mod bench;

const USAGE: &str = "usage: phlegm <command>

commands:
  run --model-dir DIR --kernels DIR --bridge EXE --prompt TEXT [--max-tokens N] [--turbo]
  serve --model-dir DIR --kernels DIR --bridge EXE [--max-ctx N] [--turbo]
  serve --mock
  bench     benchmark protocol, machine metadata, and result validation
  version   print the version

--turbo selects experimental lower-precision kernels from DIR/turbo; requires a matching bundle.
Build the resident NPU bridge with benchmarks/reference/build.ps1 first.";

fn flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    let mut at = 0;
    while at < args.len() {
        let key = &args[at];
        if key == "--mock" || key == "--turbo" {
            if out.insert(key.clone(), "true".into()).is_some() {
                return Err(format!("duplicate option {key}"));
            }
            at += 1;
            continue;
        }
        if !key.starts_with("--") || at + 1 >= args.len() || args[at + 1].starts_with("--") {
            return Err(format!("expected a value after {key}"));
        }
        if out.insert(key.clone(), args[at + 1].clone()).is_some() {
            return Err(format!("duplicate option {key}"));
        }
        at += 2;
    }
    Ok(out)
}

fn required(map: &HashMap<String, String>, key: &str) -> Result<String, String> {
    map.get(key)
        .cloned()
        .ok_or_else(|| format!("missing {key}"))
}

fn npu_options(map: &HashMap<String, String>) -> Result<NpuOptions, String> {
    let number = |key: &str, default: usize| -> Result<usize, String> {
        map.get(key)
            .map(|v| v.parse().map_err(|_| format!("invalid {key}")))
            .unwrap_or(Ok(default))
    };
    Ok(NpuOptions {
        model_dir: PathBuf::from(required(map, "--model-dir")?),
        kernel_dir: PathBuf::from(required(map, "--kernels")?),
        turbo: map.contains_key("--turbo"),
        bridge: PathBuf::from(required(map, "--bridge")?),
        context_capacity: number("--max-ctx", 4096)?,
        threads: number("--threads", 4)?,
        prefill_route: map
            .get("--prefill-route")
            .cloned()
            .unwrap_or_else(|| "auto".into()),
        prefill_mode: map
            .get("--prefill-mode")
            .cloned()
            .unwrap_or_else(|| "fast".into()),
        request_timeout: Duration::from_secs(number("--timeout", 600)? as u64),
        dispatch_timeout_ms: u32::try_from(number("--dispatch-timeout-ms", 120_000)?)
            .map_err(|_| "invalid --dispatch-timeout-ms".to_string())?,
    })
}

fn run(args: &[String]) -> Result<(), String> {
    let opts = flags(args)?;
    let npu = npu_options(&opts)?;
    let prompt = required(&opts, "--prompt")?;
    let max_tokens: usize = opts
        .get("--max-tokens")
        .map(|n| n.parse().map_err(|_| "invalid --max-tokens".to_string()))
        .unwrap_or(Ok(128))?;
    if max_tokens == 0 {
        return Err("--max-tokens must be positive".into());
    }
    let tokenizer = Tokenizer::load(&npu.model_dir)?;
    let text = tokenizer.apply_chat_template(
        &[ChatMessage {
            role: "user".into(),
            content: prompt,
        }],
        true,
        true,
    )?;
    let ids = tokenizer.encode(&text)?;
    let mut backend = NpuBackend::load(npu)?;
    let mut output = Vec::new();
    let mut params = GenParams {
        max_tokens,
        sampler: Sampler::new(0.0, 0, 1.0, 1.0, None),
        stop_ids: tokenizer.stop_ids(),
    };
    backend.generate(&ids, &mut params, &mut |id| output.push(id))?;
    println!("{}", tokenizer.decode(&output)?);
    Ok(())
}

fn serve(args: &[String]) -> Result<(), String> {
    let opts = flags(args)?;
    let mut cfg = ServerConfig::from_env();
    if opts.contains_key("--mock") {
        if opts.contains_key("--turbo") {
            return Err("--turbo requires the NPU backend; cannot combine with --mock".into());
        }
        cfg.backend_name = "mock".into();
        return server::serve(Box::new(MockBackend::new()), cfg);
    }
    let npu = npu_options(&opts)?;
    // Load before binding the port. A missing template must fail startup rather
    // than silently sending placeholder IDs to a real model.
    Tokenizer::load(&npu.model_dir)?;
    cfg.model_id = npu
        .model_dir
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or("local-model")
        .into();
    cfg.model_dir = Some(npu.model_dir.clone());
    cfg.backend_name = "npu".into();
    let backend = NpuBackend::load(npu)?;
    server::serve(Box::new(backend), cfg)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench::command(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("version") | Some("--version") => {
            println!("phlegm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err(USAGE.into()),
    };
    if let Err(e) = result {
        eprintln!("phlegm: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_is_a_boolean_and_cannot_silently_select_mock() {
        assert!(flags(&["--turbo".into()]).unwrap().contains_key("--turbo"));
        assert!(flags(&["--turbo".into(), "--turbo".into()]).is_err());
        assert!(serve(&["--mock".into(), "--turbo".into()]).unwrap_err().contains("cannot combine"));
        let mut map = HashMap::from([
            ("--model-dir".into(), "model".into()),
            ("--kernels".into(), "kernels".into()),
            ("--bridge".into(), "bridge".into()),
            ("--turbo".into(), "true".into()),
        ]);
        assert_eq!(npu_options(&map).unwrap().prefill_route, "auto");
        map.insert("--prefill-route".into(), "block".into());
        assert_eq!(npu_options(&map).unwrap().prefill_route, "block");
    }
}
