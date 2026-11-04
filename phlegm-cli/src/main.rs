use phlegm_core::backend::{Backend, GenParams};
use phlegm_core::bench::reference::{NpuBackend, NpuOptions};
use phlegm_core::image::{self, ImageConfig, ImageRequest};
use phlegm_core::mock_backend::MockBackend;
use phlegm_core::sampler::Sampler;
use phlegm_core::server::{self, GenDefaults, ServerConfig};
use phlegm_core::settings::{Config, Resolved};
use phlegm_core::tokenizer::{ChatMessage, Tokenizer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

mod bench;

const USAGE: &str = "usage: phlegm <command>

commands:
  run --model-dir DIR --kernels DIR --bridge EXE --prompt TEXT [--max-tokens N (0 or -1: until the context is full)] [--seed N] [--turbo]
  serve --model-dir DIR --kernels DIR --bridge EXE [--turbo]
  serve --mock
  config show [--model-dir DIR | --config-model TAG] [--json]   the settings in effect, and where each comes from
  config set [--model-dir DIR | --config-model TAG] --config-key KEY --config-value VALUE
  config unset [--model-dir DIR | --config-model TAG] --config-key KEY
  config list                                                   the registry's models
  image --config JSON --prompt TEXT --out DIR [--image REF] [--size 512x512] [--steps N] [--guidance N]
  serve-images --config JSON [--host 127.0.0.1] [--port 52626]
  bench     benchmark protocol, machine metadata, and result validation
  version   print the version

run and serve take the model's settings (phlegm config show --model-dir DIR), each overridable by its flag:
  --max-ctx N  --prefill-route auto|block|sequential  --prefill-mode fast|lean  --think true|false
  --temperature T  --top-k K  --top-p P  --repetition-penalty R  --max-tokens N
--turbo selects experimental lower-precision kernels from DIR/turbo; requires a matching bundle.
Build the resident NPU bridge with benchmarks/reference/build.ps1 first.";

fn flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    let mut at = 0;
    while at < args.len() {
        let key = &args[at];
        if key == "--mock" || key == "--turbo" || key == "--json" {
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

/// The model's settings (SERVER-SETTINGS): the registry and the user's settings.json under the
/// command line. `run` and `serve` take every sampling, template and prefill default from here.
fn model_settings(map: &HashMap<String, String>) -> Result<Resolved, String> {
    let cfg = Config::load()?;
    for w in &cfg.warnings {
        eprintln!("phlegm: {w}");
    }
    let dir = map.get("--model-dir").map(PathBuf::from);
    let tag = cfg.model_ref(map.get("--config-model").map(String::as_str), dir.as_deref())?;
    cfg.resolve(tag.as_deref(), dir.as_deref(), map)
}

fn npu_options(map: &HashMap<String, String>, settings: &Resolved) -> Result<NpuOptions, String> {
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
        context_capacity: settings.usize("ctx_len"),
        threads: number("--threads", 4)?,
        prefill_route: settings.str("prefill_route"),
        prefill_mode: settings.str("prefill_mode"),
        request_timeout: Duration::from_secs(number("--timeout", 600)? as u64),
        dispatch_timeout_ms: u32::try_from(number("--dispatch-timeout-ms", 120_000)?)
            .map_err(|_| "invalid --dispatch-timeout-ms".to_string())?,
    })
}

// Like llama.cpp (`-n -1`, its default): generate until a stop token or the end of the context.
// A positive --max-tokens bounds the reply, clamped to the room the context has left.
fn run(args: &[String]) -> Result<(), String> {
    let mut opts = flags(args)?;
    // -1 (llama.cpp's spelling) and 0 both mean until a stop token or the end of the context.
    if opts.get("--max-tokens").is_some_and(|v| v == "-1") {
        opts.insert("--max-tokens".into(), "0".into());
    }
    let settings = model_settings(&opts)?;
    let npu = npu_options(&opts, &settings)?;
    let prompt = required(&opts, "--prompt")?;
    let requested = settings.usize("max_tokens");
    let seed = opts
        .get("--seed")
        .map(|v| v.parse::<u64>().map_err(|_| "invalid --seed".to_string()))
        .transpose()?;
    let tokenizer = Tokenizer::load(&npu.model_dir)?;
    let text = tokenizer.apply_chat_template(
        &[ChatMessage {
            role: "user".into(),
            content: prompt,
        }],
        true,
        settings.bool("think"),
    )?;
    let ids = tokenizer.encode(&text)?;
    // The backend holds prompt + max_tokens - 1 tokens (the last one is never fed back).
    let capacity = npu.context_capacity;
    if ids.len() > capacity {
        return Err(format!(
            "prompt is {} tokens; --max-ctx {capacity} holds fewer",
            ids.len()
        ));
    }
    let room = capacity + 1 - ids.len();
    let max_tokens = if requested == 0 { room } else { requested.min(room) };
    let mut backend = NpuBackend::load(npu)?;
    let mut output = Vec::new();
    let mut params = GenParams {
        max_tokens,
        sampler: Sampler::new(
            settings.f32("temperature"),
            settings.usize("top_k"),
            settings.f32("top_p"),
            settings.f32("repetition_penalty"),
            seed,
        ),
        stop_ids: tokenizer.stop_ids(),
    };
    backend.generate(&ids, &mut params, &mut |id| output.push(id))?;
    let reply = tokenizer.decode(&output)?;
    // A reasoning model (harmony, Gemma 4's thought channel, Qwen's <think>) thinks before it
    // answers: the reasoning on stderr, the answer alone on stdout.
    match tokenizer.reasoning_for(&text) {
        Some(r) => {
            let (reasoning, answer) = r.split(&reply);
            if !reasoning.is_empty() {
                eprintln!("[reasoning]\n{reasoning}\n[/reasoning]");
            }
            if answer.is_empty() && output.len() >= max_tokens {
                if max_tokens == room {
                    eprintln!("[no answer before the context filled; raise --max-ctx]");
                } else {
                    eprintln!("[no answer within --max-tokens {max_tokens}; raise it]");
                }
            }
            println!("{answer}");
        }
        None => println!("{reply}"),
    }
    Ok(())
}

fn serve(args: &[String]) -> Result<(), String> {
    let opts = flags(args)?;
    let mut cfg = ServerConfig::from_env();
    let settings = model_settings(&opts)?;
    cfg.default_max_tokens = settings.usize("max_tokens");
    cfg.defaults = GenDefaults::from_settings(&settings);
    if opts.contains_key("--mock") {
        if opts.contains_key("--turbo") {
            return Err("--turbo requires the NPU backend; cannot combine with --mock".into());
        }
        cfg.backend_name = "mock".into();
        return server::serve(Box::new(MockBackend::new()), cfg);
    }
    let npu = npu_options(&opts, &settings)?;
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

/// `phlegm config`: OpenFlowLM's `oflm config` show / set / unset over the same two files.
fn config(args: &[String]) -> Result<(), String> {
    let (sub, rest) = args.split_first().ok_or("usage: phlegm config show|set|unset|list ...")?;
    let opts = flags(rest)?;
    const ALLOWED: &[&str] = &["--model-dir", "--config-model", "--config-key", "--config-value", "--json"];
    if let Some(k) = opts.keys().find(|k| !ALLOWED.contains(&k.as_str())) {
        return Err(format!("unknown config option {k}"));
    }
    let mut cfg = Config::load()?;
    for w in &cfg.warnings {
        eprintln!("phlegm: {w}");
    }
    let dir = opts.get("--model-dir").map(PathBuf::from);
    let named = opts.get("--config-model").map(String::as_str);
    let tag = cfg.model_ref(named, dir.as_deref())?;
    let registry = cfg
        .shipped_path
        .as_ref()
        .map_or("built in (phlegm-core/config/model_list.json)".to_string(), |p| p.display().to_string());
    let user_file = format!(
        "{}{}",
        cfg.user_path.display(),
        if cfg.user_path.exists() { "" } else { " (absent)" }
    );
    match sub.as_str() {
        "show" => {
            // What run and serve would use without flags (env vars included).
            let r = cfg.resolve_with(tag.as_deref(), dir.as_deref(), &HashMap::new(), &|k| std::env::var(k).ok())?;
            if opts.contains_key("--json") {
                let settings: serde_json::Map<String, serde_json::Value> = r
                    .values
                    .iter()
                    .map(|(k, (v, src))| (k.clone(), serde_json::json!({"value": v, "source": src.name()})))
                    .collect();
                let out = serde_json::json!({
                    "paths": {"registry": registry, "settings": cfg.user_path},
                    "model": tag, "model_dir": dir, "settings": settings,
                });
                println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
            } else {
                println!("registry  {registry}");
                println!("settings  {user_file}");
                match (&tag, &dir) {
                    (Some(t), _) => println!("model     {t}"),
                    (None, Some(d)) => println!("model     {} (not in the registry)", d.display()),
                    (None, None) => println!("model     (none: the defaults every model starts from)"),
                }
                for (k, (v, src)) in &r.values {
                    println!("  {k:<20} {:<10} ({})", v.to_string(), src.name());
                }
            }
            Ok(())
        }
        "list" => {
            for (t, e) in cfg.tags() {
                let name = e.get("name").and_then(serde_json::Value::as_str).unwrap_or("");
                let sampling = e.get("sampling").map(|s| s.to_string()).unwrap_or_default();
                println!("{t:<28} {name:<32} {sampling}");
            }
            Ok(())
        }
        "set" | "unset" => {
            if dir.is_some() && named.is_none() && tag.is_none() {
                return Err("that model directory is not in the registry; add it under \"models\" in \
                            settings.json (its \"name\" the directory's name) or pass --config-model"
                    .into());
            }
            let key = required(&opts, "--config-key")?;
            if sub == "set" {
                let def = cfg.def(&key).ok_or_else(|| format!("unknown setting {key:?}"))?;
                let value = def.parse(&required(&opts, "--config-value")?)?;
                cfg.set_user(tag.as_deref(), &key, value)?;
            } else if !cfg.unset_user(tag.as_deref(), &key)? {
                println!("{key} was not set{}", tag.as_ref().map(|t| format!(" for {t}")).unwrap_or_default());
                return Ok(());
            }
            cfg.save_user()?;
            println!("{} {key}{} in {}", if sub == "set" { "set" } else { "unset" },
                     tag.as_ref().map(|t| format!(" for {t}")).unwrap_or_default(), cfg.user_path.display());
            Ok(())
        }
        other => Err(format!("unknown config command {other:?} (show, set, unset, list)")),
    }
}

fn image_run(args: &[String]) -> Result<(), String> {
    let opts = flags(args)?;
    const ALLOWED: &[&str] = &["--config", "--prompt", "--out", "--model",
        "--negative-prompt", "--size", "--steps", "--guidance", "--seed", "--image"];
    for key in opts.keys() {
        if !ALLOWED.contains(&key.as_str()) {
            return Err(format!("unknown image option {key}"));
        }
    }
    let cfg = ImageConfig::load(&PathBuf::from(required(&opts, "--config")?))?;
    let out = PathBuf::from(required(&opts, "--out")?);
    let out = if out.is_absolute() { out } else {
        std::env::current_dir().map_err(|e| e.to_string())?.join(out)
    };
    let request = ImageRequest {
        prompt: required(&opts, "--prompt")?,
        model: opts.get("--model").cloned(),
        negative_prompt: opts.get("--negative-prompt").cloned().unwrap_or_default(),
        size: opts.get("--size").cloned().unwrap_or_else(|| "512x512".into()),
        steps: opts.get("--steps").map(|v| v.parse()
            .map_err(|_| "invalid --steps".to_string())).transpose()?,
        guidance_scale: opts.get("--guidance").map(|v| v.parse()
            .map_err(|_| "invalid --guidance".to_string())).transpose()?,
        seed: opts.get("--seed").map(|v| v.parse()
            .map_err(|_| "invalid --seed".to_string())).unwrap_or(Ok(41))?,
        n: 1,
        response_format: "b64_json".into(),
        // --image REF edits REF instead of generating from text alone.
        image: opts.get("--image").map(|p| std::fs::canonicalize(p)
            .map_err(|e| format!("reading --image {p}: {e}"))).transpose()?,
    };
    let path = image::generate(&cfg, &request, &out)?;
    println!("{}", path.display());
    Ok(())
}

fn serve_images(args: &[String]) -> Result<(), String> {
    let opts = flags(args)?;
    for key in opts.keys() {
        if !["--config", "--host", "--port"].contains(&key.as_str()) {
            return Err(format!("unknown image server option {key}"));
        }
    }
    let cfg = ImageConfig::load(&PathBuf::from(required(&opts, "--config")?))?;
    let host = opts.get("--host").map(String::as_str).unwrap_or("127.0.0.1");
    let port: u16 = opts.get("--port").map(|v| v.parse()
        .map_err(|_| "invalid --port".to_string())).unwrap_or(Ok(52626))?;
    image::serve(cfg, host, port)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench::command(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("image") => image_run(&args[1..]),
        Some("serve-images") => serve_images(&args[1..]),
        Some("config") => config(&args[1..]),
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
        let resolve = |map: &HashMap<String, String>| {
            Config::with_user(serde_json::json!({}), PathBuf::from("/nonexistent/settings.json"))
                .resolve_with(None, None, map, &|_| None)
                .unwrap()
        };
        assert_eq!(npu_options(&map, &resolve(&map)).unwrap().prefill_route, "auto");
        assert_eq!(npu_options(&map, &resolve(&map)).unwrap().context_capacity, 4096);
        map.insert("--prefill-route".into(), "block".into());
        map.insert("--max-ctx".into(), "8192".into());
        assert_eq!(npu_options(&map, &resolve(&map)).unwrap().prefill_route, "block");
        assert_eq!(npu_options(&map, &resolve(&map)).unwrap().context_capacity, 8192);
        assert!(flags(&["--json".into()]).unwrap().contains_key("--json"));
    }
}
