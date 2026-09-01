mod backend;
mod forward;
mod generate_5li3;
mod generate_l30;
mod generate_l40;
mod l30_buffers;
mod mock_backend;
mod placeholder_tokenizer;
mod pools;
mod q4nx;
mod sampler;
mod server;
mod state_io;
mod tokenizer;
#[cfg(feature = "npu")]
mod decode;
#[cfg(feature = "npu")]
mod npu_lock;
#[cfg(feature = "npu")]
mod xrt;

use backend::{Backend, GenParams};
use forward::{final_hidden, open_model, run_prefill, KvState, LinearState, HIDDEN};
use std::collections::HashSet;
use std::path::Path;
use tokenizer::{ChatMessage, Tokenizer};

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i64(path: &Path) -> Vec<i64> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mut num = 0f64;
    let mut da = 0f64;
    let mut db = 0f64;
    for i in 0..a.len() {
        let x = a[i] as f64 - ma;
        let y = b[i] as f64 - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt())
}

fn max_absdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn cmd_prefill(model_path: &Path, ref_dir: &Path) {
    let m = open_model(model_path);
    let ids = read_i64(&ref_dir.join("prompt_ids.i64"));
    let t = ids.len();
    let (x, _) = run_prefill(&m, &ids);
    let hn = final_hidden(&m, &x[(t - 1) * HIDDEN..]);
    let ref_hn = read_f32(&ref_dir.join("prefill_final_hidden.f32"));
    println!(
        "final hidden: corr {:.6} maxdiff {:.5}",
        corr(&hn, &ref_hn),
        max_absdiff(&hn, &ref_hn)
    );
    let logits = m.logits(&x[(t - 1) * HIDDEN..]);
    let ref_lg = read_f32(&ref_dir.join("prefill_logits.f32"));
    let my_arg = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let ref_arg = ref_lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    println!(
        "logits: corr {:.6} maxdiff {:.5} argmax {}/{}",
        corr(&logits, &ref_lg),
        max_absdiff(&logits, &ref_lg),
        my_arg,
        ref_arg
    );
}

/// Load a captured 3MB linear-state buffer (conv bf16[3,8192] + S fp32[32,128,128]).
fn load_linear_state(path: &Path) -> LinearState {
    state_io::load_linear_state_bytes(&std::fs::read(path).unwrap())
}

/// Load the captured KV pack (k bf16[T,512] @0, v @byte 1073152), T=11.
fn load_kv_state(path: &Path, t: usize) -> KvState {
    state_io::load_kv_state_bytes(&std::fs::read(path).unwrap(), t)
}

fn cmd_decode(model_path: &Path, ref_dir: &Path, caps_dir: &Path) {
    let m = open_model(model_path);
    let toks = read_i64(&ref_dir.join("decode_tokens.i64"));
    let ref_hns = read_f32(&ref_dir.join("decode_hns.f32"));
    let mut st0 = load_linear_state(&caps_dir.join("000898.bo"));
    let mut st1 = load_linear_state(&caps_dir.join("000900.bo"));
    let mut kv = load_kv_state(&caps_dir.join("000902.bo"), 11);
    let mut pos = 11usize;
    for (bi, tok) in toks.iter().enumerate() {
        let mut x = m.embed(*tok as usize);
        m.linear_attn_decode(0, &mut x, &mut st0);
        m.moe(0, &mut x, 1);
        m.linear_attn_decode(1, &mut x, &mut st1);
        m.moe(1, &mut x, 1);
        m.full_attn_decode(2, &mut x, &mut kv, pos);
        m.moe(2, &mut x, 1);
        let hn = final_hidden(&m, &x);
        let r = &ref_hns[bi * HIDDEN..(bi + 1) * HIDDEN];
        println!(
            "decode block {} (tok {tok}, pos {pos}): corr {:.6} maxdiff {:.5}",
            bi + 2,
            corr(&hn, r),
            max_absdiff(&hn, r)
        );
        pos += 1;
    }
}

/// Run prefill on an arbitrary model with a comma-separated token list and
/// report logits health (the interval-3 acid test: FLM's engine overflows to
/// NaN on these configs; correct math keeps |logit| ~ 10).
fn cmd_run(model_path: &Path, ids_csv: &str) {
    let m = open_model(model_path);
    let ids: Vec<i64> = ids_csv.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let t = ids.len();
    let (x, _) = run_prefill(&m, &ids);
    let logits = m.logits(&x[(t - 1) * HIDDEN..]);
    let finite = logits.iter().all(|v| v.is_finite());
    let absmax = logits.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    println!("logits: finite={finite} absmax={absmax:.3}");
    println!("top5: {:?}", idx[..5].iter().map(|i| (*i, logits[*i])).collect::<Vec<_>>());
    // hidden-state health per position
    let hn_absmax = x.iter().map(|v| v.abs()).fold(0f64, f64::max);
    println!("residual-stream absmax: {hn_absmax:.3}");
}

/// `tokenize <model_dir> <text>` — print token ids (space-separated) for
/// cross-checking against `tools/kernel-interp/tokenizer.py`.
fn cmd_tokenize(model_dir: &Path, text: &str) {
    let tok = Tokenizer::load(model_dir).expect("load tokenizer");
    let ids = tok.encode(text).expect("encode");
    let ids_str: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    println!("{}", ids_str.join(" "));
    println!("{}", tok.decode(&ids).expect("decode"));
}

/// `chattemplate <model_dir> <user_msg> [--no-think]` — print the rendered
/// chat template for a single user message, for cross-checking against
/// `Qwen36Tokenizer.apply_chat_template`.
fn cmd_chattemplate(model_dir: &Path, user_msg: &str, enable_thinking: bool, system_msg: Option<&str>) {
    let tok = Tokenizer::load(model_dir).expect("load tokenizer");
    let mut msgs = Vec::new();
    if let Some(s) = system_msg {
        msgs.push(ChatMessage { role: "system".to_string(), content: s.to_string() });
    }
    msgs.push(ChatMessage { role: "user".to_string(), content: user_msg.to_string() });
    let rendered = tok.apply_chat_template(&msgs, true, enable_thinking).expect("render chat template");
    print!("{rendered}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("prefill") => cmd_prefill(Path::new(&args[2]), Path::new(&args[3])),
        Some("decode") => cmd_decode(Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4])),
        Some("run") => cmd_run(Path::new(&args[2]), &args[3]),
        Some("tokenize") => cmd_tokenize(Path::new(&args[2]), &args[3]),
        Some("chattemplate") => {
            let no_think = args.iter().any(|a| a == "--no-think");
            let system = args.iter().position(|a| a == "--system").and_then(|i| args.get(i + 1)).map(|s| s.as_str());
            cmd_chattemplate(Path::new(&args[2]), &args[3], !no_think, system)
        }
        #[cfg(feature = "npu")]
        Some("npu") => {
            if let Err(e) = decode::run_config(Path::new(&args[2])) {
                eprintln!("npu driver error: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "npu")]
        Some("step-test") => {
            // In-process smoke test for decode::load_resident/step_bytes — no
            // subprocess, no stdin protocol, no temp files. `args[4]` (output
            // path) lets this be byte-diffed against a real capture from the
            // old subprocess path for the same config+act.
            let cfg = Path::new(&args[2]);
            let act = std::fs::read(&args[3]).expect("read act file");
            let (mut drv, prog) = decode::load_resident(cfg).expect("load_resident");
            let hidden = drv.step_bytes(&prog, &act).expect("step_bytes");
            std::fs::write(&args[4], hidden).expect("write hidden output");
            eprintln!("step-test: wrote {} bytes to {}", hidden.len(), args[4]);
        }
        #[cfg(feature = "npu")]
        Some("li3-run") => {
            let prompt = args.get(2).cloned().unwrap_or_else(|| "What is the capital of France?".to_string());
            let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
            let cfg = generate_5li3::Li3Config::default();
            let model_dir = cfg.model_path.parent().unwrap().to_path_buf();
            run_backend_cli(&model_dir, &prompt, max_tokens, generate_5li3::Li3Backend::new(cfg));
        }
        #[cfg(feature = "npu")]
        Some("l40-run") => {
            let prompt = args.get(2).cloned().unwrap_or_else(|| "What is the capital of France?".to_string());
            let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
            let cfg = generate_l40::L40Config::default();
            let model_dir = cfg.model_path.parent().unwrap().to_path_buf();
            run_backend_cli(&model_dir, &prompt, max_tokens, generate_l40::L40Backend::new(cfg));
        }
        #[cfg(feature = "npu")]
        Some("l30-run") => {
            let prompt = args.get(2).cloned().unwrap_or_else(|| "What is the capital of France?".to_string());
            let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
            let model_dir = Path::new(generate_l30::DEFAULT_MODEL).parent().unwrap().to_path_buf();
            run_backend_cli(&model_dir, &prompt, max_tokens, generate_l30::L30Backend::open_default());
        }
        Some("serve") => {
            let mut cfg = server::ServerConfig::from_env();
            let backend_flag = args.iter().position(|a| a == "--backend").and_then(|i| args.get(i + 1)).map(|s| s.as_str());
            let backend: Box<dyn Backend + Send> = match backend_flag.unwrap_or("mock") {
                "mock" => Box::new(mock_backend::MockBackend::new()),
                #[cfg(feature = "npu")]
                "li3" => {
                    cfg.model_dir = Some(generate_5li3::Li3Config::default().model_path.parent().unwrap().to_path_buf());
                    cfg.backend_name = "li3".to_string();
                    Box::new(generate_5li3::Li3Backend::new(generate_5li3::Li3Config::default()))
                }
                #[cfg(feature = "npu")]
                "l40" => {
                    let l40cfg = generate_l40::L40Config::default();
                    cfg.model_dir = Some(l40cfg.model_path.parent().unwrap().to_path_buf());
                    cfg.backend_name = "l40".to_string();
                    if std::env::var("FLM_MODEL_ID").is_err() {
                        cfg.model_id = "qwen3.6-35b-a3b-npu".to_string();
                    }
                    Box::new(generate_l40::L40Backend::new(l40cfg))
                }
                #[cfg(feature = "npu")]
                "l30" => {
                    cfg.model_dir = Some(Path::new(generate_l30::DEFAULT_MODEL).parent().unwrap().to_path_buf());
                    cfg.backend_name = "l30".to_string();
                    Box::new(generate_l30::L30Backend::open_default())
                }
                other => {
                    eprintln!("unknown --backend {other} (mock{})", if cfg!(feature = "npu") { ", li3, l30, l40" } else { " — build with --features npu for li3/l30/l40" });
                    std::process::exit(2);
                }
            };
            if let Err(e) = server::serve(backend, cfg) {
                eprintln!("server error: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: open-qwen-npu prefill      <model.q4nx> <ref_dir>");
            eprintln!("       open-qwen-npu decode       <model.q4nx> <ref_dir> <caps_dir>");
            eprintln!("       open-qwen-npu run          <model.q4nx> <ids,csv>");
            eprintln!("       open-qwen-npu tokenize     <model_dir> <text>");
            eprintln!("       open-qwen-npu chattemplate <model_dir> <user_msg> [--no-think]");
            eprintln!("       open-qwen-npu serve        [--backend mock|li3|l30]   (OpenAI-compatible HTTP server)");
            #[cfg(feature = "npu")]
            {
                eprintln!("       open-qwen-npu npu          <driver-config>   (NPU decode driver)");
                eprintln!("       open-qwen-npu li3-run      [prompt] [max_tokens]   (5li3 resident generate loop)");
                eprintln!("       open-qwen-npu l30-run      [prompt] [max_tokens]   (l30 streamed generate loop)");
                eprintln!("       open-qwen-npu l40-run      [prompt] [max_tokens]   (base 40L resident loop, NPU prefill)");
            }
            std::process::exit(2);
        }
    }
}

/// Shared CLI driver for any `Backend`: render the chat template, generate,
/// stream-print the decoded text. Used by both `li3-run` and `l30-run` so
/// they behave identically from the command line.
#[cfg(feature = "npu")]
fn run_backend_cli(model_dir: &Path, prompt: &str, max_tokens: usize, mut backend: impl Backend) {
    let tok = Tokenizer::load(model_dir).expect("load tokenizer");
    let msgs = vec![ChatMessage { role: "user".to_string(), content: prompt.to_string() }];
    let rendered = tok.apply_chat_template(&msgs, true, true).expect("render chat template");
    let ids = tok.encode(&rendered).expect("encode prompt");
    eprintln!("prompt ({} tokens): {prompt:?}", ids.len());
    let sampler = sampler::Sampler::new(0.0, 0, 1.0, 1.0, None); // greedy
    let stop_ids: HashSet<u32> = tokenizer::EOS_TOKEN_IDS.iter().copied().collect();
    let mut params = GenParams { max_tokens, sampler, stop_ids };
    let mut generated = Vec::new();
    let result = backend.generate(&ids, &mut params, &mut |t| generated.push(t));
    match result {
        Ok(()) => {
            println!("{}", tok.decode(&generated).unwrap_or_default());
            eprintln!("[{} tokens generated]", generated.len());
        }
        Err(e) => {
            eprintln!("generate error: {e}");
            std::process::exit(1);
        }
    }
}
