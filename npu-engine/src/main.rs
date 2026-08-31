mod forward;
mod q4nx;
#[cfg(feature = "npu")]
mod decode;
#[cfg(feature = "npu")]
mod xrt;

use forward::{Model, KvState, LinearState, HIDDEN};
use q4nx::{bf16_to_f32, Q4nx};
use std::path::Path;

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

fn open_model(model_path: &Path) -> Model {
    let file = Q4nx::open(model_path).unwrap();
    // derive layer schedule from tensor names
    let mut n = 0;
    while file.tensors.contains_key(&format!("model.layer.{n}.input_layernorm.weight")) {
        n += 1;
    }
    let layer_types: Vec<String> = (0..n)
        .map(|l| {
            if file.tensors.contains_key(&format!("model.layer.{l}.linear_attn.qkv_proj.weight")) {
                "linear_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        })
        .collect();
    eprintln!("model format: {}", file.fmt);
    eprintln!("model: {} layers, schedule {:?}", n, layer_types);
    Model { file, layer_types }
}

/// prefill: returns (residual stream [T*2048], per-layer states)
fn run_prefill(m: &Model, ids: &[i64]) -> (Vec<f64>, Vec<LinearState>, Vec<KvState>) {
    let t = ids.len();
    let mut x = vec![0f64; t * HIDDEN];
    for (i, id) in ids.iter().enumerate() {
        let e = m.embed(*id as usize);
        x[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&e);
    }
    let mut lin_states = Vec::new();
    let mut kv_states = Vec::new();
    for (l, lt) in m.layer_types.clone().iter().enumerate() {
        if lt == "linear_attention" {
            lin_states.push(m.linear_attn_prefill(l, &mut x, t));
        } else {
            kv_states.push(m.full_attn_prefill(l, &mut x, t));
        }
        m.moe(l, &mut x, t);
        eprintln!("  layer {l} ({lt}) done");
    }
    (x, lin_states, kv_states)
}

fn final_hidden(m: &Model, x_last: &[f64]) -> Vec<f32> {
    let nw = m.file.bf16("model.norm.weight");
    forward::rms_norm(x_last, &nw, HIDDEN).iter().map(|v| *v as f32).collect()
}

fn cmd_prefill(model_path: &Path, ref_dir: &Path) {
    let m = open_model(model_path);
    let ids = read_i64(&ref_dir.join("prompt_ids.i64"));
    let t = ids.len();
    let (x, _, _) = run_prefill(&m, &ids);
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
    let raw = std::fs::read(path).unwrap();
    let conv: Vec<f32> = raw[..49152]
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let s: Vec<f64> = raw[49152..49152 + 32 * 128 * 128 * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
        .collect();
    LinearState { conv, s }
}

/// Load the captured KV pack (k bf16[T,512] @0, v @byte 1073152), T=11.
fn load_kv_state(path: &Path, t: usize) -> KvState {
    let raw = std::fs::read(path).unwrap();
    let rd = |off: usize, n: usize| -> Vec<f64> {
        raw[off..off + n * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect()
    };
    KvState { k: rd(0, t * 512), v: rd(1073152, t * 512) }
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
    let (x, _, _) = run_prefill(&m, &ids);
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("prefill") => cmd_prefill(Path::new(&args[2]), Path::new(&args[3])),
        Some("decode") => cmd_decode(Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4])),
        Some("run") => cmd_run(Path::new(&args[2]), &args[3]),
        #[cfg(feature = "npu")]
        Some("npu") => {
            if let Err(e) = decode::run_config(Path::new(&args[2])) {
                eprintln!("npu driver error: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: open-qwen-npu prefill <model.q4nx> <ref_dir>");
            eprintln!("       open-qwen-npu decode  <model.q4nx> <ref_dir> <caps_dir>");
            eprintln!("       open-qwen-npu run     <model.q4nx> <ids,csv>");
            #[cfg(feature = "npu")]
            eprintln!("       open-qwen-npu npu     <driver-config>   (NPU decode driver)");
            std::process::exit(2);
        }
    }
}
