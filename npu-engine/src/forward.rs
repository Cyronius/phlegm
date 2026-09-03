//! CPU reference forward for Qwen3.6-MoE (q4nx weights), ported from the
//! capture-verified tools/kernel-interp/{full_forward,decode_step}.py.
//!
//! Residual stream is f64; weights are dequantized f32; dot products use f64
//! accumulation. Layer schedule is config-driven (layer_types), which is the
//! whole point: interval-3 models run with the same code path as interval-4.

use crate::q4nx::Q4nx;
use std::path::Path;

pub const HIDDEN: usize = 2048;
pub const VOCAB: usize = 248320;

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// x / sqrt(mean(x^2) + eps) * w, per row of `dim`. Returns f64.
pub fn rms_norm(x: &[f64], w: &[f32], dim: usize) -> Vec<f64> {
    let mut out = vec![0f64; x.len()];
    for (row_o, row_x) in out.chunks_mut(dim).zip(x.chunks(dim)) {
        let ms = row_x.iter().map(|v| v * v).sum::<f64>() / dim as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..dim {
            row_o[i] = row_x[i] * inv * w[i] as f64;
        }
    }
    out
}

/// out[t, o] = sum_i act[t, i] * w[o, i]   (act [T, in] f32, w [out, in] f32)
pub fn matmul(act: &[f32], w: &[f32], t: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * out_dim];
    for ti in 0..t {
        let a = &act[ti * in_dim..(ti + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0f64;
            for i in 0..in_dim {
                acc += a[i] as f64 * wr[i] as f64;
            }
            out[ti * out_dim + o] = acc as f32;
        }
    }
    out
}

fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|v| *v as f32).collect()
}

pub struct LinearState {
    /// last 3 tokens' post-qkv (pre-conv), [3][8192]
    pub conv: Vec<f32>,
    /// GDN state S[h][dk][dv], [32*128*128]
    pub s: Vec<f64>,
}

pub struct KvState {
    /// roped+normed k, [tokens][2][256]
    pub k: Vec<f64>,
    /// v, [tokens][2][256]
    pub v: Vec<f64>,
}

/// One layer's decode-ready state, tagged by layer kind. Carries the layer
/// index it belongs to so a caller building per-layer buffers (e.g. serializing
/// `state_L{l}.bin` files) doesn't need separate bookkeeping to line a state
/// back up with its layer.
pub enum LayerState {
    Linear(LinearState),
    Full(KvState),
}

pub struct Model {
    pub file: Q4nx,
    pub layer_types: Vec<String>,
}

/// Open a `.q4nx` model file and derive its layer schedule from tensor names
/// (linear-attention vs full-attention, per layer).
pub fn open_model(model_path: &Path) -> Model {
    let file = Q4nx::open(model_path).unwrap();
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

/// CPU prefill over `ids`: returns the residual stream (`[T*HIDDEN]`, f64) and
/// each layer's decode-ready state, in layer order.
pub fn run_prefill(m: &Model, ids: &[i64]) -> (Vec<f64>, Vec<LayerState>) {
    let t = ids.len();
    let mut x = vec![0f64; t * HIDDEN];
    for (i, id) in ids.iter().enumerate() {
        let e = m.embed(*id as usize);
        x[i * HIDDEN..(i + 1) * HIDDEN].copy_from_slice(&e);
    }
    let mut states = Vec::with_capacity(m.layer_types.len());
    for (l, lt) in m.layer_types.clone().iter().enumerate() {
        if lt == "linear_attention" {
            states.push(LayerState::Linear(m.linear_attn_prefill(l, &mut x, t)));
        } else {
            states.push(LayerState::Full(m.full_attn_prefill(l, &mut x, t)));
        }
        m.moe(l, &mut x, t);
        eprintln!("  layer {l} ({lt}) done");
    }
    (x, states)
}

/// Final-norm hidden state (pre-lm_head) for one residual-stream row.
pub fn final_hidden(m: &Model, x_last: &[f64]) -> Vec<f32> {
    let nw = m.file.bf16("model.norm.weight");
    rms_norm(x_last, &nw, HIDDEN).iter().map(|v| *v as f32).collect()
}

impl Model {
    pub fn layer_name(&self, layer: usize, suffix: &str) -> String {
        format!("model.layer.{layer}.{suffix}")
    }

    fn dq(&self, layer: usize, suffix: &str, out_dim: usize, in_dim: usize) -> Vec<f32> {
        self.file.dequant_q4(&self.layer_name(layer, suffix), out_dim, in_dim)
    }

    pub fn embed(&self, token: usize) -> Vec<f64> {
        self.file
            .embed_row("model.embed_tokens.weight", token, HIDDEN)
            .iter()
            .map(|v| *v as f64)
            .collect()
    }

    /// Linear-attention block over T tokens (prefill). Updates x_res in place,
    /// returns the decode-ready state.
    pub fn linear_attn_prefill(&self, layer: usize, x_res: &mut [f64], t: usize) -> LinearState {
        let ln = self.file.bf16(&self.layer_name(layer, "input_layernorm.weight"));
        let x = to_f32(&rms_norm(x_res, &ln, HIDDEN));
        let wqkv = self.dq(layer, "linear_attn.qkv_proj.weight", 8192, 2048);
        let wz = self.dq(layer, "self_attn.gate_proj.weight", 4096, 2048);
        let wout = self.dq(layer, "linear_attn.ssm_out_proj.weight", 2048, 4096);
        let convw = self.file.bf16(&self.layer_name(layer, "linear_attn.ssm_conv1d.weight")); // [4,8192]

        let qkv = matmul(&x, &wqkv, t, 2048, 8192);
        let z: Vec<f32> = matmul(&x, &wz, t, 2048, 4096).iter().map(|v| silu(*v)).collect();

        // causal depthwise conv k=4 with 3 zero-pad rows, then silu + qk l2norm
        let mut conv = vec![0f32; t * 8192];
        for ti in 0..t {
            for c in 0..8192 {
                let mut acc = 0f64;
                for k in 0..4 {
                    let src = ti as isize + k as isize - 3;
                    if src >= 0 {
                        acc += convw[k * 8192 + c] as f64 * qkv[src as usize * 8192 + c] as f64;
                    }
                }
                conv[ti * 8192 + c] = silu(acc as f32);
            }
        }
        let (decay, beta) = self.decay_beta(layer, &x, t);

        let mut s = vec![0f64; 32 * 128 * 128];
        let mut o = vec![0f64; t * 4096];
        for ti in 0..t {
            let row = &conv[ti * 8192..(ti + 1) * 8192];
            let (q, k, v) = split_qkv_l2norm(row);
            gdn_step(&mut s, &q, &k, &v, &decay[ti * 32..], &beta[ti * 32..], &mut o[ti * 4096..(ti + 1) * 4096]);
        }
        self.gdn_out(layer, &o, &z, x_res, &wout, t);

        // decode state: last 3 tokens' qkv
        let mut conv_state = vec![0f32; 3 * 8192];
        for k in 0..3 {
            let src = (t + k).saturating_sub(3); // t-3+k for t>=3
            conv_state[k * 8192..(k + 1) * 8192].copy_from_slice(&qkv[src * 8192..(src + 1) * 8192]);
        }
        LinearState { conv: conv_state, s }
    }

    /// One-token linear-attention decode step. Updates x_res and state.
    pub fn linear_attn_decode(&self, layer: usize, x_res: &mut [f64], st: &mut LinearState) {
        let ln = self.file.bf16(&self.layer_name(layer, "input_layernorm.weight"));
        let x = to_f32(&rms_norm(x_res, &ln, HIDDEN));
        let wqkv = self.dq(layer, "linear_attn.qkv_proj.weight", 8192, 2048);
        let wz = self.dq(layer, "self_attn.gate_proj.weight", 4096, 2048);
        let wout = self.dq(layer, "linear_attn.ssm_out_proj.weight", 2048, 4096);
        let convw = self.file.bf16(&self.layer_name(layer, "linear_attn.ssm_conv1d.weight"));

        let qkv = matmul(&x, &wqkv, 1, 2048, 8192);
        let z: Vec<f32> = matmul(&x, &wz, 1, 2048, 4096).iter().map(|v| silu(*v)).collect();
        let mut conv = vec![0f32; 8192];
        for c in 0..8192 {
            let mut acc = 0f64;
            for k in 0..3 {
                acc += convw[k * 8192 + c] as f64 * st.conv[k * 8192 + c] as f64;
            }
            acc += convw[3 * 8192 + c] as f64 * qkv[c] as f64;
            conv[c] = silu(acc as f32);
        }
        let (decay, beta) = self.decay_beta(layer, &x, 1);
        let (q, k, v) = split_qkv_l2norm(&conv);
        let mut o = vec![0f64; 4096];
        gdn_step(&mut st.s, &q, &k, &v, &decay, &beta, &mut o);
        self.gdn_out(layer, &o, &z, x_res, &wout, 1);
        // roll conv state
        st.conv.copy_within(8192..3 * 8192, 0);
        st.conv[2 * 8192..].copy_from_slice(&qkv);
    }

    fn decay_beta(&self, layer: usize, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        // wa/wb stored [2048, 32] (already transposed vs HF)
        let wa = self.file.bf16(&self.layer_name(layer, "linear_attn.ssm_alpha_proj.weight"));
        let wb = self.file.bf16(&self.layer_name(layer, "linear_attn.ssm_beta_proj.weight"));
        let a_log = self.file.f32(&self.layer_name(layer, "linear_attn.ssm_a"));
        let dt_bias = self.file.f32(&self.layer_name(layer, "linear_attn.ssm_dt.bias"));
        let mut decay = vec![0f32; t * 32];
        let mut beta = vec![0f32; t * 32];
        for ti in 0..t {
            for h in 0..32 {
                let mut a = 0f64;
                let mut b = 0f64;
                for i in 0..2048 {
                    a += x[ti * 2048 + i] as f64 * wa[i * 32 + h] as f64;
                    b += x[ti * 2048 + i] as f64 * wb[i * 32 + h] as f64;
                }
                let sp = (1.0 + (a + dt_bias[h] as f64).exp()).ln(); // softplus
                // file ssm_a is pre-baked as -exp(A_log); do NOT re-exponentiate
                decay[ti * 32 + h] = (a_log[h] as f64 * sp).exp() as f32;
                beta[ti * 32 + h] = sigmoid64(b) as f32;
            }
        }
        (decay, beta)
    }

    /// o -> RMSNorm128(o)*ssm_norm_w * z -> @ Wout^T, accumulate into x_res.
    fn gdn_out(&self, layer: usize, o: &[f64], z: &[f32], x_res: &mut [f64], wout: &[f32], t: usize) {
        let nw = self.file.bf16(&self.layer_name(layer, "linear_attn.ssm_norm.weight")); // [128]
        let mut og = vec![0f32; t * 4096];
        for ti in 0..t {
            for h in 0..32 {
                let seg = &o[ti * 4096 + h * 128..ti * 4096 + (h + 1) * 128];
                let ms = seg.iter().map(|v| v * v).sum::<f64>() / 128.0;
                let inv = 1.0 / (ms + 1e-6).sqrt();
                for d in 0..128 {
                    let idx = ti * 4096 + h * 128 + d;
                    og[idx] = (seg[d] * inv * nw[d] as f64) as f32 * z[idx];
                }
            }
        }
        let proj = matmul(&og, wout, t, 4096, 2048);
        for i in 0..t * 2048 {
            x_res[i] += proj[i] as f64;
        }
    }

    /// MoE block: router softmax top-8 renorm + experts + shared expert.
    pub fn moe(&self, layer: usize, x_res: &mut [f64], t: usize) {
        let postln = self.file.bf16(&self.layer_name(layer, "post_attention_layernorm.weight"));
        let xm = to_f32(&rms_norm(x_res, &postln, HIDDEN));
        let router = self.file.bf16(&self.layer_name(layer, "moe_router.weight")); // [2048, 256]
        let up_all = self.file.raw(&self.layer_name(layer, "mlp.up_exps_proj.weight"));
        let gt_all = self.file.raw(&self.layer_name(layer, "mlp.gate_exps_proj.weight"));
        let dn_all = self.file.raw(&self.layer_name(layer, "mlp.down_exps_proj.weight"));

        for ti in 0..t {
            let xt = &xm[ti * 2048..(ti + 1) * 2048];
            // router logits + softmax
            let mut lg = vec![0f64; 256];
            for e in 0..256 {
                let mut acc = 0f64;
                for i in 0..2048 {
                    acc += xt[i] as f64 * router[i * 256 + e] as f64;
                }
                lg[e] = acc;
            }
            let mx = lg.iter().cloned().fold(f64::MIN, f64::max);
            let mut p: Vec<f64> = lg.iter().map(|v| (v - mx).exp()).collect();
            let sum: f64 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= sum;
            }
            let mut idx: Vec<usize> = (0..256).collect();
            idx.sort_by(|a, b| p[*b].partial_cmp(&p[*a]).unwrap());
            let top = &idx[..8];
            let wsum: f64 = top.iter().map(|e| p[*e]).sum();

            let mut out = vec![0f64; 2048];
            let esz = 128 * self.file.chunk_bytes;
            for &e in top {
                let gt = self.file.dequant_q4_tile(&gt_all[e * esz..(e + 1) * esz], 512, 2048);
                let up = self.file.dequant_q4_tile(&up_all[e * esz..(e + 1) * esz], 512, 2048);
                let dn = self.file.dequant_q4_tile(&dn_all[e * esz..(e + 1) * esz], 2048, 512);
                let g = matmul(xt, &gt, 1, 2048, 512);
                let u = matmul(xt, &up, 1, 2048, 512);
                let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
                let d = matmul(&h, &dn, 1, 512, 2048);
                let w = p[e] / wsum;
                for i in 0..2048 {
                    out[i] += w * d[i] as f64;
                }
            }
            // shared expert * sigmoid(shared_gate)
            let wsg = self.dq(layer, "mlp.share_gate_exps_proj.weight", 512, 2048);
            let wsu = self.dq(layer, "mlp.share_up_exps_proj.weight", 512, 2048);
            let wsd = self.dq(layer, "mlp.share_down_exps_proj.weight", 2048, 512);
            let g = matmul(xt, &wsg, 1, 2048, 512);
            let u = matmul(xt, &wsu, 1, 2048, 512);
            let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let sh = matmul(&h, &wsd, 1, 512, 2048);
            let sgw = self.file.bf16(&self.layer_name(layer, "shared_expert_gate.weight"));
            let mut sg = 0f64;
            for i in 0..2048 {
                sg += xt[i] as f64 * sgw[i] as f64;
            }
            let sg = sigmoid64(sg);
            for i in 0..2048 {
                x_res[ti * 2048 + i] += out[i] + sg * sh[i] as f64;
            }
        }
    }

    /// Full-attention block over T tokens (prefill). pos = absolute positions.
    /// Returns the KV cache for decode.
    pub fn full_attn_prefill(&self, layer: usize, x_res: &mut [f64], t: usize) -> KvState {
        let (q, g, k, v) = self.attn_proj(layer, x_res, t);
        let mut kv = KvState { k: Vec::new(), v: Vec::new() };
        for ti in 0..t {
            kv.k.extend_from_slice(&q_rope(&k[ti * 512..(ti + 1) * 512], ti as f64, 2));
            kv.v.extend_from_slice(&v[ti * 512..(ti + 1) * 512].iter().map(|x| *x as f64).collect::<Vec<_>>());
        }
        let mut og = vec![0f32; t * 4096];
        for ti in 0..t {
            let qr = q_rope(&q[ti * 4096..(ti + 1) * 4096], ti as f64, 16);
            for h in 0..16 {
                let qh = &qr[h * 256..(h + 1) * 256];
                let kvh = h / 8;
                let mut sc = vec![0f64; ti + 1];
                for tj in 0..=ti {
                    let kh = &kv.k[tj * 512 + kvh * 256..tj * 512 + (kvh + 1) * 256];
                    sc[tj] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f64>() / 16.0;
                }
                softmax_inplace(&mut sc);
                for d in 0..256 {
                    let mut acc = 0f64;
                    for tj in 0..=ti {
                        acc += sc[tj] * kv.v[tj * 512 + kvh * 256 + d];
                    }
                    let idx = ti * 4096 + h * 256 + d;
                    og[idx] = (acc * sigmoid64(g[idx] as f64)) as f32;
                }
            }
        }
        let wo = self.dq(layer, "self_attn.o_proj.weight", 2048, 4096);
        let proj = matmul(&og, &wo, t, 4096, 2048);
        // Diagnostic: OPEN_QWEN_ZERO_FULL_ATTN=1 keeps the KV cache but drops the
        // block's contribution to the residual stream — the "FLM contributes
        // nothing from the full-attention block on interval-3" hypothesis, so a
        // suspect NPU token stream can be checked against it (see
        // [[flm-capture-oracle]]).
        if std::env::var("OPEN_QWEN_ZERO_FULL_ATTN").is_err() {
            for i in 0..t * 2048 {
                x_res[i] += proj[i] as f64;
            }
        }
        kv
    }

    /// One-token full-attention decode step at absolute position `pos`.
    pub fn full_attn_decode(&self, layer: usize, x_res: &mut [f64], kv: &mut KvState, pos: usize) {
        let (q, g, k, v) = self.attn_proj(layer, x_res, 1);
        kv.k.extend_from_slice(&q_rope(&k, pos as f64, 2));
        kv.v.extend_from_slice(&v.iter().map(|x| *x as f64).collect::<Vec<_>>());
        let ntok = kv.k.len() / 512;
        let qr = q_rope(&q, pos as f64, 16);
        let mut og = vec![0f32; 4096];
        for h in 0..16 {
            let qh = &qr[h * 256..(h + 1) * 256];
            let kvh = h / 8;
            let mut sc = vec![0f64; ntok];
            for tj in 0..ntok {
                let kh = &kv.k[tj * 512 + kvh * 256..tj * 512 + (kvh + 1) * 256];
                sc[tj] = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f64>() / 16.0;
            }
            softmax_inplace(&mut sc);
            for d in 0..256 {
                let mut acc = 0f64;
                for tj in 0..ntok {
                    acc += sc[tj] * kv.v[tj * 512 + kvh * 256 + d];
                }
                og[h * 256 + d] = (acc * sigmoid64(g[h * 256 + d] as f64)) as f32;
            }
        }
        let wo = self.dq(layer, "self_attn.o_proj.weight", 2048, 4096);
        let proj = matmul(&og, &wo, 1, 4096, 2048);
        for i in 0..2048 {
            x_res[i] += proj[i] as f64;
        }
    }

    /// q (per-head normed), gate (raw), k (normed), v — file q_proj is planar [q|gate].
    fn attn_proj(&self, layer: usize, x_res: &[f64], t: usize) -> (Vec<f64>, Vec<f32>, Vec<f64>, Vec<f32>) {
        let ln = self.file.bf16(&self.layer_name(layer, "input_layernorm.weight"));
        let x = to_f32(&rms_norm(x_res, &ln, HIDDEN));
        let wqg = self.dq(layer, "self_attn.q_proj.weight", 8192, 2048);
        let wk = self.dq(layer, "self_attn.k_proj.weight", 512, 2048);
        let wv = self.dq(layer, "self_attn.v_proj.weight", 512, 2048);
        let qn = self.file.bf16(&self.layer_name(layer, "self_attn.q_norm.weight"));
        let kn = self.file.bf16(&self.layer_name(layer, "self_attn.k_norm.weight"));
        let qg = matmul(&x, &wqg, t, 2048, 8192);
        let mut q = vec![0f64; t * 4096];
        let mut g = vec![0f32; t * 4096];
        for ti in 0..t {
            for i in 0..4096 {
                g[ti * 4096 + i] = qg[ti * 8192 + 4096 + i];
            }
            let qrow: Vec<f64> = qg[ti * 8192..ti * 8192 + 4096].iter().map(|v| *v as f64).collect();
            let qn64 = rms_norm(&qrow, &qn, 256);
            q[ti * 4096..(ti + 1) * 4096].copy_from_slice(&qn64);
        }
        let kraw = matmul(&x, &wk, t, 2048, 512);
        let mut k = vec![0f64; t * 512];
        for ti in 0..t {
            let krow: Vec<f64> = kraw[ti * 512..(ti + 1) * 512].iter().map(|v| *v as f64).collect();
            k[ti * 512..(ti + 1) * 512].copy_from_slice(&rms_norm(&krow, &kn, 256));
        }
        let v = matmul(&x, &wv, t, 2048, 512);
        (q, g, k, v)
    }

    /// Final norm + q8 lm_head -> logits [VOCAB].
    pub fn logits(&self, x_last: &[f64]) -> Vec<f32> {
        let nw = self.file.bf16("model.norm.weight");
        let hn = to_f32(&rms_norm(x_last, &nw, HIDDEN));
        let lm = self.file.raw("lm_head.weight");
        let mut logits = vec![0f32; VOCAB];
        for g in 0..VOCAB / 32 {
            let rows = self.file.dequant_q8_rows(lm, g * 32, 32);
            for r in 0..32 {
                let wr = &rows[r * 2048..(r + 1) * 2048];
                let mut acc = 0f64;
                for i in 0..2048 {
                    acc += hn[i] as f64 * wr[i] as f64;
                }
                logits[g * 32 + r] = acc as f32;
            }
        }
        logits
    }
}

/// split conv row into q/k (per-head L2-normed, 16 heads x 128) and v (32 x 128).
fn split_qkv_l2norm(row: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let l2n = |seg: &[f32]| -> Vec<f64> {
        let ss = seg.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
        let inv = 1.0 / (ss + 1e-6).sqrt();
        seg.iter().map(|v| *v as f64 * inv).collect()
    };
    let mut q = Vec::with_capacity(2048);
    let mut k = Vec::with_capacity(2048);
    for h in 0..16 {
        q.extend(l2n(&row[h * 128..(h + 1) * 128]));
        k.extend(l2n(&row[2048 + h * 128..2048 + (h + 1) * 128]));
    }
    let v: Vec<f64> = row[4096..8192].iter().map(|v| *v as f64).collect();
    (q, k, v)
}

/// One gated-delta-rule step over all 32 v-heads (k-head = h/2).
fn gdn_step(s: &mut [f64], q: &[f64], k: &[f64], v: &[f64], decay: &[f32], beta: &[f32], o: &mut [f64]) {
    let scale = 1.0 / (128f64).sqrt();
    for h in 0..32 {
        let sh = &mut s[h * 128 * 128..(h + 1) * 128 * 128];
        let kk = &k[(h / 2) * 128..(h / 2 + 1) * 128];
        let qq = &q[(h / 2) * 128..(h / 2 + 1) * 128];
        let vh = &v[h * 128..(h + 1) * 128];
        let dec = decay[h] as f64;
        // S *= decay
        for x in sh.iter_mut() {
            *x *= dec;
        }
        // kv_mem[dv] = sum_dk S[dk,dv]*k[dk]; delta = beta*(v - kv_mem); S += k (x) delta
        let mut delta = [0f64; 128];
        for dv in 0..128 {
            let mut acc = 0f64;
            for dk in 0..128 {
                acc += sh[dk * 128 + dv] * kk[dk];
            }
            delta[dv] = beta[h] as f64 * (vh[dv] - acc);
        }
        for dk in 0..128 {
            let kdk = kk[dk];
            let row = &mut sh[dk * 128..(dk + 1) * 128];
            for dv in 0..128 {
                row[dv] += kdk * delta[dv];
            }
        }
        // o = (S^T q) / sqrt(128)
        for dv in 0..128 {
            let mut acc = 0f64;
            for dk in 0..128 {
                acc += sh[dk * 128 + dv] * qq[dk];
            }
            o[h * 128 + dv] = acc * scale;
        }
    }
}

/// Partial RoPE: first 64 dims of each 256-dim head, half-split pairs
/// (i, i+32), theta 1e7.
fn q_rope(x: &[f64], pos: f64, nheads: usize) -> Vec<f64> {
    let mut y = x.to_vec();
    for h in 0..nheads {
        let base = h * 256;
        for i in 0..32 {
            let freq = (1e7f64).powf(-(i as f64) / 32.0);
            let (sin, cos) = (pos * freq).sin_cos();
            let a = x[base + i];
            let b = x[base + 32 + i];
            y[base + i] = a * cos - b * sin;
            y[base + 32 + i] = b * cos + a * sin;
        }
    }
    y
}

fn softmax_inplace(x: &mut [f64]) {
    let mx = x.iter().cloned().fold(f64::MIN, f64::max);
    let mut sum = 0f64;
    for v in x.iter_mut() {
        *v = (*v - mx).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}
