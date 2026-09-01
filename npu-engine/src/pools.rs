//! Build NPU weight pools/packs/sides from a `.q4nx` file (the loader-
//! permutation step), ported from `tools/kernel-interp/build_pools.py`.
//!
//! Pool chunk laws (all verified against captured pools / HF weights, see the
//! Python module's docstring for the full derivation):
//!   standard matmul tensor `[out, in]`: pool chunk `c` covers
//!     `rows0 = 64*(c/per_band) + 32*(c%2)`, `per_band = in/128`
//!     `cols0 = 1024*((c/8) % max(1,in/1024)) + 256*((c/2)%4)`
//!   file raster: chunk `f` covers `rows0 = 32*(f/ncol)`, `cols0 = 256*(f%ncol)`
//!   -> permute file chunks into pool order by matching `(rows0, cols0)`.
//!   expert gate_up: pool = interleaved 163840B stripes `[up_k | gate_k]` x4/expert,
//!     within-stripe transpose `pool_c = 4*(f%8) + (f/8)`
//!   expert/shared down `[2048,512]`: `pool_c = 8*(rt/4) + 4*cg + rt%4` (`f = 2rt+cg`)
//!
//! Layer-pool layout (from L0/L1 captures), all byte offsets into the 512MB pool:
//!   `[0)`          gate_up expert stripes
//!   `[335544320)`  down experts (permuted per expert)
//!   `[503316480)`  share_up   `[503971840)` share_gate   `[504627200)` share_down
//!   `[505282560)`  main proj A (linear: qkv / full-attn: q_proj)   10485760 B
//!   `[515768320)`  main proj B (linear: z-gate / full-attn: o_proj) 5242880 B
//!   `[521011200)`  (full-attn: k_proj 655360) `[521666560)` (v_proj 655360)
//! lm_head pool: q8 chunks (8704B) in standard-in2048 pool order at 0; tail zeros.
//!
//! Only the q4_1 (FLM 1.0.2) format is supported today (`CH = 5120`), matching
//! `build_pools.py` — this is the format the NPU kernels being targeted were
//! built against. `Q4nx::raw()` never applies the FLM-1.0.3 reorder-undo (that
//! only affects `bf16()`/`f32()`/`dequant_q4()`), so it's the right source for
//! pool bytes regardless of on-disk format, exactly as in the Python version.

use crate::q4nx::Q4nx;

const CH: usize = 5120;

/// pool chunk index -> file chunk index (standard law <-> raster).
fn std_perm(nch: usize, out_dim: usize, in_dim: usize) -> Vec<usize> {
    let ncol = in_dim / 256;
    let per_band = in_dim / 128;
    let kgroups = (in_dim / 1024).max(1);
    (0..nch)
        .map(|c| {
            let rows0 = 64 * (c / per_band) + 32 * (c % 2);
            let cols0 = (1024 * ((c / 8) % kgroups) + 256 * ((c / 2) % 4)) % in_dim;
            (rows0 / 32) * ncol + cols0 / 256
        })
        .collect()
}

/// `out_chunk[c] = raw_chunk[perm[c]]`, `chunk_bytes`-wide chunks.
fn permute_chunks(raw: &[u8], perm: &[usize], chunk_bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; raw.len()];
    for (c, &f) in perm.iter().enumerate() {
        out[c * chunk_bytes..(c + 1) * chunk_bytes]
            .copy_from_slice(&raw[f * chunk_bytes..(f + 1) * chunk_bytes]);
    }
    out
}

fn down_perm() -> [usize; 128] {
    let mut perm = [0usize; 128];
    for c in 0..128 {
        let rt = 4 * (c / 8) + (c % 4);
        let cg = (c / 4) % 2;
        perm[c] = 2 * rt + cg;
    }
    perm
}

fn stripe_transpose() -> [usize; 32] {
    let mut perm = [0usize; 32];
    for c in 0..32 {
        let rt = c % 4;
        let cg = c / 4;
        perm[c] = 8 * rt + cg;
    }
    perm
}

fn put(buf: &mut [u8], off: usize, bytes: &[u8]) {
    buf[off..off + bytes.len()].copy_from_slice(bytes);
}

/// Reorder 32 `CH`-byte chunks of `src` by `tp` (a 32-entry stripe transpose),
/// writing into `dst` (both exactly `32 * CH` bytes).
fn apply32(dst: &mut [u8], src: &[u8], tp: &[usize; 32]) {
    for (c, &f) in tp.iter().enumerate() {
        dst[c * CH..(c + 1) * CH].copy_from_slice(&src[f * CH..(f + 1) * CH]);
    }
}

pub fn layer_name(layer: usize, suffix: &str) -> String {
    format!("model.layer.{layer}.{suffix}")
}

/// Build one layer's 512MB weight pool.
pub fn build_layer_pool(m: &Q4nx, layer: usize, full_attn: bool) -> Vec<u8> {
    const S: usize = 163840; // 32 chunks * CH
    let mut pool = vec![0u8; 536_870_912];

    let up = m.raw(&layer_name(layer, "mlp.up_exps_proj.weight"));
    let gt = m.raw(&layer_name(layer, "mlp.gate_exps_proj.weight"));
    let dn = m.raw(&layer_name(layer, "mlp.down_exps_proj.weight"));
    let tp = stripe_transpose();
    for e in 0..256 {
        for k in 0..4 {
            let src_off = (4 * e + k) * S;
            let mut us = vec![0u8; S];
            apply32(&mut us, &up[src_off..src_off + S], &tp);
            let mut gs = vec![0u8; S];
            apply32(&mut gs, &gt[src_off..src_off + S], &tp);
            put(&mut pool, (8 * e + 2 * k) * S, &us);
            put(&mut pool, (8 * e + 2 * k + 1) * S, &gs);
        }
    }

    let dp = down_perm();
    for e in 0..256 {
        let seg_src = &dn[e * 655_360..(e + 1) * 655_360];
        let mut seg = vec![0u8; 655_360];
        for (c, &f) in dp.iter().enumerate() {
            seg[c * CH..(c + 1) * CH].copy_from_slice(&seg_src[f * CH..(f + 1) * CH]);
        }
        put(&mut pool, 335_544_320 + e * 655_360, &seg);
    }

    let p128 = std_perm(128, 512, 2048);
    put(&mut pool, 503_316_480, &permute_chunks(m.raw(&layer_name(layer, "mlp.share_up_exps_proj.weight")), &p128, CH));
    put(&mut pool, 503_971_840, &permute_chunks(m.raw(&layer_name(layer, "mlp.share_gate_exps_proj.weight")), &p128, CH));
    put(
        &mut pool,
        504_627_200,
        &permute_chunks(m.raw(&layer_name(layer, "mlp.share_down_exps_proj.weight")), &std_perm(128, 2048, 512), CH),
    );

    if full_attn {
        // layout decoded from op-ctrlcode addresses (pool device base 0x20000):
        // [q-half 5242880][k 655360][v 655360][gate-half 5242880][o 5242880]
        let qg = m.raw(&layer_name(layer, "self_attn.q_proj.weight"));
        let p1024 = std_perm(1024, 4096, 2048);
        let q_half = permute_chunks(&qg[..1024 * CH], &p1024, CH);
        let gate_half = permute_chunks(&qg[1024 * CH..], &p1024, CH);
        put(&mut pool, 505_282_560, &q_half);
        put(&mut pool, 510_525_440, &permute_chunks(m.raw(&layer_name(layer, "self_attn.k_proj.weight")), &std_perm(128, 512, 2048), CH));
        put(&mut pool, 511_180_800, &permute_chunks(m.raw(&layer_name(layer, "self_attn.v_proj.weight")), &std_perm(128, 512, 2048), CH));
        put(&mut pool, 511_836_160, &gate_half);
        put(&mut pool, 517_079_040, &permute_chunks(m.raw(&layer_name(layer, "self_attn.o_proj.weight")), &std_perm(1024, 2048, 4096), CH));
    } else {
        put(&mut pool, 505_282_560, &permute_chunks(m.raw(&layer_name(layer, "linear_attn.qkv_proj.weight")), &std_perm(2048, 8192, 2048), CH));
        put(&mut pool, 515_768_320, &permute_chunks(m.raw(&layer_name(layer, "self_attn.gate_proj.weight")), &std_perm(1024, 4096, 2048), CH));
    }
    pool
}

/// lm_head q8 pool: 128-row supertile transpose (decoded via one-hot probes),
/// NOT the standard matmul perm. pool chunk `k` <- file chunk
/// `(4*(k/32) + (k%4))*8 + ((k%32)/4)`.
pub fn build_lmhead_pool(m: &Q4nx) -> Vec<u8> {
    const LM_CH: usize = 8704;
    let raw = m.raw("lm_head.weight");
    let nch = raw.len() / LM_CH;
    let mut out = vec![0u8; 542_113_792];
    for k in 0..nch {
        let (s, r) = (k / 32, k % 32);
        let (cg, rg) = (r / 4, r % 4);
        let f = (4 * s + rg) * 8 + cg;
        out[k * LM_CH..(k + 1) * LM_CH].copy_from_slice(&raw[f * LM_CH..(f + 1) * LM_CH]);
    }
    out
}

/// `[ln@0][postln@4096][sgate@8192][router@12288..1060863]` + zeros.
pub fn build_pack(m: &Q4nx, layer: usize) -> Vec<u8> {
    let mut pk = vec![0u8; 2_097_152];
    put(&mut pk, 0, m.raw(&layer_name(layer, "input_layernorm.weight")));
    put(&mut pk, 4096, m.raw(&layer_name(layer, "post_attention_layernorm.weight")));
    put(&mut pk, 8192, m.raw(&layer_name(layer, "shared_expert_gate.weight")));
    put(&mut pk, 12288, m.raw(&layer_name(layer, "moe_router.weight")));
    pk
}

pub fn build_side(m: &Q4nx, layer: usize, full_attn: bool) -> Vec<u8> {
    let mut side = vec![0u8; 6_291_456];
    if full_attn {
        put(&mut side, 128, m.raw(&layer_name(layer, "self_attn.q_norm.weight")));
        put(&mut side, 640, m.raw(&layer_name(layer, "self_attn.k_norm.weight")));
    } else {
        put(&mut side, 0, m.raw(&layer_name(layer, "linear_attn.ssm_conv1d.weight"))); // 65536
        put(&mut side, 65536, m.raw(&layer_name(layer, "linear_attn.ssm_norm.weight"))); // 256
        put(&mut side, 65792, m.raw(&layer_name(layer, "linear_attn.ssm_a"))); // 128 (f32)
        put(&mut side, 65920, m.raw(&layer_name(layer, "linear_attn.ssm_dt.bias"))); // 128
        put(&mut side, 66048, m.raw(&layer_name(layer, "linear_attn.ssm_alpha_proj.weight"))); // 131072
        put(&mut side, 197120, m.raw(&layer_name(layer, "linear_attn.ssm_beta_proj.weight"))); // 131072
        // out_proj: q4, needs pool permutation (std law, [2048,4096])
        let perm = permute_chunks(m.raw(&layer_name(layer, "linear_attn.ssm_out_proj.weight")), &std_perm(1024, 2048, 4096), CH);
        put(&mut side, 328_192, &perm);
    }
    side
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Byte-exact cross-check against a real captured NPU pool blob — mirrors
    /// build_pools.py's own `__main__` self-check.
    #[test]
    #[ignore] // requires the model file + capture blob to be present locally
    fn layer0_matches_captured_pool() {
        let model_dir = std::env::var("FLM_MODEL_DIR")
            .unwrap_or_else(|_| "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2".to_string());
        let cap_path = std::env::var("FLM_CAP_BLOB")
            .unwrap_or_else(|_| "C:/caps/m0d/blob_536870912_836fd8e49f35a0b6.bin".to_string());
        let m = Q4nx::open(&Path::new(&model_dir).join("model_3LiF.q4nx")).unwrap();
        let p0 = build_layer_pool(&m, 0, false);
        let cap = std::fs::read(cap_path).unwrap();
        assert_eq!(p0.len(), cap.len());
        let regions: [(&str, usize, usize); 8] = [
            ("gate_up", 0, 335_544_320),
            ("down", 335_544_320, 503_316_480),
            ("share_up", 503_316_480, 503_971_840),
            ("share_gate", 503_971_840, 504_627_200),
            ("share_down", 504_627_200, 505_282_560),
            ("qkv", 505_282_560, 515_768_320),
            ("z", 515_768_320, 521_011_200),
            ("tail", 521_011_200, 536_870_912),
        ];
        let mut all_ok = true;
        for (name, a, b) in regions {
            let eq = p0[a..b] == cap[a..b];
            let nd = p0[a..b].iter().zip(&cap[a..b]).filter(|(x, y)| x != y).count();
            println!("L0 rebuild {name:10}: {}", if eq { "MATCH".to_string() } else { format!("{nd} bytes differ") });
            all_ok &= eq;
        }
        assert!(all_ok, "pool rebuild does not match captured NPU pool");
    }

    fn open_3lif() -> Q4nx {
        let model_dir = std::env::var("FLM_MODEL_DIR")
            .unwrap_or_else(|_| "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2".to_string());
        Q4nx::open(&Path::new(&model_dir).join("model_3LiF.q4nx")).unwrap()
    }

    /// The m0d capture sequence for the 3LiF (L,L,F) schedule dumps buffers in
    /// program order: 000117=pool_L0 000118=pack_L0 000119=side_L0
    /// 000120=pool_L1 ... 000127=lm_head pool. 000117 is independently
    /// confirmed identical to the blob `layer0_matches_captured_pool` checks,
    /// which anchors this ordering.
    #[test]
    #[ignore] // requires local captures
    fn pack_and_side_l0_match_captured() {
        let m = open_3lif();
        let cap_dir = std::env::var("FLM_CAP_M0D").unwrap_or_else(|_| "C:/caps/m0d".to_string());
        let pack = build_pack(&m, 0);
        let pack_cap = std::fs::read(Path::new(&cap_dir).join("000118.bo")).unwrap();
        assert_eq!(pack, pack_cap, "pack_L0 mismatch");
        let side = build_side(&m, 0, false);
        let side_cap = std::fs::read(Path::new(&cap_dir).join("000119.bo")).unwrap();
        assert_eq!(side, side_cap, "side_L0 mismatch");
    }

    /// L0 above only exercises the linear-attention branch of
    /// `build_layer_pool`; L2 (full-attention, 3LiF's schedule is L,L,F)
    /// exercises the other branch (q/k/v/o layout instead of qkv/gate).
    #[test]
    #[ignore] // requires local captures
    fn full_attention_layer_pool_matches_captured() {
        let m = open_3lif();
        let cap_dir = std::env::var("FLM_CAP_M0D").unwrap_or_else(|_| "C:/caps/m0d".to_string());
        let p2 = build_layer_pool(&m, 2, true);
        let cap = std::fs::read(Path::new(&cap_dir).join("000123.bo")).unwrap();
        assert_eq!(p2.len(), cap.len());
        let regions: [(&str, usize, usize); 9] = [
            ("gate_up", 0, 335_544_320),
            ("down", 335_544_320, 503_316_480),
            ("share_up", 503_316_480, 503_971_840),
            ("share_gate", 503_971_840, 504_627_200),
            ("share_down", 504_627_200, 505_282_560),
            ("q_half", 505_282_560, 510_525_440),
            ("k_proj", 510_525_440, 511_180_800),
            ("v_proj", 511_180_800, 511_836_160),
            ("gate_half+o_proj", 511_836_160, 522_321_920),
        ];
        let mut all_ok = true;
        for (name, a, b) in regions {
            let eq = p2[a..b] == cap[a..b];
            println!("L2 (full-attn) rebuild {name:18}: {}", if eq { "MATCH" } else { "DIFFERS" });
            all_ok &= eq;
        }
        assert!(all_ok, "full-attention pool rebuild does not match captured NPU pool");
    }

    #[test]
    #[ignore] // requires local captures
    fn lmhead_pool_matches_captured() {
        let m = open_3lif();
        let cap_path = std::env::var("FLM_CAP_LMHEAD").unwrap_or_else(|_| "C:/caps/m0d/000127.bo".to_string());
        let lm = build_lmhead_pool(&m);
        let cap = std::fs::read(cap_path).unwrap();
        assert_eq!(lm.len(), cap.len());
        assert_eq!(lm, cap, "lm_head pool mismatch");
    }
}
