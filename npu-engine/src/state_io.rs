//! Serialize CPU-computed prefill state into the NPU driver's binary buffer
//! formats, and build the per-token `act` buffer — ported from
//! `tools/kernel-interp/l30_build.py`'s `serialize_linear_state` /
//! `serialize_kv_state` and `generate_npu.py`'s `write_act`.
//!
//! These are the exact inverse of `main.rs`'s `load_linear_state` /
//! `load_kv_state` (which read captured states back for validation against
//! real NPU captures) — see the round-trip tests below, which lean on that
//! existing, already-verified read path instead of a fresh capture.

use crate::forward::{KvState, LinearState};
use crate::q4nx::{bf16_to_f32, f32_to_bf16};

const STATE_BYTES: usize = 3_145_728; // 3 MB kernel state buffer
const ACT_BYTES: usize = 1_048_576; // 1 MB activation buffer

/// Parse a captured/generated 3MB linear-state buffer (conv bf16[3,8192] +
/// S fp32[32,128,128]). Canonical counterpart to [`serialize_linear_state`].
pub fn load_linear_state_bytes(raw: &[u8]) -> LinearState {
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

/// Parse a captured/generated KV pack (k bf16[T,512] @0, v @byte 1073152).
/// Canonical counterpart to [`serialize_kv_state`].
pub fn load_kv_state_bytes(raw: &[u8], t: usize) -> KvState {
    let rd = |off: usize, n: usize| -> Vec<f64> {
        raw[off..off + n * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect()
    };
    KvState { k: rd(0, t * 512), v: rd(1_073_152, t * 512) }
}

fn write_bf16(dst: &mut [u8], vals: &[f32]) {
    for (i, &v) in vals.iter().enumerate() {
        dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
}

/// linear-attention decode state: `[conv bf16[3,8192] @0 | S f32[32,128,128] @49152]`.
pub fn serialize_linear_state(st: &LinearState) -> Vec<u8> {
    let mut buf = vec![0u8; STATE_BYTES];
    write_bf16(&mut buf[..49152], &st.conv); // 3*8192 = 24576 values * 2B
    let s32: Vec<f32> = st.s.iter().map(|v| *v as f32).collect();
    for (i, v) in s32.iter().enumerate() {
        buf[49152 + i * 4..49152 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    buf
}

/// full-attention decode state (KV cache): `[k bf16[T,512] @0 | v bf16[T,512] @1073152]`.
pub fn serialize_kv_state(st: &KvState) -> Vec<u8> {
    let mut buf = vec![0u8; STATE_BYTES];
    let k32: Vec<f32> = st.k.iter().map(|v| *v as f32).collect();
    let v32: Vec<f32> = st.v.iter().map(|v| *v as f32).collect();
    write_bf16(&mut buf[..k32.len() * 2], &k32);
    write_bf16(&mut buf[1_073_152..1_073_152 + v32.len() * 2], &v32);
    buf
}

/// One decode-step activation buffer: `[embed(tok) bf16[2048] @0 | model.norm.weight bf16[2048] @4096]`.
pub fn write_act(embed: &[f64], norm_weight: &[f32]) -> Vec<u8> {
    let mut act = vec![0u8; ACT_BYTES];
    let embed32: Vec<f32> = embed.iter().map(|v| *v as f32).collect();
    write_bf16(&mut act[..4096], &embed32);
    write_bf16(&mut act[4096..8192], norm_weight);
    act
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_state_round_trips_through_the_verified_loader() {
        let conv: Vec<f32> = (0..3 * 8192).map(|i| (i as f32 * 0.001).sin()).collect();
        let s: Vec<f64> = (0..32 * 128 * 128).map(|i| (i as f64 * 0.0001).cos()).collect();
        let st = LinearState { conv: conv.clone(), s: s.clone() };
        let buf = serialize_linear_state(&st);
        let back = load_linear_state_bytes(&buf);
        for (a, b) in conv.iter().zip(&back.conv) {
            assert!((a - b).abs() < 1e-2, "conv round-trip mismatch (bf16 precision): {a} vs {b}");
        }
        for (a, b) in s.iter().zip(&back.s) {
            assert!((a - b).abs() < 1e-5, "S round-trip mismatch (f32 precision): {a} vs {b}");
        }
    }

    #[test]
    fn kv_state_round_trips_through_the_verified_loader() {
        let t = 11;
        let k: Vec<f64> = (0..t * 512).map(|i| (i as f64 * 0.001).sin()).collect();
        let v: Vec<f64> = (0..t * 512).map(|i| (i as f64 * 0.0002).cos()).collect();
        let st = KvState { k: k.clone(), v: v.clone() };
        let buf = serialize_kv_state(&st);
        let back = load_kv_state_bytes(&buf, t);
        for (a, b) in k.iter().zip(&back.k) {
            assert!((a - b).abs() < 1e-2, "k round-trip mismatch (bf16 precision): {a} vs {b}");
        }
        for (a, b) in v.iter().zip(&back.v) {
            assert!((a - b).abs() < 1e-2, "v round-trip mismatch (bf16 precision): {a} vs {b}");
        }
    }
}
