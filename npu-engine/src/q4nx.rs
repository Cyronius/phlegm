//! q4nx container access: safetensors header parse + verified dequantization.
//!
//! Byte-level formats verified against HF reference weights and FLM's live
//! NPU buffers (see .claude/plans/npu-open-engine.md "Step 3 STATUS"):
//!
//! q4 tensor (I8, [A, B, 5120]): chunks of 5120 B, each covering 32 rows x
//! 256 cols of the logical [out, in] matrix in plain raster order
//! (chunk f -> rows 32*(f/ncol).., cols 256*(f%ncol).., ncol = in/256).
//! Chunk = 256 bf16 scales d + 256 bf16 mins m (planar, q4_1 block-32 along
//! in-dim, meta index j = bc*32 + r), then 4096 B of nibbles in a 16-lane
//! interleave: nibble[(r/16)*4096 + bc*512 + i*16 + (r%16)] = elem(r, bc*32+i)
//! (byte b: even nibble index = low nibble). value = n*d[j] + m[j].
//!
//! q8 tensor (lm_head, [A, B, 8704]): 512 B bf16 scales (j = bc*32 + r,
//! block 32) + 8192 int8 bytes with the SAME 16-lane index formula (bytes
//! instead of nibbles). value = q*d[j]. Raster order, in = 2048.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub struct Q4nx {
    mmap: Mmap,
    data_base: usize,
    pub tensors: HashMap<String, TensorMeta>,
    /// "q4_1" (FLM 1.0.2) or "q4k" (FLM 1.0.3).
    pub fmt: String,
    /// quantized chunk size in bytes: 5120 (q4_1) or 4736 (q4k).
    pub chunk_bytes: usize,
}

// FLM 1.0.3 (q4k) reorder-undo constants (mirror of q4nx_v103.py).
const STATE_SIZE: usize = 128; // ssm.state_size (INFERRED); reorder p-dim
const N_HEADS: usize = 16; // g: KV heads across the paired halves

#[inline]
pub fn bf16_to_f32(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}

impl Q4nx {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let n = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n]).unwrap();
        let mut tensors = HashMap::new();
        for (k, v) in hdr.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let off = v["data_offsets"].as_array().unwrap();
            tensors.insert(
                k.clone(),
                TensorMeta {
                    dtype: v["dtype"].as_str().unwrap().to_string(),
                    shape: v["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect(),
                    start: off[0].as_u64().unwrap() as usize,
                    end: off[1].as_u64().unwrap() as usize,
                },
            );
        }
        // Detect format from a non-lm_head I8 tensor's chunk size (no version
        // field exists in the header): 4736 -> q4k (1.0.3), else 5120 -> q4_1.
        let mut fmt = "q4_1".to_string();
        let mut chunk_bytes = 5120usize;
        for (k, t) in &tensors {
            if t.dtype == "I8" && k != "lm_head.weight" {
                let cb = *t.shape.last().unwrap();
                if cb == 4736 {
                    fmt = "q4k".to_string();
                    chunk_bytes = 4736;
                } else {
                    fmt = "q4_1".to_string();
                    chunk_bytes = 5120;
                }
                break;
            }
        }
        Ok(Q4nx { mmap, data_base: 8 + n, tensors, fmt, chunk_bytes })
    }

    /// 'linear' | 'full' | None for the layer a tensor belongs to (mirror of
    /// q4nx.py `_layer_type`): by presence of qkv_proj vs self_attn.q_proj.
    fn layer_type(&self, name: &str) -> Option<&'static str> {
        if !name.contains(".layer.") {
            return None;
        }
        let l = name.split(".layer.").nth(1)?.split('.').next()?;
        if self.tensors.contains_key(&format!("model.layer.{l}.linear_attn.qkv_proj.weight")) {
            return Some("linear");
        }
        if self.tensors.contains_key(&format!("model.layer.{l}.self_attn.q_proj.weight")) {
            return Some("full");
        }
        None
    }

    /// (out, in) dims of a tensor: shape[0] and product of the rest (1 if 1-D).
    fn shape_oi(&self, name: &str) -> (usize, usize) {
        let s = &self.tensors[name].shape;
        let o = s[0];
        let i: usize = if s.len() > 1 { s[1..].iter().product() } else { 1 };
        (o, i)
    }

    /// Undo the FLM 1.0.3 head-pairing reorders baked into linear-attention
    /// tensors (mirror of q4nx_v103.apply_undo). No-op unless fmt == "q4k".
    /// `w` is the dequantized [out, in] matrix (row-major) in the stored order.
    fn apply_undo(&self, name: &str, w: &mut [f32], out_dim: usize, in_dim: usize) {
        if self.fmt != "q4k" {
            return;
        }
        if self.layer_type(name) != Some("linear") {
            return;
        }
        let p = STATE_SIZE;
        if name.contains("linear_attn.qkv_proj") {
            // [8192,2048]: v-half (rows 4096:8192) reordered (g q p)->(q g p)
            undo_qgp_rows(w, 4096, in_dim, p);
        } else if name.contains("self_attn.gate_proj") {
            // z-gate [4096,2048]: all rows (g q p)->(q g p)
            undo_qgp_rows(w, 0, in_dim, p);
        } else if name.contains("linear_attn.ssm_out_proj") {
            // [2048,4096]: columns (g q p)->(q g p)
            undo_qgp_cols(w, out_dim, in_dim, 0, p);
        } else if name.contains("ssm_alpha_proj") || name.contains("ssm_beta_proj") {
            // bf16 [2048,32] (transposed): the 32 out-dim columns (g q)->(q g)
            undo_qg_cols(w, out_dim, in_dim);
        } else if name.contains("ssm_conv1d") {
            // bf16 [4,8192]: v-half columns (4096:8192) (g q p)->(q g p)
            undo_qgp_cols(w, out_dim, in_dim, 4096, p);
        } else if name.ends_with("linear_attn.ssm_a") {
            // f32 [32]: (g q)->(q g)
            undo_qg_vec(w);
        } else if name.contains("linear_attn.ssm_dt.bias") {
            // f32 [32]: (g q)->(q g)
            undo_qg_vec(w);
        }
    }

    pub fn raw(&self, name: &str) -> &[u8] {
        let t = self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        &self.mmap[self.data_base + t.start..self.data_base + t.end]
    }

    /// BF16 tensor -> f32 vec (row-major). For q4k, applies the 1.0.3
    /// reorder-undo (ssm_conv1d, ssm_alpha_proj, ssm_beta_proj).
    pub fn bf16(&self, name: &str) -> Vec<f32> {
        let raw = self.raw(name);
        assert_eq!(self.tensors[name].dtype, "BF16");
        let mut w: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (o, i) = self.shape_oi(name);
        self.apply_undo(name, &mut w, o, i);
        w
    }

    /// F32 tensor -> f32 vec. For q4k, applies the 1.0.3 reorder-undo
    /// (ssm_a, ssm_dt.bias).
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let raw = self.raw(name);
        assert_eq!(self.tensors[name].dtype, "F32");
        let mut w: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (o, i) = self.shape_oi(name);
        self.apply_undo(name, &mut w, o, i);
        w
    }

    /// One embedding row [2048] as f32 (embed_tokens is plain BF16).
    pub fn embed_row(&self, name: &str, row: usize, dim: usize) -> Vec<f32> {
        let raw = self.raw(name);
        let start = row * dim * 2;
        raw[start..start + dim * 2]
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    /// Dequantize a q4 matmul tensor -> logical f32 [out_dim * in_dim].
    /// Format-aware (q4_1 or q4k) and applies the 1.0.3 reorder-undo, so this
    /// is the `matmul_w` equivalent forward.rs consumes.
    pub fn dequant_q4(&self, name: &str, out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut w = if self.fmt == "q4k" {
            dequant_q4k_bytes(self.raw(name), out_dim, in_dim)
        } else {
            dequant_q4_bytes(self.raw(name), out_dim, in_dim)
        };
        self.apply_undo(name, &mut w, out_dim, in_dim);
        w
    }

    /// Dequantize raw quantized chunk bytes -> [out, in] fp, PLAIN RASTER, NO
    /// reorder. Used for expert slices (experts carry no head-pairing reorder).
    pub fn dequant_q4_tile(&self, bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
        if self.fmt == "q4k" {
            dequant_q4k_bytes(bytes, out_dim, in_dim)
        } else {
            dequant_q4_bytes(bytes, out_dim, in_dim)
        }
    }

    /// Format-aware q8 lm_head dequant of rows [row0, row0+nrows).
    pub fn dequant_q8_rows(&self, bytes: &[u8], row0: usize, nrows: usize) -> Vec<f32> {
        if self.fmt == "q4k" {
            dequant_q8_rows_q4k(bytes, row0, nrows)
        } else {
            dequant_q8_rows(bytes, row0, nrows)
        }
    }
}

// ---- FLM 1.0.3 reorder-undo helpers (mirror of q4nx_v103.py) -----------------
// All invert the head-pairing perm `(g q p)->(q g p)` (g=16, q=2, p=state).

/// rows laid as (g q p) -> (q g p) over the sub-block of G*2*p rows starting at
/// `row_off`; each row is `in_dim` wide.
fn undo_qgp_rows(w: &mut [f32], row_off: usize, in_dim: usize, p: usize) {
    let nrows = N_HEADS * 2 * p;
    let mut tmp = vec![0f32; nrows * in_dim];
    for q in 0..2 {
        for g in 0..N_HEADS {
            for pp in 0..p {
                let new_row = q * (N_HEADS * p) + g * p + pp;
                let old_row = g * (2 * p) + q * p + pp;
                let src = &w[(row_off + old_row) * in_dim..(row_off + old_row + 1) * in_dim];
                tmp[new_row * in_dim..(new_row + 1) * in_dim].copy_from_slice(src);
            }
        }
    }
    w[row_off * in_dim..(row_off + nrows) * in_dim].copy_from_slice(&tmp);
}

/// cols laid as (g q p) -> (q g p) over the column range [col_off, col_off+G*2*p)
/// of every row.
fn undo_qgp_cols(w: &mut [f32], out_dim: usize, in_dim: usize, col_off: usize, p: usize) {
    let ncols = N_HEADS * 2 * p;
    let mut tmp = vec![0f32; ncols];
    for row in 0..out_dim {
        let base = row * in_dim + col_off;
        for q in 0..2 {
            for g in 0..N_HEADS {
                for pp in 0..p {
                    let new_col = q * (N_HEADS * p) + g * p + pp;
                    let old_col = g * (2 * p) + q * p + pp;
                    tmp[new_col] = w[base + old_col];
                }
            }
        }
        w[base..base + ncols].copy_from_slice(&tmp);
    }
}

/// cols laid as (g q) -> (q g), 32 cols per row (in_dim == 32).
fn undo_qg_cols(w: &mut [f32], out_dim: usize, in_dim: usize) {
    let mut tmp = vec![0f32; in_dim];
    for row in 0..out_dim {
        let base = row * in_dim;
        for q in 0..2 {
            for g in 0..N_HEADS {
                tmp[q * N_HEADS + g] = w[base + g * 2 + q];
            }
        }
        w[base..base + in_dim].copy_from_slice(&tmp);
    }
}

/// 1-D vector (g q) -> (q g), length 32.
fn undo_qg_vec(v: &mut [f32]) {
    let mut tmp = vec![0f32; v.len()];
    for q in 0..2 {
        for g in 0..N_HEADS {
            tmp[q * N_HEADS + g] = v[g * 2 + q];
        }
    }
    v.copy_from_slice(&tmp);
}

/// Dequantize verbatim FLM 1.0.3 Q4_K chunk bytes (raster order) into
/// [out_dim, in_dim]. One 4736 B chunk = one 32-row x 256-col tile, concat
/// order [s8|m8|q|S|M]; value(r,c) = S[r]*s8[g,r]*q(r,c) + M[r]*m8[g,r],
/// g=c//32, M stored negated. (Mirror of q4nx_v103.dequant_q4k_chunks.)
pub fn dequant_q4k_bytes(bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let nch = bytes.len() / 4736;
    assert_eq!(nch * 8192, out_dim * in_dim, "q4k size mismatch");
    let ncol = in_dim / 256;
    let mut w = vec![0f32; out_dim * in_dim];
    for f in 0..nch {
        let chunk = &bytes[f * 4736..(f + 1) * 4736];
        let rows0 = 32 * (f / ncol);
        let cols0 = 256 * (f % ncol);
        let s8 = &chunk[0..256];
        let m8 = &chunk[256..512];
        let q = &chunk[512..4608];
        let sb = &chunk[4608..4672];
        let mb = &chunk[4672..4736];
        for r in 0..32 {
            let s_super = bf16_to_f32(u16::from_le_bytes([sb[2 * r], sb[2 * r + 1]]));
            let m_super = bf16_to_f32(u16::from_le_bytes([mb[2 * r], mb[2 * r + 1]]));
            let out = &mut w[(rows0 + r) * in_dim + cols0..];
            for c in 0..256 {
                let g = c / 32;
                let sm = g * 32 + r;
                let byte = q[c * 16 + r / 2];
                let nib = if r % 2 == 1 { byte >> 4 } else { byte & 0x0F };
                out[c] = s_super * s8[sm] as f32 * nib as f32 + m_super * m8[sm] as f32;
            }
        }
    }
    w
}

/// Dequantize FLM 1.0.3 column-major Q8_0 lm_head rows [row0, row0+nrows) ->
/// f32 [nrows * 2048]. chunk 8704 B = [scales 512B (256 bf16, g*32+r) | data
/// 8192 int8, c*32+r]; value(r,c) = scale[g,r]*q, g=c//32. (in_dim fixed 2048,
/// ncol = 8.) Mirror of q4nx_v103.dequant_q8_q4k_file.
pub fn dequant_q8_rows_q4k(bytes: &[u8], row0: usize, nrows: usize) -> Vec<f32> {
    assert_eq!(row0 % 32, 0);
    let mut w = vec![0f32; nrows * 2048];
    for rr in 0..nrows {
        let row = row0 + rr;
        let gt = row / 32;
        let r = row % 32;
        for bc8 in 0..8 {
            let chunk_idx = gt * 8 + bc8;
            let chunk = &bytes[chunk_idx * 8704..(chunk_idx + 1) * 8704];
            let scales = &chunk[..512];
            let data = &chunk[512..];
            let out = &mut w[rr * 2048 + bc8 * 256..];
            for c in 0..256 {
                let g = c / 32;
                let sidx = g * 32 + r;
                let d = bf16_to_f32(u16::from_le_bytes([scales[2 * sidx], scales[2 * sidx + 1]]));
                out[c] = data[c * 32 + r] as i8 as f32 * d;
            }
        }
    }
    w
}

/// Dequantize verbatim q4 chunk bytes (raster order) into [out_dim, in_dim].
pub fn dequant_q4_bytes(bytes: &[u8], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let nch = bytes.len() / 5120;
    assert_eq!(nch * 8192, out_dim * in_dim, "q4 size mismatch");
    let ncol = in_dim / 256;
    let mut w = vec![0f32; out_dim * in_dim];
    for f in 0..nch {
        let chunk = &bytes[f * 5120..(f + 1) * 5120];
        let rows0 = 32 * (f / ncol);
        let cols0 = 256 * (f % ncol);
        let meta = &chunk[..1024];
        let q = &chunk[1024..];
        for r in 0..32 {
            for bc in 0..8 {
                let j = bc * 32 + r;
                let d = bf16_to_f32(u16::from_le_bytes([meta[2 * j], meta[2 * j + 1]]));
                let mj = 512 + 2 * j;
                let m = bf16_to_f32(u16::from_le_bytes([meta[mj], meta[mj + 1]]));
                let row = &mut w[(rows0 + r) * in_dim + cols0 + bc * 32..];
                for i in 0..32 {
                    let p = (r / 16) * 4096 + bc * 512 + i * 16 + (r % 16);
                    let byte = q[p / 2];
                    let n = if p % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    row[i] = n as f32 * d + m;
                }
            }
        }
    }
    w
}

/// Dequantize q8 lm_head rows [row0, row0+nrows) -> f32 [nrows * 2048].
/// (in_dim fixed 2048, ncol = 8.)
pub fn dequant_q8_rows(bytes: &[u8], row0: usize, nrows: usize) -> Vec<f32> {
    assert_eq!(row0 % 32, 0);
    let mut w = vec![0f32; nrows * 2048];
    for rr in 0..nrows {
        let row = row0 + rr;
        let g = row / 32;
        let r = row % 32;
        for bc8 in 0..8 {
            // colgroup cg: chunk g*8+cg covers cols 256*cg
            let c = g * 8 + bc8;
            let chunk = &bytes[c * 8704..(c + 1) * 8704];
            let meta = &chunk[..512];
            let q = &chunk[512..];
            for bc in 0..8 {
                let j = bc * 32 + r;
                let d = bf16_to_f32(u16::from_le_bytes([meta[2 * j], meta[2 * j + 1]]));
                let out = &mut w[rr * 2048 + bc8 * 256 + bc * 32..];
                for i in 0..32 {
                    let p = (r / 16) * 4096 + bc * 512 + i * 16 + (r % 16);
                    out[i] = q[p] as i8 as f32 * d;
                }
            }
        }
    }
    w
}
