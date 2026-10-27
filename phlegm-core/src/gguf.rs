//! Bounded GGUF tensor reader and lossless Q4 expert pool packing.
//!
//! GGUF stays the model file. These routines stream its blocks into the
//! f32-scale chunk layout consumed by the NPU kernels; they create no Q4NX
//! intermediate. The C++ reference remains the inference engine while the
//! corresponding Rust runtime is brought up.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const POOL_CHUNK: usize = 6144;
const VALUES_PER_BLOCK: u64 = 32;
const MAX_STRING: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    Q4_0,
    Q4_1,
}

impl Quant {
    fn from_type(ty: u32) -> Result<Self, String> {
        match ty {
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            _ => Err(format!("GGUF tensor type {ty} cannot use the exact Q4 expert packer")),
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// GGUF dimensions are fastest first: [input, output, expert] for MoE.
    pub dims: Vec<u64>,
    pub gguf_type: u32,
    pub offset: u64,
    pub byte_len: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Metadata {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

pub struct Gguf<R: Read + Seek> {
    reader: R,
    data_start: u64,
    metadata: HashMap<String, Metadata>,
    tensors: HashMap<String, TensorInfo>,
}

/// Check a direct-GGUF bundle before taking the NPU lock or hashing large weights.
/// Legacy model directories without model.gguf continue through the reference
/// bridge until the Rust inference path replaces them.
pub fn preflight_direct_route(model_dir: &Path, kernel_dir: &Path) -> Result<(), String> {
    let path = model_dir.join("model.gguf");
    if !path.is_file() { return Ok(()); }
    if model_dir.join("model.q4nx").is_file() {
        return Err(format!("{} contains both model.gguf and model.q4nx; keep one model file", model_dir.display()));
    }
    let gguf = Gguf::open(&path)?;
    let architecture = match gguf.metadata("general.architecture") {
        Some(Metadata::String(s)) => s.as_str(),
        _ => return Err(format!("{} lacks GGUF general.architecture", path.display())),
    };
    let manifest_path = kernel_dir.join("manifest.json");
    let manifest_file = File::open(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_reader(manifest_file)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    if !manifest.get("gguf").is_some_and(serde_json::Value::is_object) {
        return Err(format!("kernel set {} has no direct GGUF route for {architecture}", kernel_dir.display()));
    }
    Ok(())
}

impl Gguf<File> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path.as_ref()).map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
        Self::new(file)
    }
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(mut reader: R) -> Result<Self, String> {
        let file_len = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        let mut magic = [0; 4];
        read_exact(&mut reader, &mut magic)?;
        if &magic != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = read_u32(&mut reader)?;
        if !(2..=3).contains(&version) {
            return Err(format!("GGUF version {version} is unsupported (need v2 or v3)"));
        }
        let tensor_count = read_u64(&mut reader)?;
        let metadata_count = read_u64(&mut reader)?;
        // Each entry needs a nonempty name and fixed fields. Bound counts before
        // allocating a map or iterating over an implausible corrupt header.
        if tensor_count > file_len / 32 || metadata_count > file_len / 13 {
            return Err("GGUF header count exceeds file size".into());
        }
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut reader, file_len)?;
            let ty = read_u32(&mut reader)?;
            let value = read_metadata(&mut reader, file_len, ty, 0)?;
            if let Some(value) = value {
                if metadata.insert(key.clone(), value).is_some() {
                    return Err(format!("duplicate GGUF metadata key {key}"));
                }
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut reader, file_len)?;
            let nd = read_u32(&mut reader)?;
            if nd == 0 || nd > 8 {
                return Err(format!("GGUF tensor {name} has invalid dimension count {nd}"));
            }
            let mut dims = Vec::with_capacity(nd as usize);
            let mut values = 1u64;
            for _ in 0..nd {
                let d = read_u64(&mut reader)?;
                if d == 0 {
                    return Err(format!("GGUF tensor {name} has a zero dimension"));
                }
                values = values.checked_mul(d).ok_or("GGUF tensor dimension product overflow")?;
                dims.push(d);
            }
            let ty = read_u32(&mut reader)?;
            let offset = read_u64(&mut reader)?;
            let byte_len = tensor_bytes(ty, values)?;
            if tensors.insert(name.clone(), TensorInfo { dims, gguf_type: ty, offset, byte_len }).is_some() {
                return Err(format!("duplicate GGUF tensor {name}"));
            }
        }
        let header_end = reader.stream_position().map_err(|e| e.to_string())?;
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(Metadata::Unsigned(v)) => *v,
            _ => return Err("general.alignment must be an unsigned integer".into()),
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("general.alignment must be a nonzero power of two".into());
        }
        let data_start = header_end.checked_add(alignment - 1)
            .map(|v| v & !(alignment - 1)).ok_or("GGUF data offset overflow")?;
        if data_start > file_len {
            return Err("GGUF data starts past EOF".into());
        }
        for (name, info) in &tensors {
            let start = data_start.checked_add(info.offset).ok_or("GGUF tensor offset overflow")?;
            if start > file_len || info.byte_len.is_some_and(|n| start.checked_add(n).is_none_or(|end| end > file_len)) {
                return Err(format!("GGUF tensor {name} runs past EOF"));
            }
            if info.offset % alignment != 0 {
                return Err(format!("GGUF tensor {name} offset is not {alignment}-byte aligned"));
            }
        }
        Ok(Self { reader, data_start, metadata, tensors })
    }

    pub fn metadata(&self, key: &str) -> Option<&Metadata> {
        self.metadata.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }

    fn read_block(&mut self, name: &str, block: u64, dst: &mut [u8]) -> Result<(), String> {
        let info = self.tensors.get(name).ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        let n = info.byte_len.ok_or_else(|| format!("GGUF tensor {name} has unsupported type {}", info.gguf_type))?;
        let relative = block.checked_mul(dst.len() as u64).ok_or("GGUF block offset overflow")?;
        if relative.checked_add(dst.len() as u64).is_none_or(|end| end > n) {
            return Err(format!("GGUF tensor {name} block {block} runs past tensor"));
        }
        let at = self.data_start.checked_add(info.offset).and_then(|v| v.checked_add(relative))
            .ok_or("GGUF file offset overflow")?;
        self.reader.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        read_exact(&mut self.reader, dst)
    }

    /// Pack a Q4_0/Q4_1 [K, FF, E] pair in expert/stripe/up-then-gate order.
    /// Each stripe holds 128 output rows; one chunk covers 32 rows x 256 K.
    pub fn pack_expert_stripes<W: Write>(&mut self, up: &str, gate: &str, out: &mut W) -> Result<u64, String> {
        let (ue, rows, cols, uq) = self.expert_shape(up)?;
        let (ge, gr, gc, gq) = self.expert_shape(gate)?;
        if (ue, rows, cols, uq) != (ge, gr, gc, gq) {
            return Err("GGUF up/gate expert tensor shapes or types differ".into());
        }
        let mut written = 0u64;
        for e in 0..ue {
            for stripe in 0..rows / 128 {
                for name in [up, gate] {
                    for k in 0..cols / 256 {
                        for rb in 0..4 {
                            let chunk = self.pack_chunk(name, e, rows, cols, stripe * 128 + rb * 32, k * 256, uq)?;
                            out.write_all(&chunk).map_err(|e| e.to_string())?;
                            written += POOL_CHUNK as u64;
                        }
                    }
                }
            }
        }
        Ok(written)
    }

    /// Pack a Q4_0/Q4_1 [FF, hidden, E] down tensor, one expert at a time.
    pub fn pack_expert_down<W: Write>(&mut self, name: &str, out: &mut W) -> Result<u64, String> {
        let (experts, rows, cols, quant) = self.expert_shape(name)?;
        let mut written = 0u64;
        for e in 0..experts {
            for band in 0..rows / 128 {
                for k in 0..cols / 256 {
                    for rb in 0..4 {
                        let chunk = self.pack_chunk(name, e, rows, cols, band * 128 + rb * 32, k * 256, quant)?;
                        out.write_all(&chunk).map_err(|e| e.to_string())?;
                        written += POOL_CHUNK as u64;
                    }
                }
            }
        }
        Ok(written)
    }

    fn expert_shape(&self, name: &str) -> Result<(u64, u64, u64, Quant), String> {
        let info = self.tensors.get(name).ok_or_else(|| format!("missing GGUF tensor {name}"))?;
        if info.dims.len() != 3 || info.dims[0] % 256 != 0 || info.dims[1] % 128 != 0 {
            return Err(format!("GGUF tensor {name} needs [K, rows, experts] with 256-column and 128-row tiles"));
        }
        let quant = Quant::from_type(info.gguf_type)?;
        Ok((info.dims[2], info.dims[1], info.dims[0], quant))
    }

    fn pack_chunk(&mut self, name: &str, expert: u64, rows: u64, cols: u64,
                  row0: u64, col0: u64, quant: Quant) -> Result<[u8; POOL_CHUNK], String> {
        let mut out = [0u8; POOL_CHUNK];
        let block_bytes = quant.block_bytes();
        let blocks_per_row = cols / VALUES_PER_BLOCK;
        let mut block = [0u8; 20];
        for kb in 0..8u64 {
            for r in 0..32u64 {
                let source_block = (expert * rows + row0 + r) * blocks_per_row + (col0 / 32 + kb);
                self.read_block(name, source_block, &mut block[..block_bytes])?;
                let d = f32_from_f16(u16::from_le_bytes([block[0], block[1]]));
                let m = if quant == Quant::Q4_1 {
                    f32_from_f16(u16::from_le_bytes([block[2], block[3]]))
                } else { -8.0 * d };
                let plane = 4 * (kb as usize * 32 + r as usize);
                out[plane..plane + 4].copy_from_slice(&d.to_le_bytes());
                out[1024 + plane..1024 + plane + 4].copy_from_slice(&m.to_le_bytes());
                let codes = &block[if quant == Quant::Q4_1 { 4..20 } else { 2..18 }];
                let row_half = r as usize / 16;
                let row_lane = r as usize % 16;
                for i in 0..16usize {
                    let byte = codes[i];
                    let p0 = row_half * 4096 + (kb as usize * 32 + i) * 16 + row_lane;
                    set_nibble(&mut out[2048..], p0, byte & 0x0f);
                    set_nibble(&mut out[2048..], p0 + 256, byte >> 4);
                }
            }
        }
        Ok(out)
    }
}

fn set_nibble(codes: &mut [u8], at: usize, q: u8) {
    let b = &mut codes[at / 2];
    if at & 1 == 0 { *b |= q; } else { *b |= q << 4; }
}

fn f32_from_f16(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            let shift = mant.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((mant << shift & 0x03ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

fn tensor_bytes(ty: u32, values: u64) -> Result<Option<u64>, String> {
    let (block, bytes) = match ty {
        0 => (1, 4), 1 | 30 => (1, 2), 2 => (32, 18), 3 => (32, 20),
        8 => (32, 34), 12 => (256, 144), 14 => (256, 210),
        _ => return Ok(None),
    };
    if values % block != 0 { return Err(format!("GGUF tensor has {values} values, not whole type-{ty} blocks")); }
    Ok(Some((values / block).checked_mul(bytes).ok_or("GGUF tensor byte size overflow")?))
}

fn read_exact<R: Read>(reader: &mut R, dst: &mut [u8]) -> Result<(), String> {
    reader.read_exact(dst).map_err(|e| format!("truncated GGUF header or tensor: {e}"))
}
fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> { let mut b = [0]; read_exact(r, &mut b)?; Ok(b[0]) }
fn read_u16<R: Read>(r: &mut R) -> Result<u16, String> { let mut b = [0; 2]; read_exact(r, &mut b)?; Ok(u16::from_le_bytes(b)) }
fn read_u32<R: Read>(r: &mut R) -> Result<u32, String> { let mut b = [0; 4]; read_exact(r, &mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64<R: Read>(r: &mut R) -> Result<u64, String> { let mut b = [0; 8]; read_exact(r, &mut b)?; Ok(u64::from_le_bytes(b)) }

fn read_string<R: Read + Seek>(r: &mut R, len: u64) -> Result<String, String> {
    let n = read_u64(r)?;
    if n > MAX_STRING || n > len.saturating_sub(r.stream_position().map_err(|e| e.to_string())?) {
        return Err("GGUF string length is invalid".into());
    }
    let mut bytes = vec![0; n as usize];
    read_exact(r, &mut bytes)?;
    String::from_utf8(bytes).map_err(|e| format!("GGUF string is not UTF-8: {e}"))
}

fn skip_bytes<R: Seek>(r: &mut R, len: u64, n: u64) -> Result<(), String> {
    let at = r.stream_position().map_err(|e| e.to_string())?;
    let end = at.checked_add(n).ok_or("GGUF metadata offset overflow")?;
    if end > len { return Err("GGUF metadata runs past EOF".into()); }
    r.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
    Ok(())
}

fn read_metadata<R: Read + Seek>(r: &mut R, len: u64, ty: u32, depth: u8) -> Result<Option<Metadata>, String> {
    if depth > 4 { return Err("GGUF metadata array nesting is too deep".into()); }
    let value = match ty {
        0 => Some(Metadata::Unsigned(read_u8(r)? as u64)),
        1 => Some(Metadata::Signed(read_u8(r)? as i8 as i64)),
        2 => Some(Metadata::Unsigned(read_u16(r)? as u64)),
        3 => Some(Metadata::Signed(read_u16(r)? as i16 as i64)),
        4 => Some(Metadata::Unsigned(read_u32(r)? as u64)),
        5 => Some(Metadata::Signed(read_u32(r)? as i32 as i64)),
        6 => Some(Metadata::Float(f32::from_bits(read_u32(r)?) as f64)),
        7 => Some(Metadata::Bool(read_u8(r)? != 0)),
        8 => Some(Metadata::String(read_string(r, len)?)),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            if let Some(size) = match elem_type { 0 | 1 | 7 => Some(1), 2 | 3 => Some(2),
                4 | 5 | 6 => Some(4), 10 | 11 | 12 => Some(8), _ => None } {
                skip_bytes(r, len, count.checked_mul(size).ok_or("GGUF array size overflow")?)?;
            } else {
                if count > len / 8 { return Err("GGUF metadata array count exceeds file size".into()); }
                for _ in 0..count { read_metadata(r, len, elem_type, depth + 1)?; }
            }
            None
        }
        10 => Some(Metadata::Unsigned(read_u64(r)?)),
        11 => Some(Metadata::Signed(read_u64(r)? as i64)),
        12 => Some(Metadata::Float(f64::from_bits(read_u64(r)?))),
        _ => return Err(format!("unknown GGUF metadata type {ty}")),
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

    fn u32le(out: &mut Vec<u8>, n: u32) { out.extend_from_slice(&n.to_le_bytes()); }
    fn u64le(out: &mut Vec<u8>, n: u64) { out.extend_from_slice(&n.to_le_bytes()); }
    fn string(out: &mut Vec<u8>, s: &str) { u64le(out, s.len() as u64); out.extend_from_slice(s.as_bytes()); }
    fn align(out: &mut Vec<u8>) { while out.len() % 32 != 0 { out.push(0); } }

    fn source_block(quant: Quant, role: u8, expert: u8, row: u16, kb: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(quant.block_bytes());
        block.extend_from_slice(&0x3c00u16.to_le_bytes()); // d = 1 exactly
        if quant == Quant::Q4_1 { block.extend_from_slice(&0x3800u16.to_le_bytes()); } // m = 0.5
        for i in 0..16u8 {
            let lo = (role.wrapping_add(expert).wrapping_add(row as u8).wrapping_add(kb).wrapping_add(i)) & 15;
            let hi = (lo + 7) & 15;
            block.push(lo | hi << 4);
        }
        block
    }

    fn fixture() -> Vec<u8> {
        let names = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"];
        let quants = [Quant::Q4_1, Quant::Q4_1, Quant::Q4_0];
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for (role, quant) in quants.iter().enumerate() {
            align(&mut data);
            offsets.push(data.len() as u64);
            for expert in 0..2u8 {
                for row in 0..128u16 {
                    for kb in 0..8u8 { data.extend(source_block(*quant, role as u8, expert, row, kb)); }
                }
            }
        }
        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        u32le(&mut file, 3);
        u64le(&mut file, names.len() as u64);
        u64le(&mut file, 3);
        string(&mut file, "general.architecture"); u32le(&mut file, 8); string(&mut file, "qwen35moe");
        string(&mut file, "general.alignment"); u32le(&mut file, 4); u32le(&mut file, 32);
        string(&mut file, "tokenizer.ggml.tokens"); u32le(&mut file, 9); u32le(&mut file, 8);
        u64le(&mut file, 2); string(&mut file, "a"); string(&mut file, "b");
        for ((name, quant), offset) in names.iter().zip(quants).zip(offsets) {
            string(&mut file, name);
            u32le(&mut file, 3);
            for d in [256, 128, 2] { u64le(&mut file, d); }
            u32le(&mut file, if quant == Quant::Q4_0 { 2 } else { 3 });
            u64le(&mut file, offset);
        }
        align(&mut file);
        file.extend(data);
        file
    }

    fn pool_code(chunk: &[u8], row: usize, col: usize) -> u8 {
        let p = (row / 16) * 4096 + col * 16 + row % 16;
        let b = chunk[2048 + p / 2];
        if p & 1 == 0 { b & 15 } else { b >> 4 }
    }

    #[test]
    fn metadata_tensor_index_and_expert_pools_preserve_q4_values() {
        let mut file = Gguf::new(Cursor::new(fixture())).unwrap();
        assert_eq!(file.metadata("general.architecture"), Some(&Metadata::String("qwen35moe".into())));
        assert_eq!(file.tensors().len(), 3);
        assert_eq!(file.tensor("blk.0.ffn_down_exps.weight").unwrap().dims, vec![256, 128, 2]);
        let mut stripes = Vec::new();
        let n = file.pack_expert_stripes("blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", &mut stripes).unwrap();
        assert_eq!(n as usize, 2 * 2 * 4 * POOL_CHUNK);
        for expert in 0..2usize {
            for role in 0..2usize {
                let chunks = &stripes[(expert * 8 + role * 4) * POOL_CHUNK..];
                for rb in 0..4usize {
                    let chunk = &chunks[rb * POOL_CHUNK..(rb + 1) * POOL_CHUNK];
                    assert_eq!(f32::from_le_bytes(chunk[0..4].try_into().unwrap()), 1.0);
                    assert_eq!(f32::from_le_bytes(chunk[1024..1028].try_into().unwrap()), 0.5);
                    for row in [0, 15, 16, 31] {
                        for col in [0, 17, 32, 255] {
                            let kb = (col / 32) as u8;
                            let i = (col % 32) as u8;
                            let lo = (role as u8 + expert as u8 + (rb * 32 + row) as u8 + kb + (i % 16)) & 15;
                            let expected = if i < 16 { lo } else { (lo + 7) & 15 };
                            assert_eq!(pool_code(chunk, row, col), expected);
                        }
                    }
                }
            }
        }
        let mut down = Vec::new();
        file.pack_expert_down("blk.0.ffn_down_exps.weight", &mut down).unwrap();
        assert_eq!(down.len(), 2 * 4 * POOL_CHUNK);
        let second = &down[4 * POOL_CHUNK..5 * POOL_CHUNK];
        assert_eq!(f32::from_le_bytes(second[1024..1028].try_into().unwrap()), -8.0);
        assert_eq!(pool_code(second, 0, 0), (2 + 1) & 15);
    }

    #[test]
    fn malformed_header_and_unsupported_quant_fail_before_packing() {
        let mut bad = fixture();
        bad[0] = b'X';
        assert!(Gguf::new(Cursor::new(bad)).err().unwrap().contains("not a GGUF"));
        let mut truncated = fixture();
        truncated.truncate(truncated.len() - 1);
        assert!(Gguf::new(Cursor::new(truncated)).err().unwrap().contains("past EOF"));
        assert!(Quant::from_type(12).unwrap_err().contains("exact Q4"));
        assert_eq!(f32_from_f16(1), 2f32.powi(-24));
    }

    #[test]
    fn direct_route_preflight_rejects_a_kernel_set_without_gguf() {
        let root = std::env::temp_dir().join(format!("phlegm-gguf-test-{}-{}", std::process::id(), NEXT_DIR.fetch_add(1, Ordering::Relaxed)));
        let model = root.join("model");
        let kernels = root.join("kernels");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::create_dir_all(&kernels).unwrap();
        std::fs::write(model.join("model.gguf"), fixture()).unwrap();
        std::fs::write(kernels.join("manifest.json"), b"{\"family\":\"qwen36moe\"}").unwrap();
        let err = preflight_direct_route(&model, &kernels).unwrap_err();
        assert!(err.contains("no direct GGUF route for qwen35moe"), "{err}");
        std::fs::write(kernels.join("manifest.json"), b"{\"gguf\":{}}").unwrap();
        preflight_direct_route(&model, &kernels).unwrap();
        std::fs::write(model.join("model.q4nx"), b"legacy").unwrap();
        assert!(preflight_direct_route(&model, &kernels).unwrap_err().contains("both model.gguf and model.q4nx"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
