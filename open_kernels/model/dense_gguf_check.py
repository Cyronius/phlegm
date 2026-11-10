"""No-XRT dense GGUF carriage gate, using gguf-py as the independent decoder.

Run with python -I. --packer is the patched gguf_pack_test binary. Every selected
projection is packed with its actual carriage, decoded, and compared bit for bit
with the specified rounding law. Source embed_row must also equal gguf-py bit for
bit. This does not accept a runtime manifest or establish per-layer dispatch.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_gguf_npulike import bf16, kept_q4
from nanbeige_gguf_check import HF_TO_GGUF, unpermute, std_perm_matrix, q8_head_matrix
from qwen35moe_gguf_check import Gguf, q41_chunks, q8_chunks, from_bf16, stats

KEPT = {"Q4_0", "Q4_1", "Q4_K"}
INT8 = {"Q8_0", "Q6_K", "Q5_K", "Q5_0", "Q5_1", "F16", "BF16"}
PROJECTIONS = {v for k, v in HF_TO_GGUF.items() if 'proj' in k}


def q8_matrix(buf, rows, cols):
    """Padded 5120-byte half-tiles, 64-row bands, 16-row quarters."""
    a = buf.reshape(-1, 5120)
    scales = from_bf16(a[:, :256].copy().view('<u2')).reshape(-1, 8, 16).transpose(0, 2, 1)
    codes = a[:, 256:4352].view(np.int8).reshape(-1, 256, 16).transpose(0, 2, 1)
    if np.any(a[:, 4352:]):
        raise AssertionError('nonzero q8 padding')
    values = codes.astype(np.float32) * np.repeat(scales, 32, axis=2)
    return values.reshape(rows // 64, cols // 256, 4, 16, 256).transpose(0, 2, 3, 1, 4).reshape(rows, cols)


def source_codes(raw, typ):
    if typ == 'Q4_K':
        a = raw.reshape(-1, 144)[:, 16:].reshape(-1, 4, 32)
        return np.stack((a & 15, a >> 4), axis=2).reshape(raw.shape[0], -1)
    a = raw.reshape(-1, 18 if typ == 'Q4_0' else 20)[:, 2 if typ == 'Q4_0' else 4:]
    return np.concatenate((a & 15, a >> 4), axis=1).reshape(raw.shape[0], -1)


def source_scales(raw, typ):
    if typ == 'Q4_K':
        a = raw.reshape(-1,144)
        s = a[:,4:16]
        ds = np.concatenate((s[:,:4] & 63, (s[:,8:] & 15) | ((s[:,:4] >> 6) << 4)), axis=1)
        ms = np.concatenate((s[:,4:8] & 63, (s[:,8:] >> 4) | ((s[:,4:8] >> 6) << 4)), axis=1)
        d = a[:,:2].copy().view('<f2').astype(np.float32) * ds
        m = -a[:,2:4].copy().view('<f2').astype(np.float32) * ms
    else:
        a = raw.reshape(-1,18 if typ == 'Q4_0' else 20)
        d = a[:,:2].copy().view('<f2').astype(np.float32)
        m = -8 * d if typ == 'Q4_0' else a[:,2:4].copy().view('<f2').astype(np.float32)
    return tuple(bf16(x).reshape(raw.shape[0],-1) for x in (d,m))


def scale_matrix(buf, offset, rows, cols):
    scales = from_bf16(buf[:,offset:offset+512].copy().view('<u2')).reshape(-1,8,32).transpose(0,2,1)
    return scales.reshape(rows//64, cols//256,2,32,8).transpose(0,2,3,1,4).reshape(rows,cols//32)


def q8_target(raw, typ, values, *, head=False):
    if typ == 'Q8_0':
        a = raw.reshape(-1, 34)
        return (bf16(a[:, :2].copy().view('<f2').astype(np.float32)) *
                a[:, 2:].view(np.int8).astype(np.float32)).reshape(values.shape)
    blocks = np.asarray(values, np.float32).reshape(-1, 32)
    scale = bf16(np.max(np.abs(blocks), axis=1) / np.float32(127))[:, None]
    if head:
        # Preserve lmhead_q8's existing reciprocal multiply, including half-step
        # boundaries; projection q8 packing uses division. std::lround rounds
        # the f32 ratio in unbounded precision, not f32(abs(ratio) + .5).
        inv = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
        ratio = blocks * inv
    else:
        ratio = np.divide(blocks, scale, out=np.zeros_like(blocks), where=scale != 0)
    code = np.copysign(np.floor(np.abs(ratio.astype(np.float64)) + .5), ratio).clip(-127,127).astype(np.int8)
    return (code.astype(np.float32) * scale).reshape(values.shape)


def hf_name(name):
    _, layer, module, _ = name.split('.')
    inverse = {v: k for k, v in HF_TO_GGUF.items()}
    return f'model.layers.{layer}.{inverse[module]}.weight'


def check_tensor(g, name, packer, work, head=False):
    t = g.t[name]
    typ = t.tensor_type.name
    if typ not in KEPT | INT8:
        raise ValueError(f'{name}: unsupported type {typ}')
    cols, rows = map(int, t.shape)
    op = 'lmhead_q8' if head else 'std_perm_gguf' if typ in KEPT else 'q8_perm'
    hf = 'lm_head.weight' if head else hf_name(name)
    output = work / 'tensor.bin'
    subprocess.run([str(packer), '--dump-tensor', str(g.path), hf, op, str(output)], check=True)
    # Compare in row bands to bound memory, including the tied head.
    source = np.memmap(str(output) + '.source', '<f4', 'r', shape=(rows, cols))
    packed = np.memmap(output, np.uint8, 'r')
    raw = t.data.reshape(rows, -1)
    arch = g.r.fields['general.architecture'].contents()
    heads = None
    if not head and arch == 'llama' and name.split('.')[2] in ('attn_q', 'attn_k'):
        key = 'llama.attention.head_count' + ('_kv' if '.attn_k.' in name else '')
        heads = int(g.r.fields[key].contents())
    # A head is 128-row supertiles; projections 64-row bands. Q/K unpermute
    # can cross bands, so decode a projection whole (head stays bounded).
    step = 4096 if head else rows
    se = sw = 0.0
    for r0 in range(0, rows, step):
        nr = min(step, rows-r0)
        values = g.dq(raw[r0:r0+nr], t.tensor_type).astype(np.float32).reshape(nr, cols)
        if not np.array_equal(source[r0:r0+nr].view('<u4'), values.view('<u4')):
            raise AssertionError(f'{name}: C++ source dequant differs from gguf-py')
        target = q8_target(raw[r0:r0+nr], typ, values, head=head) if op != 'std_perm_gguf' else kept_q4(raw, typ).reshape(rows, cols)
        if heads:
            values, target = unpermute(values, heads), unpermute(target, heads)
        nch = nr * cols // (4096 if op == 'q8_perm' else 8192)
        ch = 8704 if head else 5120
        start = r0 * cols // 8192 * ch if head else 0
        buf = np.asarray(packed[start:start+nch*ch]).reshape(nch, ch)
        got = (q8_head_matrix(q8_chunks(buf), nr, cols) if head else
               q8_matrix(buf, rows, cols) if op == 'q8_perm' else
               std_perm_matrix(q41_chunks(buf), rows, cols))
        if not np.array_equal(got.view('<u4'), target.view('<u4')):
            raise AssertionError(f'{name}: packed values differ from carriage oracle ({np.count_nonzero(got != target)} values)')
        if op == 'std_perm_gguf':
            from qwen35moe_gguf_check import CI
            nib = np.empty((nch, 8192), np.uint8)
            nib[:, 0::2], nib[:, 1::2] = buf[:, 1024:] & 15, buf[:, 1024:] >> 4
            got_codes = std_perm_matrix(nib[:, CI], rows, cols)
            want_codes = source_codes(raw, typ)
            if heads: want_codes = unpermute(want_codes, heads)
            if not np.array_equal(got_codes, want_codes):
                raise AssertionError(f'{name}: published Q4 codes changed')
            for offset, want_scale in zip((0,512), source_scales(raw, typ)):
                if heads: want_scale = unpermute(want_scale, heads)
                actual = scale_matrix(buf, offset, rows, cols)
                if not np.array_equal(actual.view('<u4'), want_scale.view('<u4')):
                    raise AssertionError(f'{name}: scale/offset plane differs')
        if typ == 'Q8_0':
            from qwen35moe_gguf_check import CI
            a8 = raw[r0:r0+nr].reshape(nr, -1, 34)
            want_code = a8[:,:,2:].view(np.int8).reshape(nr,cols)
            want_scale = bf16(a8[:,:,:2].copy().view('<f2').astype(np.float32)).reshape(nr,cols//32)
            if head:
                actual_code = q8_head_matrix(buf[:,512:].view(np.int8)[:,CI], nr, cols)
                sc = from_bf16(buf[:,:512].copy().view('<u2')).reshape(-1,8,32).transpose(0,2,1)
                actual_scale = sc.reshape(nr//128,cols//256,4,32,8).transpose(0,2,3,1,4).reshape(nr,cols//32)
            else:
                actual_code = buf[:,256:4352].view(np.int8).reshape(rows//64,cols//256,4,256,16).transpose(0,2,4,1,3).reshape(rows,cols)
                sc = from_bf16(buf[:,:256].copy().view('<u2')).reshape(rows//64,cols//256,4,8,16)
                actual_scale = sc.transpose(0,2,4,1,3).reshape(rows,cols//32)
            if heads: want_code, want_scale = unpermute(want_code,heads), unpermute(want_scale,heads)
            if not np.array_equal(actual_code, want_code) or not np.array_equal(actual_scale.view('<u4'),want_scale.view('<u4')):
                raise AssertionError(f'{name}: Q8_0 codes or scale changed')
        err = got.astype(np.float64) - values
        se += float(np.sum(err*err)); sw += float(np.sum(values.astype(np.float64)**2))
    rel = (se / max(sw, 1e-30)) ** .5
    if rel > .008:
        raise AssertionError(f'{name}: relative L2 {rel} > .008')
    result = dict(tensor=name, type=typ, op=op, bytes=output.stat().st_size,
                  relative_l2=rel, bit_exact_carriage=True, bit_exact_source=True)
    del packed, source
    output.unlink(); Path(str(output)+'.source').unlink()
    return result


def check_vector(g, name, packer, work):
    output = work / 'vector.bin'
    subprocess.run([str(packer), '--dump-tensor', str(g.path), name, 'put', str(output)], check=True)
    values = g.array(name).reshape(-1)
    source = np.fromfile(str(output)+'.source', '<f4')
    got = from_bf16(np.fromfile(output, '<u2'))
    if not np.array_equal(source.view('<u4'), values.view('<u4')):
        raise AssertionError(f'{name}: vector source differs from gguf-py')
    if not np.array_equal(got.view('<u4'), bf16(values).view('<u4')):
        raise AssertionError(f'{name}: norm is not RNE bf16')
    result = dict(tensor=name, type=g.type(name), op='put', bytes=output.stat().st_size,
                  relative_l2=stats(got, values)[0], bit_exact_carriage=True, bit_exact_source=True)
    output.unlink(); Path(str(output)+'.source').unlink()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', required=True, type=Path)
    ap.add_argument('--packer', required=True, type=Path)
    ap.add_argument('--gguf-py', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--layers', default='all')
    ap.add_argument('--head', action='store_true')
    ap.add_argument('--norms-only', action='store_true', help='check the norm vectors without repacking matrices')
    a = ap.parse_args()
    g = Gguf(str(a.gguf), a.gguf_py); g.path = a.gguf.resolve()
    layers = None if a.layers == 'all' else set(map(int, a.layers.split(',')))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='dense-pack-', dir=a.out.parent) as tmp:
        for name in (() if a.norms_only else g.t):
            parts = name.split('.')
            if len(parts) != 4 or parts[0] != 'blk' or parts[2] not in PROJECTIONS: continue
            if layers is not None and int(parts[1]) not in layers: continue
            result = check_tensor(g, name, a.packer.resolve(), Path(tmp))
            results.append(result); print(json.dumps(result), flush=True)
        for name in g.t:
            if not name.endswith('norm.weight'): continue
            if name.startswith('blk.') and layers is not None and int(name.split('.')[1]) not in layers: continue
            result = check_vector(g, name, a.packer.resolve(), Path(tmp))
            results.append(result); print(json.dumps(result), flush=True)
        if a.head:
            name = 'output.weight' if 'output.weight' in g.t else 'token_embd.weight'
            result = check_tensor(g, name, a.packer.resolve(), Path(tmp), head=True)
            results.append(result); print(json.dumps(result), flush=True)
    a.out.write_text(json.dumps(results, indent=2)+'\n')
    print(f'PASS {len(results)} tensors; zero tensors refitted to 4 bits')


if __name__ == '__main__':
    main()
