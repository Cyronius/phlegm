"""Independent float32 Llama forward pass over the GGUF's dequantized tensors.

Uses gguf's standard dequantizer and PyTorch SDPA, without NPU packing,
kernel arithmetic, or the C++ host attention. Outputs teacher-forced logits.
"""
import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--ids-file', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--at-position', type=int, default=0,
                    help='artificial zero-filled KV history before these tokens; not a retrieval test')
    ap.add_argument('--effective-scales', action='store_true',
                    help='also model the existing kernels\' BF16 scale/norm narrowing')
    ap.add_argument('--head-q8-repack', action='store_true',
                    help='use the explicitly labelled Q8/BF16-scale head target instead of the source head')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads(args.config.read_text())
    ids = [int(x) for x in args.ids_file.read_text().replace(',', ' ').split()]
    tensors = {t.name: t for t in gguf.GGUFReader(str(args.model)).tensors}
    lm_name = 'output.weight' if 'output.weight' in tensors else 'token_embd.weight'

    def weight(name, compute=True):
        t = tensors[name]
        if args.head_q8_repack and compute and name == lm_name:
            w = np.array(gguf.dequantize(t.data, t.tensor_type), copy=True).astype(np.float32)
            blocks = w.reshape(w.shape[0], -1, 32)
            scales = (np.max(np.abs(blocks), -1, keepdims=True) / np.float32(127)).astype(bfloat16).astype(np.float32)
            ratio = np.divide(blocks, scales, out=np.zeros_like(blocks), where=scales != 0)
            codes = np.copysign(np.floor(np.abs(ratio) + .5), ratio).clip(-127, 127)
            return torch.from_numpy((codes * scales).reshape(w.shape))
        if args.effective_scales and compute and t.tensor_type == gguf.GGMLQuantizationType.Q4_1:
            raw = t.data.reshape(int(t.shape[1]), -1, 20)
            d = np.ascontiguousarray(raw[..., :2]).view('<f2').astype(bfloat16).astype(np.float32)
            m = np.ascontiguousarray(raw[..., 2:4]).view('<f2').astype(bfloat16).astype(np.float32)
            q = np.concatenate((raw[..., 4:] & 15, raw[..., 4:] >> 4), -1).astype(np.float32)
            return torch.from_numpy((q * d + m).reshape(int(t.shape[1]), int(t.shape[0])))
        if args.effective_scales and compute and t.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            raw = t.data.reshape(int(t.shape[1]), -1, 34)
            d = np.ascontiguousarray(raw[..., :2]).view('<f2').astype(bfloat16).astype(np.float32)
            return torch.from_numpy((raw[..., 2:].view(np.int8).astype(np.float32) * d).reshape(int(t.shape[1]), int(t.shape[0])))
        w = torch.from_numpy(np.array(gguf.dequantize(t.data, t.tensor_type), copy=True)).float()
        return w.bfloat16().float() if args.effective_scales and compute and 'norm.weight' in name else w

    nh, nk = cfg['num_attention_heads'], cfg['num_key_value_heads']
    hd = cfg.get('head_dim', cfg['hidden_size'] // nh)
    eps = cfg['rms_norm_eps']
    inv = cfg['rope_theta'] ** (-torch.arange(0, hd, 2, dtype=torch.float64) / hd)
    if 'rope_freqs.weight' in tensors:
        inv /= weight('rope_freqs.weight').reshape(-1).double()
    elif cfg.get('rope_scaling'):
        sc = cfg['rope_scaling']
        if sc.get('rope_type', sc.get('type')) != 'llama3':
            raise ValueError('only full-rotary Llama3 scaling is supported')
        wave = 2 * math.pi / inv
        lo, hi, old = sc['low_freq_factor'], sc['high_freq_factor'], sc['original_max_position_embeddings']
        smooth = (old / wave - lo) / (hi - lo)
        scaled = inv / sc['factor']
        inv = torch.where(wave < old / hi, inv,
                          torch.where(wave > old / lo, scaled, (1 - smooth) * scaled + smooth * inv))
    if args.at_position < 0:
        ap.error('at-position must be nonnegative')
    phase = torch.outer(torch.arange(args.at_position, args.at_position + len(ids), dtype=torch.float64), inv).float()
    cs, sn = torch.cos(phase)[None], torch.sin(phase)[None]

    def norm(x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w

    def projection(x, name, heads=None):
        w = weight(name)
        if heads:
            # Convert adjacent rotary-pair rows in GGUF to the split-half basis.
            w = w.reshape(heads, hd // 2, 2, w.shape[-1]).transpose(1, 2).reshape(-1, w.shape[-1])
        return x @ w.T

    def rotate(x):
        a, b = x.chunk(2, -1)
        return torch.cat((a * cs - b * sn, b * cs + a * sn), -1)

    with torch.inference_mode():
        x = weight('token_embd.weight', compute=False)[ids]
        for layer in range(cfg['num_hidden_layers']):
            prefix = f'blk.{layer}.'
            z = norm(x, weight(prefix + 'attn_norm.weight'))
            q = rotate(projection(z, prefix + 'attn_q.weight', nh).reshape(-1, nh, hd).transpose(0, 1))
            k = rotate(projection(z, prefix + 'attn_k.weight', nk).reshape(-1, nk, hd).transpose(0, 1))
            v = projection(z, prefix + 'attn_v.weight').reshape(-1, nk, hd).transpose(0, 1)
            mask = None
            if args.at_position:
                empty = torch.zeros(nk, args.at_position, hd)
                k, v = torch.cat((empty, k), 1), torch.cat((empty, v), 1)
                mask = torch.arange(args.at_position + len(ids))[None, :] <= (
                    args.at_position + torch.arange(len(ids))[:, None])
            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(nh // nk, 0),
                                                  v.repeat_interleave(nh // nk, 0),
                                                  attn_mask=mask, is_causal=mask is None)
            x = x + projection(attn.transpose(0, 1).reshape(len(ids), -1), prefix + 'attn_output.weight')
            z = norm(x, weight(prefix + 'ffn_norm.weight'))
            h = F.silu(projection(z, prefix + 'ffn_gate.weight')) * projection(z, prefix + 'ffn_up.weight')
            x = x + projection(h, prefix + 'ffn_down.weight')
        z = norm(x, weight('output_norm.weight'))
        logits = projection(z, lm_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    logits.numpy().astype('<f4').tofile(args.out)
    print(json.dumps(dict(positions=len(ids), vocab=logits.shape[1], finite=bool(torch.isfinite(logits).all()),
                          head_q8_repack=args.head_q8_repack, effective_scales=args.effective_scales)))


if __name__ == '__main__':
    main()
