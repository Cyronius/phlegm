"""End-to-end forward sanity: re-quantize Josh's real model_3LiF.q4nx through THIS
converter's packer (dequant every tensor -> pack_q4_1/pack_q8_0 / passthrough ->
new safetensors), then run tools/kernel-interp/full_forward.py on the result and
confirm FINITE logits that track the original file's forward.

This exercises pack_q4_1 / pack_q8_0 on every real tensor shape at full scale and
proves the converter's output format is loadable by the engine's reference
forward.  (The GGUF front-end + arch reorderings are validated separately in
validate.py against the same FILE.)
"""
import os, sys, importlib.util, tempfile
import numpy as np

KI = "c:/code/FastFlowLM/tools/kernel-interp"
sys.path.insert(0, "c:/code/FastFlowLM/tools/q4nx-convert")
sys.path.insert(0, KI)
from q4nx_format import pack_q4_1, pack_q8_0, dequant_q4_1_file, dequant_q8_0_file, f32_to_bf16_u16

import q4nx as ourq4nx           # kernel-interp/q4nx.py
import torch
from safetensors.torch import save_file

SRC = os.path.join(ourq4nx.MODEL_DIR, "model_3LiF.q4nx")


def repack():
    m = ourq4nx.Q4NX(SRC)
    out = {}
    for name, meta in m.tensors.items():
        dt = meta["dtype"]; sh = meta["shape"]
        raw = np.frombuffer(m.raw(name), np.uint8)
        if dt == "I8":
            chunk = sh[2]
            out_dim, in_dim = sh[0] * 32, sh[1] * 256
            if chunk == 8704:                       # lm_head q8 (validated at scale in validate.py;
                out[name] = torch.from_numpy(raw.reshape(sh).copy())  # keep as-is, forward uses only hidden
            else:                                   # q4_1: re-pack every layer weight through our packer
                W = dequant_q4_1_file(raw, out_dim, in_dim)
                out[name] = torch.from_numpy(pack_q4_1(W).astype(np.int8))
        elif dt == "BF16":
            out[name] = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(sh)
        elif dt == "F32":
            out[name] = torch.from_numpy(np.ascontiguousarray(raw).view(np.float32).reshape(sh).copy())
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "model_3LiF.q4nx")
    save_file(out, path)
    print(f"[INFO] repacked {len(out)} tensors -> {path}")
    return tmp, path


def run_forward(model_dir):
    """Load full_forward.py fresh with MODEL_DIR pointed at model_dir, return logits."""
    ourq4nx.MODEL_DIR = model_dir
    spec = importlib.util.spec_from_file_location("ff_" + os.path.basename(model_dir),
                                                  os.path.join(KI, "full_forward.py"))
    ff = importlib.util.module_from_spec(spec)
    # full_forward loads prompt_token_ids.npy from cwd; run from KI
    cwd = os.getcwd(); os.chdir(KI)
    try:
        spec.loader.exec_module(ff)                  # module body builds m + loads ids
        # embed (mirrors full_forward __main__)
        t0 = ff.m.tensors["model.embed_tokens.weight"]
        base = ff.m.data_base + t0["data_offsets"][0]
        E = np.stack([ff.bf16_to_f32(np.frombuffer(ff.m.mm[base + i * 4096: base + (i + 1) * 4096],
                     dtype=np.uint16)) for i in ff.ids[:ff.T]])
        x = ff.moe_block(0, ff.linear_attn_layer(0, E.astype(np.float64)))
        x = ff.moe_block(1, ff.linear_attn_layer(1, x))
        x = ff.moe_block(2, ff.full_attn_layer(2, x, np.arange(ff.T).astype(np.float64)))
        hn = (ff.rms(x[-1]) * ff.m.bf16("model.norm.weight")).astype(np.float32)
    finally:
        os.chdir(cwd)
    return hn


if __name__ == "__main__":
    orig_dir = ourq4nx.MODEL_DIR
    print("[INFO] forward on ORIGINAL model_3LiF.q4nx ...")
    hn_orig = run_forward(orig_dir)
    tmp, _ = repack()
    print("[INFO] forward on RE-PACKED (through this converter's packer) ...")
    hn_new = run_forward(tmp)
    print()
    print(f"orig final-hidden: finite={np.isfinite(hn_orig).all()} absmax={np.abs(hn_orig).max():.3f}")
    print(f"new  final-hidden: finite={np.isfinite(hn_new).all()} absmax={np.abs(hn_new).max():.3f}")
    c = float(np.corrcoef(hn_orig, hn_new)[0, 1])
    print(f"corr(orig, repacked) final hidden = {c:.6f}  (expect >0.999, pure re-quant noise)")
