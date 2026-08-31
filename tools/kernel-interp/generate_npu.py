"""Text-in / text-out autoregressive generation of the interval-3 model.

Pipeline (a real prompt now drives the whole thing):

  text --chat template--> string --tokenizer--> ids
       --run_5li3_npu.py (CPU prefill)--> per-layer NPU states + prefill hidden
       --sample(prefill hidden)--> first generated token
       --NPU decode serve loop--> hidden per token --CPU full logits--> sample
       --detokenizer--> streamed text, until an eos token

The NPU runs all 5 transformer layers per token (the interval-3-critical path)
via decode_driver.exe `serve` (pools + states resident); the CPU computes the
full [248320] logits from the NPU's final hidden (the lm_head kernel emits only
the odd vocab half, so the generate loop does the projection on CPU with a
cached dequantized lm_head matrix) and a real Sampler chooses the next token.
This proves interval-3 generates FINITE, coherent text on hardware where FLM's
closed engine NaN-collapses ("////////").

Usage:
  # tokenizer + sampler + CPU prefill + first token — NO NPU needed:
  python generate_npu.py --prompt "What is the capital of France?" --cpu-first-only

  # full text-out on the NPU (needs the shared NPU device + built pools):
  python generate_npu.py --prompt "Explain gravity briefly." --max-tokens 40 \
      --temperature 0.7 --top-p 0.8 --seed 0

  # legacy: positional token count, default fixed prompt
  python generate_npu.py 8
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16
from tokenizer import Qwen36Tokenizer
from sampler import Sampler

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
except Exception:
    pass

D = "C:/code/FastFlowLM/npu-engine/m3out/5li3"
DRIVER = "C:/code/FastFlowLM/npu-engine/m0/out/decode_driver.exe"
XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
CAP = "C:/caps/m0c"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(MODEL_DIR, "model_5Li3.q4nx")

m = Q4NX(MODEL)
NW = m.bf16("model.norm.weight")


# ---- CPU lm_head projection (NPU emits only the odd vocab half) -------------
def _build_lmhead_matrix():
    """Dequantize lm_head ONCE into [248320, 2048] f32 (cached to disk, ~2GB)
    so per-token sampling is a single matmul instead of re-dequantizing 517MB."""
    cache = f"{D}/lmhead_W.f32.npy"
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    lmb = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = bc * 32 + r + 0 * i
    W = np.zeros((248320, 2048), dtype=np.float32)
    for c0 in range(0, lmb.shape[0], 8192):
        ce = min(c0 + 8192, lmb.shape[0])
        vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce - c0, 32, 8, 32).astype(np.float32)
        dd = d[c0:ce][:, j.reshape(-1)].reshape(ce - c0, 32, 8, 32)
        w = (vals * dd).reshape(ce - c0, 32, 256)
        for cc in range(c0, ce):
            W[32 * (cc // 8):32 * (cc // 8) + 32, 256 * (cc % 8):256 * (cc % 8) + 256] = w[cc - c0]
    np.save(cache, W)
    return W


_W = None


def full_logits(hidden_row0):
    """Raw residual hidden [2048] -> full [248320] f32 logits (rms + norm.weight
    folded in, then the lm_head matmul)."""
    global _W
    if _W is None:
        _W = _build_lmhead_matrix()
    hn = (hidden_row0 / np.sqrt((hidden_row0.astype(np.float64) ** 2).mean() + 1e-6) * NW).astype(np.float32)
    return _W @ hn


def embed(tok):
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    return bf16_to_f32(np.frombuffer(m.mm[base + tok * 4096:base + (tok + 1) * 4096], dtype=np.uint16))


def write_act(tok):
    act = np.zeros(1048576, dtype=np.uint8)
    act[:4096] = f32_to_bf16(embed(tok)).view(np.uint8)
    act[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act.tofile(f"{D}/gen_act.bin")


# ---- prompt -> prefill ------------------------------------------------------
def prepare_prompt(ids, rebuild_pools=False):
    """Write the prompt token ids and (re)build the NPU prefill buffers +
    per-layer states + prefill hidden via run_5li3_npu.py (pure CPU)."""
    np.save(os.path.join(HERE, "prompt_token_ids.npy"), np.asarray(ids, dtype=np.int64))
    env = dict(os.environ)
    if rebuild_pools:
        env["REBUILD_POOLS"] = "1"
    print(f"prefilling {len(ids)} prompt tokens (CPU)...", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_5li3_npu.py"), D],
                       cwd=HERE, env=env)
    if r.returncode != 0:
        raise RuntimeError("run_5li3_npu.py prefill failed")
    print(f"prefill done ({time.time()-t:.1f}s)", flush=True)


def first_token_logits():
    """Full logits for the FIRST generated token, from the prefill's last-token
    raw residual hidden (position T-1 predicts token T)."""
    res = np.load(f"{D}/prefill_final_residual.npy").astype(np.float64)
    return full_logits(res)


# ---- NPU decode serve loop --------------------------------------------------
def serve_config():
    """One persistent driver: all buffers + states resident, layer program
    between serve/endserve; tokens streamed over stdin."""
    lines = ["device", f"xclbin L {XB}/layer.xclbin", f"xclbin LM {XB}/lm_head.xclbin",
             f"kernel k0 L {CAP}/elf_000005.bin", f"kernel k1 L {CAP}/elf_000006.bin",
             f"kernel klm LM {CAP}/elf_000003.bin"]
    for l in range(5):
        lines.append(f"buf pool{l} 536870912 {D}/pool_L{l}.bin")
        lines.append(f"buf pack{l} 2097152 {D}/pack_L{l}.bin")
        lines.append(f"buf side{l} 6291456 {D}/side_L{l}.bin")
        lines.append(f"buf state{l} 3145728 {D}/state_L{l}.bin")   # initial = prefill state
    lines.append(f"buf lmpool 542113792 {D}/pool_lmhead.bin")
    lines.append("buf act 1048576")
    lines.append("buf logits 1048576")
    lines.append("serve")
    kern = lambda l: "k0" if l == 0 else "k1"
    chunk = []
    for l in range(5):
        chunk.append(l)
        if len(chunk) == 3:
            lines.append("runlist L")
            for c in chunk:
                lines.append(f"layer {kern(c)} pool{c} act pack{c} side{c} state{c}")
            lines.append("submit")
            lines.append("barrier klm logits lmpool act")
            chunk = []
    if chunk:
        lines.append("runlist L")
        for c in chunk:
            lines.append(f"layer {kern(c)} pool{c} act pack{c} side{c} state{c}")
        lines.append("submit")
    lines.append("barrier klm logits lmpool act")   # trailing barrier: reset context before next step's chunk
    lines.append("endserve")
    cfg = f"{D}/gen_serve.txt"
    open(cfg, "w").write("\n".join(lines) + "\n")
    return cfg


def npu_available():
    return os.path.exists(DRIVER) and os.path.exists(f"{D}/pool_L0.bin")


def generate_npu(first_tok, sampler, tok, history, max_tokens, stream=True):
    """Drive the NPU serve loop; sample each step; stream detokenized text.
    `history` starts as the prompt ids + first_tok (for repetition penalty)."""
    cfg = serve_config()
    proc = subprocess.Popen([DRIVER, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    while True:
        ln = proc.stdout.readline()
        if not ln:
            raise RuntimeError("driver died before READY (NPU busy? see handoff note)")
        if "SERVE READY" in ln:
            break
    print("driver ready (pools + states resident)\n", flush=True)

    generated = [first_tok]
    cur = first_tok
    printed = ""
    if stream:
        printed = tok.decode(generated)
        sys.stdout.write(printed); sys.stdout.flush()

    for step in range(max_tokens - 1):
        if tok.is_eos(cur):
            break
        write_act(cur)
        proc.stdin.write(f"step {D}/gen_act.bin {D}/gen_hidden.bin\n"); proc.stdin.flush()
        while True:
            ln = proc.stdout.readline()
            if not ln:
                raise RuntimeError("driver died mid-step")
            if "STEP OK" in ln:
                break
            if "STEP" in ln and ("ERR" in ln or "FAILED" in ln):
                raise RuntimeError("driver step failed: " + ln)
        hidden = bf16_to_f32(np.fromfile(f"{D}/gen_hidden.bin", dtype=np.uint16))[:2048]
        assert np.isfinite(hidden).all(), "NON-FINITE hidden (interval-3 blowup!)"
        lg = full_logits(hidden.astype(np.float64))
        nxt = sampler.sample(lg, history=history)
        history.append(nxt)
        generated.append(nxt)
        cur = nxt
        if stream:
            full = tok.decode(generated)
            sys.stdout.write(full[len(printed):]); sys.stdout.flush()
            printed = full
    proc.stdin.write("quit\n"); proc.stdin.flush()
    proc.wait()
    if stream:
        print()
    return generated


def build_sampler(a):
    return Sampler(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                   repetition_penalty=a.rep_penalty, seed=a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legacy_n", nargs="?", type=int, default=None,
                    help="legacy: token count with the default fixed prompt")
    ap.add_argument("--prompt", type=str, default=None, help="user text prompt")
    ap.add_argument("--system", type=str, default=None, help="optional system prompt")
    ap.add_argument("--raw", action="store_true",
                    help="treat --prompt as raw text (skip the chat template)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable the model's <think> reasoning block")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rebuild-pools", action="store_true",
                    help="force the ~2.5GB pool rebuild (else reuse if present)")
    ap.add_argument("--cpu-first-only", action="store_true",
                    help="prefill + sample the FIRST token on CPU, no NPU (validation)")
    args = ap.parse_args()

    tok = Qwen36Tokenizer()
    sampler = build_sampler(args)

    # ---- legacy mode: positional token count, fixed prompt, no text I/O ----
    if args.prompt is None and args.legacy_n is not None:
        n = args.legacy_n
        first = int(np.load(f"{D}/first_token.npy")) if os.path.exists(f"{D}/first_token.npy") else 276
        if not npu_available():
            print("NPU driver/pools not present — HANDOFF (see module docstring).")
            return
        gen = generate_npu(first, sampler, tok, [], n, stream=False)
        print("GENERATED token ids:", gen)
        print("decoded:", repr(tok.decode(gen)))
        return

    prompt_text = args.prompt if args.prompt is not None else "What is the capital of France?"

    # ---- text -> ids (chat template unless --raw) --------------------------
    if args.raw:
        prompt_str = prompt_text
    else:
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": prompt_text})
        prompt_str = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             enable_thinking=not args.no_think)
    ids = tok.encode(prompt_str)
    print(f"prompt ({len(ids)} tokens): {prompt_text!r}")
    print("rendered:", repr(prompt_str[:120]) + ("..." if len(prompt_str) > 120 else ""))

    # ---- CPU prefill (prompt drives it) ------------------------------------
    prepare_prompt(ids, rebuild_pools=args.rebuild_pools)

    # ---- first generated token (CPU lm_head on the prefill hidden) ---------
    lg0 = first_token_logits()
    assert np.isfinite(lg0).all(), "non-finite prefill logits (interval-3 blowup!)"
    history = list(ids)
    first = sampler.sample(lg0, history=history)
    history.append(first)
    print(f"first token: {first} ({tok.id_to_token(first)!r}) "
          f"logit {lg0[first]:.2f}, absmax {np.abs(lg0).max():.2f}, finite=True")

    if args.cpu_first_only or not npu_available():
        if not args.cpu_first_only:
            print("\nNPU driver/pools not available — decode is a HANDOFF.")
        print("\n--- generated (first token only, CPU) ---")
        print(tok.decode([first]))
        print("\nHANDOFF: run without --cpu-first-only on the NPU box (device idle,")
        print("pools built) to stream the full continuation via decode_driver serve.")
        return

    # ---- NPU decode: stream the continuation -------------------------------
    print("\n--- generated ---")
    gen = generate_npu(first, sampler, tok, history, args.max_tokens, stream=True)
    finite = "all finite (interval-3 healthy)"
    print(f"\n[{len(gen)} tokens, {finite}]")
    print("token ids:", gen)


if __name__ == "__main__":
    main()
