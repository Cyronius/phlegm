"""Apples-to-apples end-to-end tok/s benchmark of the OPEN engine vs FLM, on the
SAME 40-layer base model (Qwen3.6-35B-A3B, interval-4) and the SAME full path
(NPU decode + CPU sampling + detok), so it is directly comparable to FLM's
measured decoding_speed_tps.

Uses:
  - 40 RESIDENT pools (no per-token disk reload)     [the streaming fix]
  - PING-PONG across two layer.xclbin contexts, NO lm_head barriers
    (CPU computes logits from the dumped hidden)      [the barrier fix]
Times steady-state per-token wall clock (NPU step + CPU lm_head + sample + I/O),
dropping warmup tokens.

Prereq: python l30_build.py <base model.q4nx> <L40 dir>   (builds the 40 pools).
Usage:  python bench_e2e_l40.py [n_tokens=32]
"""
import os, re, sys, time, subprocess
import numpy as np
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16
from sampler import Sampler

D = os.environ.get("L40_DIR", "C:/code/FastFlowLM/npu-engine/m3out/l40")
DRIVER = "C:/code/FastFlowLM/npu-engine/m0/out/decode_driver_nobarrier.exe"
XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
ELF = "C:/caps/m0c/elf_000005.bin"          # universal layer executor
MODEL = os.environ.get("BASE_MODEL", os.path.join(MODEL_DIR, "model.q4nx"))
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 32
POKE = os.environ.get("POKE", "0") != "0"        # per-token 480B position poke.
# Default OFF: measured to have NO effect on decode output (pos 19 vs 500 ->
# identical greedy tokens, both layer contexts poked). The decode kernel
# self-tracks its KV position inside the resident state buffer (M4's replay
# was byte-exact across multiple tokens with no pokes). Kept for future
# prefill-side ELF patching experiments.
POKE_TPL = os.environ.get("POKE_TPL", "C:/caps/i4/000797.seq")
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_LEN = 19
POS0 = int(os.environ.get("POS0", "-1"))         # debug: override start position

m = Q4NX(MODEL)
NW = m.bf16("model.norm.weight")
nlayers = 0
while f"model.layer.{nlayers}.input_layernorm.weight" in m.tensors:
    nlayers += 1
print(f"model {os.path.basename(MODEL)}  nlayers={nlayers}  fmt={m.fmt}")


def lmhead_W():
    cache = f"{D}/lmhead_W.f32.npy"
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    lmb = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16); j = bc * 32 + r + 0 * i
    W = np.zeros((248320, 2048), np.float32)
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
def full_logits(h):
    global _W
    if _W is None:
        _W = lmhead_W()
    hn = (h / np.sqrt((h.astype(np.float64) ** 2).mean() + 1e-6) * NW).astype(np.float32)
    return _W @ hn


def embed(tok):
    t0 = m.tensors["model.embed_tokens.weight"]; base = m.data_base + t0["data_offsets"][0]
    return bf16_to_f32(np.frombuffer(m.mm[base + tok * 4096:base + (tok + 1) * 4096], dtype=np.uint16))


def write_act(tok):
    act = np.zeros(1048576, np.uint8)
    act[:4096] = f32_to_bf16(embed(tok)).view(np.uint8)
    act[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act.tofile(f"{D}/bench_act.bin")


LM_ELF = "C:/caps/m0c/elf_000003.bin"       # lm_head: norm + full-vocab projection


def serve_config():
    """40 resident pools + ping-pong 2 contexts + NPU lm_head at end of step.
    The lm_head runs on its own xclbin context, so it doubles as the trailing
    cross-context barrier (FLM's own architecture, per the seq-capture research)."""
    L = ["device", f"xclbin L {XB}/layer.xclbin", "context L2 L",
         f"xclbin LM {XB}/lm_head.xclbin",
         f"kernel kL L {ELF}", f"kernel kL2 L2 {ELF}", f"kernel klm LM {LM_ELF}"]
    for l in range(nlayers):
        L.append(f"buf pool{l} 536870912 {D}/pool_L{l}.bin")
        L.append(f"buf pack{l} 2097152 {D}/pack_L{l}.bin")
        L.append(f"buf side{l} 6291456 {D}/side_L{l}.bin")
        L.append(f"buf state{l} 3145728 {D}/state_L{l}.bin")
    L.append(f"buf lmpool 542113792 {D}/pool_lmhead.bin")
    L.append("buf logits 1048576")
    L.append("buf act 1048576")
    if POKE:
        # per-token position ELF template (FLM's 480B seqlen poke). Tile memory
        # is per-hw_context, so BOTH layer contexts get poked. Order L,L2:
        # the L2 poke resets L's run budget before chunk 0 (on L) submits 3.
        L.append(f"poketpl L,L2 {POKE_TPL}")
    # servep: pipelined submits (fastest: 7.73 tok/s; tokens verified identical
    # to serve). serve: sequential submit/wait (7.5). serveq: prebuilt runlists
    # reused per token (SLOWER, 6.4: XRT re-execute revalidation costs more
    # than fresh construction).
    L.append(os.environ.get("SERVE_MODE", "servep"))
    g = 0
    for c0 in range(0, nlayers, 3):
        ctx, kn = ("L", "kL") if g % 2 == 0 else ("L2", "kL2")
        L.append(f"runlist {ctx}")
        for l in range(c0, min(c0 + 3, nlayers)):
            L.append(f"layer {kn} pool{l} act pack{l} side{l} state{l}")
        L.append("submit")
        g += 1
    L.append("lmhead klm logits lmpool act")
    L.append("endserve")
    cfg = f"{D}/bench_serve.txt"
    open(cfg, "w").write("\n".join(L) + "\n")
    return cfg


def main():
    if not os.path.exists(f"{D}/pool_L{nlayers-1}.bin"):
        print(f"MISSING pools in {D} (run l30_build.py on the base model first)"); return
    first = int(np.load(f"{D}/first_token.npy")) if os.path.exists(f"{D}/first_token.npy") else 276
    global PROMPT_LEN
    PROMPT_LEN = len(np.load(os.path.join(HERE, "prompt_token_ids.npy"))) \
        if os.path.exists(os.path.join(HERE, "prompt_token_ids.npy")) else 19
    print(f"poke={'on' if POKE else 'off'} prompt_len={PROMPT_LEN}")
    cfg = serve_config()
    proc = subprocess.Popen([DRIVER, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    while True:
        ln = proc.stdout.readline()
        if not ln:
            raise RuntimeError("driver died before READY")
        if "SERVE READY" in ln:
            break
    print("driver ready (40 pools + states resident, ping-pong)\n", flush=True)

    sampler = Sampler(temperature=0.0)
    cur, hist = first, [first]
    tstep = []
    # Per-step phase timings (seconds). ipc_h2d/npu_submit_wait/ipc_d2h come
    # from the driver's "MARK recv/h2d/npu/d2h" stdout lines (steady_clock
    # deltas inside the C++ process); write_act/send/read_hidden/lmhead/sample
    # are timed here in Python. See docs/lm-head-npu-bottleneck-instrumentation.md.
    phases = {k: [] for k in
              ("write_act", "send", "ipc_h2d", "npu_submit_wait", "npu_lmhead", "ipc_d2h", "read_logits", "sample")}
    # Per-chunk timing: one (chunk_idx, ctx, n_layers, dur_ms) per submit/wait
    # round trip inside a step, from "MARK c<idx>_<ctx>_n<n>_start/end" lines.
    CHUNK_RE = re.compile(r"c(\d+)_(\w+)_n(\d+)_(start|end)")
    chunk_rows = []  # list of (step, idx, ctx, n, dur_ms) across all steps
    for step in range(NTOK):
        t0 = time.time()
        write_act(cur)
        t1 = time.time()
        # decode position: prompt length + steps generated so far (first_token
        # was sampled at position T-1, so step 0 decodes it at position T)
        pos = (POS0 if POS0 >= 0 else PROMPT_LEN) + step if POKE else -1
        proc.stdin.write(f"step {D}/bench_act.bin {D}/bench_hidden.bin logits {D}/bench_logits.bin {pos}\n")
        proc.stdin.flush()
        t2 = time.time()
        marks = {}
        while True:
            ln = proc.stdout.readline()
            if not ln:
                raise RuntimeError("driver died mid-step")
            if ln.startswith("MARK "):
                _, label, ms = ln.split()
                marks[label] = float(ms)
                continue
            if "STEP OK" in ln:
                break
            if "STEP" in ln and ("ERR" in ln or "FAILED" in ln):
                raise RuntimeError("step failed: " + ln)
        t3 = time.time()
        # NPU lm_head output: full vocab as bf16[248320] (NOT f32 odd-half --
        # see the research findings in the plan doc)
        lg = bf16_to_f32(np.fromfile(f"{D}/bench_logits.bin", dtype=np.uint16)[:248320])
        t4 = time.time()
        if step == 0:
            hidden = bf16_to_f32(np.fromfile(f"{D}/bench_hidden.bin", dtype=np.uint16))[:2048]
            if not np.isfinite(hidden).all():
                print("(note: zero-state pools-only build -> non-finite decode; TIMING ONLY)", flush=True)
            else:
                cpu_arg = int(np.argmax(full_logits(hidden.astype(np.float64))))
                npu_arg = int(np.argmax(np.nan_to_num(lg, nan=-1e30)))
                print(f"(sanity: argmax NPU {npu_arg} vs CPU {cpu_arg} -> "
                      f"{'MATCH' if npu_arg == cpu_arg else 'MISMATCH'})", flush=True)
        nxt = sampler.sample(np.nan_to_num(lg), history=hist)
        hist.append(nxt); cur = nxt
        t5 = time.time()
        tstep.append(t5 - t0)
        phases["write_act"].append(t1 - t0)
        phases["send"].append(t2 - t1)
        phases["read_logits"].append(t4 - t3)
        phases["sample"].append(t5 - t4)
        if {"recv", "h2d", "npu", "d2h"} <= marks.keys():
            phases["ipc_h2d"].append((marks["h2d"] - marks["recv"]) / 1000.0)
            phases["npu_submit_wait"].append((marks["npu"] - marks["h2d"]) / 1000.0)
            phases["ipc_d2h"].append((marks["d2h"] - marks["npu"]) / 1000.0)
        if {"lmh_start", "lmh_end"} <= marks.keys():
            phases["npu_lmhead"].append((marks["lmh_end"] - marks["lmh_start"]) / 1000.0)
        for label, ms in marks.items():
            m = CHUNK_RE.fullmatch(label)
            if m and m.group(4) == "start":
                idx, ctx, n = int(m.group(1)), m.group(2), int(m.group(3))
                end_label = f"c{idx}_{ctx}_n{n}_end"
                if end_label in marks:
                    chunk_rows.append((step, idx, ctx, n, marks[end_label] - ms))
    proc.stdin.write("quit\n"); proc.stdin.flush(); proc.wait()
    print("tokens:", hist[:16], "..." if len(hist) > 16 else "")

    warm = 3
    steady = tstep[warm:]
    avg = sum(steady) / len(steady)
    print(f"\n=== END-TO-END (40 layers, full path: NPU layers + NPU lm_head + sample + I/O) ===")
    print(f"tokens: {NTOK} (first {warm} dropped as warmup)")
    print(f"per-token: mean {avg*1000:.1f} ms  => {1/avg:.2f} tok/s")
    print(f"          min {min(steady)*1000:.1f}  max {max(steady)*1000:.1f} ms")
    print(f"FLM measured baseline (same 40L model, end-to-end serve): 7.05 tok/s")

    print(f"\n=== PHASE BREAKDOWN (mean over {len(steady)} steady-state steps) ===")
    total_ms = avg * 1000.0
    for name, vals in phases.items():
        v = vals[warm:]
        if not v:
            print(f"{name:16s}  (no data)")
            continue
        m = sum(v) / len(v) * 1000.0
        print(f"{name:16s}  mean {m:7.2f} ms  min {min(v)*1000:7.2f}  max {max(v)*1000:7.2f}  "
              f"({100 * m / total_ms:5.1f}% of total)")

    steady_chunks = [r for r in chunk_rows if r[0] >= warm]
    if steady_chunks:
        by_idx = {}
        for _, idx, ctx, n, dur in steady_chunks:
            by_idx.setdefault(idx, {"ctx": ctx, "n": n, "durs": []})["durs"].append(dur)
        npu_total = sum(sum(v["durs"]) for v in by_idx.values()) / len(steady) if steady else 0
        print(f"\n=== PER-CHUNK BREAKDOWN ({len(by_idx)} chunks/token, "
              f"{sum(len(v['durs']) for v in by_idx.values())} submits over {len(steady)} steps) ===")
        print(f"{'chunk':>5}  {'ctx':4} {'nlayers':7}  {'mean ms':>8} {'min':>7} {'max':>7}  {'%step':>6}")
        for idx in sorted(by_idx):
            v = by_idx[idx]["durs"]
            m = sum(v) / len(v)
            print(f"{idx:5d}  {by_idx[idx]['ctx']:4} {by_idx[idx]['n']:7d}  "
                  f"{m:8.2f} {min(v):7.2f} {max(v):7.2f}  {100*m/total_ms:6.1f}")
        print(f"sum of per-chunk means: {npu_total:.2f} ms "
              f"(cross-check vs. npu_submit_wait phase mean above)")
        # First chunk on each context vs. later chunks on the same context:
        # tests whether a ping-pong context switch itself costs extra.
        first_on_ctx, later_on_ctx = [], []
        seen_ctx = set()
        for idx in sorted(by_idx):
            ctx = by_idx[idx]["ctx"]
            (first_on_ctx if ctx not in seen_ctx else later_on_ctx).extend(by_idx[idx]["durs"])
            seen_ctx.add(ctx)
        if first_on_ctx and later_on_ctx:
            print(f"first submit per context: mean {sum(first_on_ctx)/len(first_on_ctx):.2f} ms  "
                  f"(n={len(first_on_ctx)})")
            print(f"later submits, same ctx:  mean {sum(later_on_ctx)/len(later_on_ctx):.2f} ms  "
                  f"(n={len(later_on_ctx)})")


if __name__ == "__main__":
    main()
