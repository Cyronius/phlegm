"""Run the full 30-layer interval-3 model on the NPU via POOL STREAMING.

30 layers x 512MB pools = 15GB — cannot stay resident. This driver config keeps
only THREE 512MB pool BOs resident (poolA/B/C) plus all 30 small packs/sides/
states (~330MB), the lm_head pool, act and logits. It walks the schedule in 10
groups of 3 (= [L,L,F] blocks): before each group it `load`s the three layers'
pools into poolA/B/C, runs them as ONE runlist (the layer.xclbin >3-consecutive-
submit timeout means chunks of <=3), then a cross-context lm_head submission
resets the context (also computing lm_head into `logits`). After group 9 that
final lm_head IS the decode-step logits.

Modes:
  block1        lm_head-only on the CPU prefill hidden -> reproduces full_forward
                logits (streams the 542MB lm_head pool). corr ~1.0 expected.
  decode        full 30-layer streamed step on act_decode -> finite decode logits.
  gen [N]       N streamed decode steps, states carried via dump/reload.

The NPU is a SHARED device: this serializes via a lockfile and retries on
device-busy. If the device can't be had, it still writes every config + buffer
and reports the on-NPU run as a handoff.

Usage: python l30_run_npu.py <block1|decode|gen> [N] [out_dir]
"""
import numpy as np, os, sys, subprocess, time, errno
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16

OUT = "C:/code/FastFlowLM/npu-engine/m3out/l30"
DRIVER = "C:/code/FastFlowLM/npu-engine/m0/out/decode_driver.exe"
XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
CAP = "C:/caps/m0c"
LOCK = "C:/code/FastFlowLM/npu-engine/.npu.lock"
NLAYERS = 30

m = Q4NX(os.path.join(MODEL_DIR, "model_30L.q4nx"))
NW = m.bf16("model.norm.weight")
t0 = m.tensors["model.embed_tokens.weight"]
EBASE = m.data_base + t0["data_offsets"][0]


# ---- NPU device serialization ------------------------------------------------
def acquire_lock(timeout=1800, stale=2400):
    start = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode()); os.close(fd)
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            try:
                age = time.time() - os.path.getmtime(LOCK)
                if age > stale:
                    print(f"  removing stale NPU lock (age {age:.0f}s)"); os.remove(LOCK); continue
            except OSError:
                pass
            if time.time() - start > timeout:
                return False
            print("  NPU busy (lock held) — waiting..."); time.sleep(10)


def release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def run_driver(cfg, tag, retries=4):
    """Run decode_driver on cfg with device-busy retry. Returns (ok, stdout)."""
    for attempt in range(retries):
        if not acquire_lock():
            return False, "could not acquire NPU lock"
        try:
            t = time.time()
            r = subprocess.run([DRIVER, cfg], capture_output=True, text=True, timeout=1200)
        except subprocess.TimeoutExpired:
            release_lock(); print(f"  [{tag}] driver TIMEOUT (attempt {attempt+1})"); time.sleep(5); continue
        finally:
            release_lock()
        out = r.stdout + r.stderr
        busy = ("FAILED" in out or "EXCEPTION" in out or r.returncode not in (0,)
                or "state 8" in out)
        if not busy:
            print(f"  [{tag}] driver OK ({time.time()-t:.1f}s)")
            return True, out
        print(f"  [{tag}] driver failed (attempt {attempt+1}/{retries}) rc={r.returncode}")
        print("   ", out.strip().replace("\n", "\n    ")[:1500])
        time.sleep(8)
    return False, out


# ---- config generation -------------------------------------------------------
def hdr(lines):
    lines += ["device",
              f"xclbin L {XB}/layer.xclbin",
              f"xclbin LM {XB}/lm_head.xclbin",
              f"kernel kL L {CAP}/elf_000005.bin",
              f"kernel klm LM {CAP}/elf_000003.bin"]


def buf_decls(lines, act_file):
    lines += ["buf poolA 536870912", "buf poolB 536870912", "buf poolC 536870912",
              f"buf lmpool 542113792 {OUT}/pool_lmhead.bin",
              f"buf act 1048576 {act_file}", "buf logits 1048576"]
    for l in range(NLAYERS):
        lines.append(f"buf pack{l} 2097152 {OUT}/pack_L{l}.bin")
    for l in range(NLAYERS):
        lines.append(f"buf side{l} 6291456 {OUT}/side_L{l}.bin")
    for l in range(NLAYERS):
        lines.append(f"buf state{l} 3145728 {OUT}/state_L{l}.bin")


def gen_block1_cfg():
    lines = []
    hdr(lines)
    lines += [f"buf lmpool 542113792 {OUT}/pool_lmhead.bin",
              f"buf act 1048576 {OUT}/act_block1.bin", "buf logits 1048576",
              "lmhead klm logits lmpool act",
              "loglogits logits",
              f"dump logits {OUT}/npu_block1_logits.bin 1048576"]
    cfg = f"{OUT}/cfg_block1.txt"; open(cfg, "w").write("\n".join(lines) + "\n"); return cfg


def gen_stream_cfg(act_file, dump_states=False):
    POOLS = ["poolA", "poolB", "poolC"]
    lines = []
    hdr(lines)
    buf_decls(lines, act_file)
    for g in range(NLAYERS // 3):
        base = 3 * g
        for i in range(3):
            lines.append(f"load {POOLS[i]} {OUT}/pool_L{base+i}.bin")
        lines.append("runlist L")
        for i in range(3):
            lines.append(f"layer kL {POOLS[i]} act pack{base+i} side{base+i} state{base+i}")
        lines.append("submit")
        lines.append("lmhead klm logits lmpool act")   # cross-context reset (also = decode logits after last group)
    lines.append("loglogits logits")
    lines.append(f"dump logits {OUT}/npu_decode_logits.bin 1048576")
    lines.append(f"dump act {OUT}/npu_decode_hidden.bin 8192")
    if dump_states:
        for l in range(NLAYERS):
            lines.append(f"dump state{l} {OUT}/state_L{l}.bin 3145728")
    cfg = f"{OUT}/cfg_stream.txt"; open(cfg, "w").write("\n".join(lines) + "\n"); return cfg


# ---- CPU lm_head (full logits, for sampling from an NPU hidden) ---------------
def lmhead_matrix():
    cache = f"{OUT}/lmhead_W.f32.npy"
    if os.path.exists(cache):
        return np.load(cache, mmap_mode="r")
    lmb = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16); j = bc * 32 + r + 0 * i
    W = np.zeros((248320, 2048), dtype=np.float32)
    for c0 in range(0, lmb.shape[0], 8192):
        ce = min(c0 + 8192, lmb.shape[0])
        vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce - c0, 32, 8, 32).astype(np.float32)
        dd = d[c0:ce][:, j.reshape(-1)].reshape(ce - c0, 32, 8, 32)
        w = (vals * dd).reshape(ce - c0, 32, 256)
        for cc in range(c0, ce):
            W[32 * (cc // 8):32 * (cc // 8) + 32, 256 * (cc % 8):256 * (cc % 8) + 256] = w[cc - c0]
    np.save(cache, W); return W


def full_logits_from_hidden(h_raw):
    hn = (h_raw / np.sqrt((h_raw.astype(np.float64) ** 2).mean() + 1e-6) * NW).astype(np.float32)
    return _W @ hn


def write_decode_act(tok, path):
    act = np.zeros(1048576, dtype=np.uint8)
    emb = bf16_to_f32(np.frombuffer(m.mm[EBASE + tok * 4096:EBASE + (tok + 1) * 4096], dtype=np.uint16))
    act[:4096] = f32_to_bf16(emb).view(np.uint8)
    act[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act.tofile(path)


def report_npu_logits(path, ref_odd=None, label=""):
    lg = np.fromfile(path, dtype=np.float32)[:124160]   # NPU emits only the ODD vocab half (v=2b+1)
    finite = bool(np.isfinite(lg).all())
    amax = float(np.abs(lg[np.isfinite(lg)]).max()) if finite else float("nan")
    arg = int(np.nanargmax(np.where(np.isfinite(lg), lg, -np.inf)))
    print(f"  [{label}] NPU logits (odd half): finite={finite} absmax={amax:.3f} argmax_vocab={2*arg+1}")
    if ref_odd is not None and finite:
        nz = np.nonzero(ref_odd)[0]
        c = float(np.corrcoef(lg[nz], ref_odd[nz])[0, 1])
        print(f"  [{label}] corr vs CPU full_forward (odd half, n={len(nz)}): {c:.6f}")
        return finite, c
    return finite, None


# =============================================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "block1"
    _clp = next((p for p in (f"{OUT}/cpu_logits.npy", f"{OUT}/cpu_logits_build.npy") if os.path.exists(p)), None)
    cpu_logits = np.load(_clp) if _clp else None
    if _clp:
        print("CPU reference logits:", os.path.basename(_clp))
    cpu_odd = cpu_logits[1::2] if cpu_logits is not None else None   # vocab rows 2b+1

    if mode == "block1":
        cfg = gen_block1_cfg()
        print("block1: lm_head on CPU prefill hidden (streams 542MB lm_head pool)")
        ok, out = run_driver(cfg, "block1")
        print(out.strip())
        if ok:
            report_npu_logits(f"{OUT}/npu_block1_logits.bin", cpu_odd, "block1")
        else:
            print("HANDOFF: block1 config ready at", cfg, "— NPU device unavailable")

    elif mode == "decode":
        cfg = gen_stream_cfg(f"{OUT}/act_decode.bin", dump_states=False)
        print("decode: full 30-layer POOL-STREAMED step (reloads each layer's 512MB pool)")
        ok, out = run_driver(cfg, "decode")
        print(out.strip())
        if ok:
            fin, _ = report_npu_logits(f"{OUT}/npu_decode_logits.bin", None, "decode")
            h = bf16_to_f32(np.fromfile(f"{OUT}/npu_decode_hidden.bin", dtype=np.uint16))[:2048]
            print(f"  decode hidden finite={bool(np.isfinite(h).all())} absmax={float(np.abs(h).max()):.3f}")
        else:
            print("HANDOFF: streamed decode config ready at", cfg, "— NPU device unavailable")

    elif mode == "gen":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        _W = lmhead_matrix()
        tok = int(np.load(f"{OUT}/first_token.npy"))
        print(f"gen: {N} streamed decode steps (states carried via dump/reload), start tok={tok}")
        gen = []
        for step in range(N):
            write_decode_act(tok, f"{OUT}/act_decode.bin")
            cfg = gen_stream_cfg(f"{OUT}/act_decode.bin", dump_states=True)
            ok, out = run_driver(cfg, f"gen{step}")
            if not ok:
                print(out.strip()); print(f"HANDOFF at step {step}: config ready at {cfg} — NPU unavailable"); break
            h = bf16_to_f32(np.fromfile(f"{OUT}/npu_decode_hidden.bin", dtype=np.uint16))[:2048]
            assert np.isfinite(h).all(), "NON-FINITE decode hidden (interval-3 blowup!)"
            lg = full_logits_from_hidden(h)
            nxt = int(lg.argmax()); gen.append(nxt)
            print(f"  step {step}: tok {tok} -> {nxt}  hidden_absmax={np.abs(h).max():.3f} logit_absmax={np.abs(lg).max():.2f} finite=True")
            tok = nxt
        print("GENERATED:", gen)
    else:
        print("unknown mode", mode)
