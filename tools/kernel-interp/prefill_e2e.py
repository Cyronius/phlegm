"""Phase 0 NPU prefill: decode-as-prefill (see docs/npu-prefill.md).

Feeds the prompt tokens one at a time through the resident serve loop with
lm_head SKIPPED (step logits args "- -"), starting from ZEROED states — the
decode kernel self-tracks its KV position on-device, so sequential decode is
mathematically exact prefill. The final prompt token runs lm_head, giving
the first generated token's logits for free. Then generates NGEN tokens.

Verification (5li3 target): after quit, the driver dumps the device states;
they are compared region-by-region against the CPU-prefilled state_L*.bin
(conv bf16 / S f32 for linear layers, k/v bf16 rows for full-attn), and the
first sampled token is compared with the CPU pipeline's first_token.npy.

Usage: python prefill_e2e.py [5li3|l40] [ngen=24]
"""
import os, sys, time, subprocess
import numpy as np
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32, f32_to_bf16

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = sys.argv[1] if len(sys.argv) > 1 else "5li3"
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 24

XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
ELF = "C:/caps/m0c/elf_000005.bin"
LM_ELF = "C:/caps/m0c/elf_000003.bin"
DRIVER = "C:/code/FastFlowLM/npu-engine/m0/out/decode_driver_nobarrier.exe"

if TARGET == "5li3":
    D = "C:/code/FastFlowLM/npu-engine/m3out/5li3"
    SCHED = "LLFLL"
else:
    D = "C:/code/FastFlowLM/npu-engine/m3out/l40"
    SCHED = "".join("F" if l % 4 == 3 else "L" for l in range(40))
NL = len(SCHED)

m = Q4NX(os.environ.get("BASE_MODEL", os.path.join(MODEL_DIR, "model.q4nx")))
NW = m.bf16("model.norm.weight")
ids = np.load(os.path.join(HERE, "prompt_token_ids.npy")).tolist()


def embed(tok):
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    return bf16_to_f32(np.frombuffer(m.mm[base + tok * 4096:base + (tok + 1) * 4096], dtype=np.uint16))


def write_act(tok):
    act = np.zeros(1048576, np.uint8)
    act[:4096] = f32_to_bf16(embed(tok)).view(np.uint8)
    act[4096:8192] = f32_to_bf16(NW).view(np.uint8)
    act.tofile(f"{D}/pf_act.bin")


def make_config():
    L = ["device", f"xclbin L {XB}/layer.xclbin", "context L2 L",
         f"xclbin LM {XB}/lm_head.xclbin",
         f"kernel kL L {ELF}", f"kernel kL2 L2 {ELF}", f"kernel klm LM {LM_ELF}"]
    for l in range(NL):
        L.append(f"buf pool{l} 536870912 {D}/pool_L{l}.bin")
        L.append(f"buf pack{l} 2097152 {D}/pack_L{l}.bin")
        L.append(f"buf side{l} 6291456 {D}/side_L{l}.bin")
        L.append(f"buf state{l} 3145728")          # ZERO states: prefill from scratch
    L.append(f"buf lmpool 542113792 {D}/pool_lmhead.bin")
    L.append("buf logits 1048576")
    L.append("buf act 1048576")
    L.append("servep")
    g = 0
    for c0 in range(0, NL, 3):
        ctx, kn = ("L", "kL") if g % 2 == 0 else ("L2", "kL2")
        L.append(f"runlist {ctx}")
        for l in range(c0, min(c0 + 3, NL)):
            L.append(f"layer {kn} pool{l} act pack{l} side{l} state{l}")
        L.append("submit")
        g += 1
    L.append("lmhead klm logits lmpool act")
    L.append("endserve")
    for l in range(NL):                            # runs after quit
        L.append(f"dump state{l} {D}/npu_pf_state_L{l}.bin 3145728")
    cfg = f"{D}/prefill_serve.txt"
    open(cfg, "w").write("\n".join(L) + "\n")
    return cfg


def step(proc, with_logits):
    lg = f"logits {D}/pf_logits.bin" if with_logits else "- -"
    proc.stdin.write(f"step {D}/pf_act.bin {D}/pf_hidden.bin {lg} -1\n")
    proc.stdin.flush()
    while True:
        ln = proc.stdout.readline()
        if not ln:
            raise RuntimeError("driver died mid-step")
        if "STEP OK" in ln:
            return
        if "STEP" in ln and ("ERR" in ln or "FAILED" in ln):
            raise RuntimeError("step failed: " + ln)


def read_logits():
    raw = np.fromfile(f"{D}/pf_logits.bin", dtype=np.uint16)[:248320]
    return bf16_to_f32(raw)


def corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_states(t_prompt):
    print("\n=== STATE VERIFICATION vs CPU prefill (per layer) ===")
    for l in range(NL):
        ref_p = f"{D}/state_L{l}.bin"
        npu_p = f"{D}/npu_pf_state_L{l}.bin"
        if not (os.path.exists(ref_p) and os.path.exists(npu_p)):
            print(f"L{l}: (missing {'ref' if not os.path.exists(ref_p) else 'npu dump'})")
            continue
        ref = np.fromfile(ref_p, dtype=np.uint8)
        npu = np.fromfile(npu_p, dtype=np.uint8)
        if (ref != 0).sum() == 0:
            print(f"L{l} ({SCHED[l]}): ref state is all zeros (pools-only build) — skip")
            continue
        if SCHED[l] == "L":
            conv_r = bf16_to_f32(ref[:49152].view(np.uint16)); conv_n = bf16_to_f32(npu[:49152].view(np.uint16))
            s_r = ref[49152:49152 + 32*128*128*4].view(np.float32); s_n = npu[49152:49152 + 32*128*128*4].view(np.float32)
            print(f"L{l} (L): conv corr {corr(conv_r, conv_n):.6f}  S corr {corr(s_r, s_n):.6f}")
        else:
            kr = ref[:1073152].reshape(-1, 1024); t_ref = int(kr.any(axis=1).sum())
            t = min(t_ref, t_prompt)
            if t_ref != t_prompt:
                print(f"L{l} (F): NOTE ref has {t_ref} kv rows vs prompt {t_prompt} — comparing first {t}")
            k_r = bf16_to_f32(ref[:t*1024].view(np.uint16));  k_n = bf16_to_f32(npu[:t*1024].view(np.uint16))
            v_r = bf16_to_f32(ref[1073152:1073152+t*1024].view(np.uint16))
            v_n = bf16_to_f32(npu[1073152:1073152+t*1024].view(np.uint16))
            print(f"L{l} (F): k corr {corr(k_r, k_n):.6f}  v corr {corr(v_r, v_n):.6f}")


def main():
    print(f"target {TARGET} ({NL} layers, schedule {SCHED if NL<=8 else SCHED[:8]+'...'}) prompt {len(ids)} tokens")
    cfg = make_config()
    proc = subprocess.Popen([DRIVER, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    while True:
        ln = proc.stdout.readline()
        if not ln:
            raise RuntimeError("driver died before READY")
        if "SERVE READY" in ln:
            break
    print("driver ready (pools resident, ZERO states)")

    t0 = time.time()
    for i, tok in enumerate(ids):
        write_act(int(tok))
        step(proc, with_logits=(i == len(ids) - 1))
    ttft = time.time() - t0
    lg = read_logits()
    first = int(np.argmax(np.nan_to_num(lg, nan=-1e30)))
    print(f"\nprefill: {len(ids)} tokens in {ttft:.2f}s ({ttft/len(ids)*1000:.0f} ms/tok)  -> first token {first}")
    ref_first_p = f"{D}/first_token.npy"
    if os.path.exists(ref_first_p):
        ref_first = int(np.load(ref_first_p))
        print(f"CPU-pipeline first token: {ref_first}  -> {'MATCH' if first == ref_first else 'MISMATCH'}")

    gen = [first]
    cur = first
    t1 = time.time()
    for _ in range(NGEN - 1):
        write_act(cur)
        step(proc, with_logits=True)
        cur = int(np.argmax(np.nan_to_num(read_logits(), nan=-1e30)))
        gen.append(cur)
    dt = time.time() - t1
    print(f"generated {len(gen)} tokens ({dt/max(1,NGEN-1)*1000:.0f} ms/tok): {gen}")
    try:
        from tokenizer import Qwen36Tokenizer
        tk = Qwen36Tokenizer()
        print("prompt:", repr(tk.decode(ids)))
        print("output:", repr(tk.decode(gen)))
    except Exception as e:
        print(f"(detok unavailable: {e})")

    proc.stdin.write("quit\n"); proc.stdin.flush()
    proc.wait()          # dumps run after quit
    compare_states(len(ids))


if __name__ == "__main__":
    main()
