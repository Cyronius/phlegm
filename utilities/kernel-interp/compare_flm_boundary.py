"""Compare decode-as-prefill device states against FLM's TRUE prefill boundary.

Oracle: C:/caps/pf_t11_full — FLM serve of the base 40L model with the known
11-token prompt "Say hi." (see boundary_manifest.json: 30 GDN state dumps
taken at the prefill->decode roundtrip, first decode input verified to be
embed(<think>), prefill-final logits = odd vocab half, semantically sane).

Ours: npu-engine/m3out/l40/npu_pf_state_L{l}.bin from prefill_e2e.py l40 1
(sequential decode-as-prefill from ZEROED states, same 11 tokens).

Layer mapping: FLM's boundary dumps are in zero-init order k=0..29 over the
GDN (linear) layers only; model layer l = k + k//3 (skip every 4th, the
full-attn layers). State layout (both sides, byte-verified in decode_step.py):
conv bf16 [3,8192] at [0:49152], GDN S fp32 [32,128,128] at [49152:+2MB].

Usage: python compare_flm_boundary.py
"""
import json
import numpy as np
from q4nx import bf16_to_f32

CAP = "C:/caps/pf_t11_full"
D = "C:/code/FastFlowLM/npu-engine/m3out/l40"


def corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    man = json.load(open(f"{CAP}/boundary_manifest.json"))
    rows = []
    for st in man["boundary_state_syncs"]:
        k = st["init_order"]
        l = k + k // 3                      # GDN index -> model layer
        flm = np.fromfile(f"{CAP}/{st['sync']}.bo", dtype=np.uint8)
        try:
            ours = np.fromfile(f"{D}/npu_pf_state_L{l}.bin", dtype=np.uint8)
        except FileNotFoundError:
            print(f"k={k:2d} L{l:2d}: our dump missing"); continue
        conv_c = corr(bf16_to_f32(flm[:49152].view(np.uint16)),
                      bf16_to_f32(ours[:49152].view(np.uint16)))
        s_c = corr(flm[49152:49152 + 2097152].view(np.float32),
                   ours[49152:49152 + 2097152].view(np.float32))
        rows.append((l, conv_c, s_c))
        print(f"k={k:2d} L{l:2d}: conv corr {conv_c:.6f}   S corr {s_c:.6f}")
    if rows:
        cc = [r[1] for r in rows]; sc = [r[2] for r in rows]
        print(f"\nmedian conv corr {np.median(cc):.6f}  min {min(cc):.6f}")
        print(f"median S    corr {np.median(sc):.6f}  min {min(sc):.6f}")

    # prefill-final logits: FLM dump is fp32 odd-half (index i -> vocab 2i+1);
    # our pf_logits.bin is bf16 in the same odd-half kernel layout.
    try:
        flm_lg = np.fromfile(f"{CAP}/{man['prefill_final_logits_sync']}.bo",
                             dtype=np.float32)[:124160]
        our_lg = bf16_to_f32(np.fromfile(f"{D}/pf_logits.bin",
                                         dtype=np.uint16)[:124160])
        print(f"\nlogits (odd half): corr {corr(flm_lg, our_lg):.6f}  "
              f"argmax FLM {int(flm_lg.argmax())} vs ours {int(our_lg.argmax())} "
              f"(vocab {2*int(flm_lg.argmax())+1} vs {2*int(our_lg.argmax())+1})")
    except FileNotFoundError as e:
        print("logits compare skipped:", e)


if __name__ == "__main__":
    main()
