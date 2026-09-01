"""Build the 40 NPU pools + packs + sides + lm_head pool for the base model,
with ZERO states -- for a DECODE-SPEED benchmark (tok/s is independent of the
prefill state values; only the real 512MB pools matter for memory traffic).
Skips the slow pure-Python CPU prefill entirely.

Usage: python pools_only_l40.py <base model.q4nx> <out_dir>
"""
import numpy as np, os, sys, gc
from q4nx import Q4NX, MODEL_DIR
import build_pools as B

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODEL_DIR, "model.q4nx")
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/code/FastFlowLM/npu-engine/m3out/l40"
os.makedirs(OUT, exist_ok=True)

m = Q4NX(MODEL)
nlayers = 0
while f"model.layer.{nlayers}.input_layernorm.weight" in m.tensors:
    nlayers += 1
sched = ["full_attention" if f"model.layer.{l}.self_attn.q_proj.weight" in m.tensors
         else "linear_attention" for l in range(nlayers)]
print(f"{os.path.basename(MODEL)} nlayers={nlayers} sched={''.join('F' if s=='full_attention' else 'L' for s in sched)}",
      flush=True)

zero_state = np.zeros(3145728, np.uint8)
for l in range(nlayers):
    full = sched[l] == "full_attention"
    if not os.path.exists(f"{OUT}/pool_L{l}.bin"):
        B.build_layer_pool(m, l, full).tofile(f"{OUT}/pool_L{l}.bin")
        B.build_pack(m, l).tofile(f"{OUT}/pack_L{l}.bin")
        B.build_side(m, l, full).tofile(f"{OUT}/side_L{l}.bin")
        zero_state.tofile(f"{OUT}/state_L{l}.bin")
        gc.collect()
    print(f"  pool L{l:2d} ({'F' if full else 'L'}) done", flush=True)
if not os.path.exists(f"{OUT}/pool_lmhead.bin"):
    B.build_lmhead_pool(m).tofile(f"{OUT}/pool_lmhead.bin")
# a plausible first token so the bench can start (value irrelevant to timing)
np.save(f"{OUT}/first_token.npy", np.int64(276))
print("DONE — pools-only build in", OUT, flush=True)
