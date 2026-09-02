"""Compare the 27B open-kernel decode step (position 0) with the CPU replica: full-vocab logits + per-layer residuals."""
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).parent / "w27"
ours = np.fromfile(W / "y_logits.bin", np.float32).astype(np.float64)
ref = np.fromfile(W / "ref_logits.bin", np.float32).astype(np.float64)
n = min(len(ours), len(ref)); ours, ref = ours[:n], ref[:n]
corr = float(np.corrcoef(ours, ref)[0, 1])
print(f"logits: corr {corr:.6f}  argmax ours {int(ours.argmax())} ref {int(ref.argmax())}  top5 ours {np.argsort(-ours)[:5].tolist()} ref {np.argsort(-ref)[:5].tolist()}")
l = 0
while (W / f"y_res{l}.bin").is_file() and (W / f"ref_res{l}.bin").is_file():
    g = np.fromfile(W / f"y_res{l}.bin", np.float32).astype(np.float64); r = np.fromfile(W / f"ref_res{l}.bin", np.float32).astype(np.float64)
    gi = np.fromfile(W / f"y_rout{l}.bin", np.float32)[256:264].view(np.int32).tolist() if (W / f"y_rout{l}.bin").is_file() else None
    print(f"layer {l:2}: residual corr {np.corrcoef(g, r)[0,1]:.6f} maxrel {np.abs(g-r).max()/np.abs(r).max():.2e}  routing {gi}")
    l += 1
ok = np.isfinite(ours).all() and corr > 0.999 and int(ours.argmax()) == int(ref.argmax())
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
