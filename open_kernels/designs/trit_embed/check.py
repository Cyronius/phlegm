"""Check trit_npu against the model folder's CPU reference code, and time it.

    python open_kernels/designs/trit_embed/check.py path/to/trit-embed-3l [--texts lines.txt]

Without --texts it uses a fixed set of short and long English texts. It prints the minimum and mean
cosine between NPU and CPU vectors (NPU int8/bf16 arithmetic differs from the float reference in the
last bits, so expect about 0.998 minimum, not 1.0), then texts per second and the median dispatch time.
Run it while holding the NPU (`dibs run npu -- ...` on a shared machine).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_npu import TritNpu, load_model  # noqa: E402

TEXTS = [
    "A man is playing a guitar on the street.",
    "how do index funds work",
    "Index funds track a market index, so their holdings change only when the index does.",
    "The mitochondria produce most of the chemical energy needed to power the cell's biochemical reactions.",
    "Ternary weights take only the values minus one, zero and plus one, so a matrix product needs no multiplications.",
    "Who wrote the novel Pride and Prejudice?",
    "Jane Austen published Pride and Prejudice in 1813; it follows Elizabeth Bennet and her sisters. " * 4,
    "Rain is expected tomorrow afternoon.",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', type=Path)
    p.add_argument('--texts', type=Path, help='one text per line')
    p.add_argument('--repeat', type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(2)
    texts = [t for t in a.texts.read_text().splitlines() if t.strip()] if a.texts else TEXTS * 64
    npu = TritNpu(a.model_dir)
    try:
        vectors = npu.encode(texts)
        cpu = load_model(a.model_dir).encode(texts)
        cos = (vectors * cpu).sum(-1)
        seconds = []
        for _ in range(a.repeat):
            npu.dispatch_ms.clear()
            t = time.perf_counter()
            npu.encode(texts)
            seconds.append(time.perf_counter() - t)
        report = {'texts': len(texts), 'cosine_to_cpu': {'min': float(cos.min()), 'mean': float(cos.mean())},
                  'texts_per_second': len(texts) / statistics.median(seconds),
                  # the C++ host loop (native, the default) records no per-dispatch times
                  'median_dispatch_ms': statistics.median(npu.dispatch_ms) if npu.dispatch_ms else None,
                  'dispatches': len(npu.dispatch_ms),
                  'build': npu.manifest['artifact_sha256']}
    finally:
        npu.close()
    print(json.dumps(report, indent=2))
    if not np.isfinite(vectors).all() or cos.min() < 0.995:
        sys.exit('FAIL: NPU vectors differ from the CPU reference')


if __name__ == '__main__':
    main()
