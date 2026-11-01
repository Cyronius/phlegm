"""Time trit_embed builds against each other: NPU dispatches back to back, two in flight, fixed inputs.

    python open_kernels/designs/trit_embed/bench.py path/to/trit-embed-3l build build_L3_cap128_d2_stub-all

Each round times every build once, in an order that alternates between rounds; the first round is
discarded. Per build it prints the median over rounds of milliseconds per dispatch (elapsed time over
dispatches, host input preparation excluded), the median per-round ratio to the first build, and the
largest difference of its final-layer states from the first build's on the same input (0.0 = identical).
Stub and variant builds are accepted. For F2_PROF_CYCLES builds (tag *_cycles<mask>) it also prints
each core's timer totals by kernel for one dispatch, as the median over the 32 cores (fused2.h slots).
Hold the NPU while it runs (`dibs run npu -- ...`).
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from trit_npu import ROOT, SLOT, TritNpu


def per_dispatch_ms(npu, dispatches):
    runs = [run for run, _ in npu.sets]
    start = time.perf_counter()
    for k in range(dispatches):
        npu.dll.tdn_start(npu.handle, runs[k % 2])
        if k:
            npu.dll.tdn_wait(npu.handle, runs[(k - 1) % 2])
    npu.dll.tdn_wait(npu.handle, runs[(dispatches - 1) % 2])
    return (time.perf_counter() - start) * 1000 / dispatches


SLOTS = {1: 'quant_x', 2: 'quant_q', 3: 'linear Q', 4: 'linear K', 5: 'linear V', 6: 'linear residual',
         7: 'linear GELU', 8: 'attend', 9: 'norm', 11: 'linear calls', 15: 'mask to last out'}


def cycles(npu, ids, pos, seg):
    """One standalone dispatch of a profiling build: median over cores of each timer slot."""
    npu.start(0, ids, pos, seg)
    npu.finish(0)
    run, (_, _, _, out) = npu.sets[0]
    layer = npu.buffers[out].nbytes // npu.layers
    prof = npu.buffers[out][(npu.layers - 1) * layer:].view(np.uint32).reshape(32, 2048)[:, 1536:1552]
    return {name: int(np.median(prof[:, slot])) for slot, name in SLOTS.items() if prof[:, slot].any()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model_dir', type=Path)
    p.add_argument('builds', nargs='+', help='build directories, relative to this folder or absolute')
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--dispatches', type=int, default=400)
    a = p.parse_args()
    torch.set_num_threads(2)
    engines = {}
    try:
        states = {}
        rng = np.random.default_rng(0)
        ids = torch.from_numpy(rng.integers(1000, 20000, SLOT))
        pos = torch.from_numpy(np.tile(np.arange(32), 4))
        seg = np.repeat(np.arange(1, 5, dtype=np.int32), 32)
        for name in a.builds:
            npu = TritNpu(a.model_dir, build=ROOT / name, profiling=True)
            engines[name] = npu
            for s in range(2):
                npu.start(s, ids, pos, seg)
                states[name] = npu.finish(s)
        ms = {name: [] for name in a.builds}
        for r in range(a.rounds + 1):
            order = a.builds if r % 2 == 0 else a.builds[::-1]
            for name in order:
                value = per_dispatch_ms(engines[name], a.dispatches)
                if r:
                    ms[name].append(value)
        profiles = {name: cycles(npu, ids, pos, seg) for name, npu in engines.items()
                    if 'cycles' in npu.manifest['tag']}
    finally:
        for npu in engines.values():
            npu.close()
    base = ms[a.builds[0]]
    report = {'load_average': os.getloadavg(), 'dispatches_per_round': a.dispatches, 'rounds': a.rounds,
              'builds': {name: {'ms_per_dispatch': statistics.median(v),
                                'ratio_to_first': statistics.median(x / b for x, b in zip(v, base)),
                                'max_state_difference': float((states[name] - states[a.builds[0]]).abs().max()),
                                'samples': [round(x, 4) for x in v]} for name, v in ms.items()}}
    report['cycles'] = profiles
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
