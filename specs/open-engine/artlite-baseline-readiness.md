# ArtLite baseline readiness: 2026-10-24

Development inventory and setup checkpoints. The initial inventory below is
historical; the continuation records successful builds and an ArtLite NPU
operation. Full-model acceptance and parity measurements remain open.

## Source identities

- Phlegm HEAD: `e028a916b1fdd2f32c4620a01c2153b5c5e3f771`, dirty working tree.
  Existing CLI/server, tokenizer and bridge edits were preserved.
- External `/home/cyrus/code/openflowlm-next`:
  `478dbea3f00f868ac58205803cc537b402c6cad5`, clean. This is the issue212
  companion patch base, not a reconciled combined runtime. Its `manifest.cpp`
  rejects versions above 3; Phlegm's DIT contract uses version 7.
- `/home/cyrus/code/artlite`: `a07a7b8893cb4a835352669d237609ba017e8ab1`, clean.
- `/home/cyrus/code/cyronius-artlite`: `1de4b83e01c4daa66f4e47969920352427fc1da1`.
  Comparator selection and build provenance remain unestablished.
- Recommendations checkout: `50d8333ae1c9257ad8eba466e1932ea7b4942b06`.
  Read R-0001/3/4/6/7/8/11/12/13 for this inventory; no optimization proposed.

## Observed host and blockers

| Item | Observation |
| --- | --- |
| CPU/device | Ryzen AI MAX+ 395; Strix Halo NPU `1022:17f0`, revision 11, PCI `0000:be:00.1` |
| OS/kernel | Fedora 44, `7.2.11-200.fc44.x86_64` |
| Memory/storage | 123 GiB RAM, 112 GiB available; 1.8 TiB available on repository filesystem at inventory time |
| Driver | In-tree `amdxdna` module loaded, but NPU has no driver binding and `/dev/accel` is absent |
| Boot failure | Kernel journal: `aie2_init: Running without IOMMU not supported`, hardware init `-22`, probe failed `-22` |
| Firmware files | `17f0_11/npu_7.sbin.xz` points to `npu.sbin.1.1.2.65.xz`; legacy `npu.sbin.xz` points to `1.0.0.166`. Loaded firmware version cannot be established after failed probe |
| Memlock | Shell limit 8192 KiB; runtime validation still required |
| Build tools | `xrt-smi`, `aie-opt`, `aiecc.py`, `clang++`, `g++`, `cmake`, `ninja`, `cargo` absent from PATH |
| Python checks | Python 3 available; `pytest`, `numpy`, `ml_dtypes` unavailable |
| XRT | No matching XRT RPM reported by package query; no coreutil/xdna libraries found in scanned `/opt` or `/usr/local` |
| Selected models | No Llama 1B or Qwen3 0.6B weights found in scanned model/cache locations |

The model scan covered `~/.cache/huggingface`, `~/.flm`, `~/models`,
`~/code/model-experiments`, `/opt` and `/usr/local`. Other GGUF files exist
(Qwen3.8-27B, Bonsai 2 4B and a titler model); their compatibility was not
validated. The external runtime contains shipped xclbins, which are not fresh
combined-tree builds. Absence claims above are scoped to these searches.

## Reproduction commands and results

Run from the repository, with external paths adjusted if moved:

```bash
git status --short
git diff --stat
git -C /home/cyrus/code/openflowlm-next status --short
git -C /home/cyrus/code/openflowlm-next rev-parse HEAD
uname -a
lspci -nn
df -h .
free -h
ls /dev/accel
ls /sys/bus/pci/devices/0000:be:00.1/driver
lsmod | rg 'amdxdna|accel'
modinfo amdxdna
journalctl -k -b --no-pager --grep='amdxdna|amdnpu|npu_7' -n 50
ls -l /usr/lib/firmware/amdnpu/17f0_11
cat /proc/cmdline
ulimit -l
command -v xrt-smi aie-opt aiecc.py clang++ g++ cmake ninja cargo pytest
rpm -qa | rg -i 'xrt|xdna|ryzen|aie'
python3 -m pytest --version
rg -n 'm.version' /home/cyrus/code/openflowlm-next/src/open_qwen36/manifest.cpp
```

Missing device/binding probes failed with ENOENT; pytest failed with
`No module named pytest`. These are unavailable coverage, not passing checks.
No driver settings, firmware, power modes, external source files or weights
were changed. Power mode, other NPU clients and runtime firmware remain unknown.

## Recovery order

1. Enable IOMMU in platform firmware if disabled, reboot, and recheck the kernel
   journal, driver binding and `/dev/accel`. The journal establishes the failure;
   it does not establish which platform setting caused IOMMU to be unavailable.
2. Provision a compatible Linux XRT plus XDNA plugin, build tools and Python
   dependencies. Verify device enumeration, loaded firmware and memlock before
   dispatch. Installing firmware alone does not address the observed failure.
3. Reconcile companion patches in an isolated writable runtime checkout using
   `open_kernels/upstream-patches/sources.json`; preserve version 7 and the Llama
   high/low probability contract. Existing adapter build instructions are for
   Windows; a Linux build path must also be established for this host.
4. Obtain the selected model/tokenizer/config artifacts; build all streams of
   each selected kernel image afresh and record hashes and routes.
5. Run recipe/manifest, independent numerical, reset and serving gates before
   the matched, alternating ArtLite timing matrix. Retain all existing quality
   failures until independently resolved.

## Continuation: local Linux development stack (2026-10-24)

The current boot has IOMMU enabled and `amdxdna_accel_driver 0.10.0`
initialized on minor 0. Outside the sandbox, `/dev/accel/accel0` exists as
`root:render`, mode 0666. The sandbox hides that node; hardware commands require
execution outside it. No kernel module, firmware or boot settings were changed.

System installation via `sudo -n dnf install ...` failed with `a password is
required`. User authorization does not supply OS credentials. Instead, installed
an isolated development stack under ignored `target/npu-dev/`:

- Source `target/npu-dev/env.sh` for GCC 15.3, CMake 4.4.3, Ninja, Rust 1.99,
  Python 3.12.15, NumPy, SciPy, pytest, build headers and libraries.
- MLIR-AIE `1.4.2`, Peano `21.0.0.2026080301+c9c5ecb7` from the release's
  `utils/peano-requirements.txt`. ROT13 compiled successfully.
- XDNA `0c9c70e54a512027975639942a6c871d5988cab4`, pinned XRT
  `a29919c035c05d0a368ef6ce09d372976618dee1` (2.27.0). Built userspace only
  (`SKIP_KMOD=ON`), with required submodules including VTD and AIEBU. Built
  `xrt-smi`, `xclbinutil`, `aiebu-asm`, core libraries and the XDNA plugin.
  Local libraries/tools in `target/npu-dev/xrt` have rpaths adjusted with
  patchelf; the installed driver and firmware were preserved. Python XRT
  bindings have not been validated.
- ArtLite isolated clone at `a07a7b8893cb4a835352669d237609ba017e8ab1`,
  installed in `target/npu-dev/artlite-env` with its own pinned MLIR-AIE
  `1.4.5.dev56+g1a647f1`, Peano `22.0.0.2026102001+4f5df13d` and CPU Torch.
  Its direct-driver `test_kmath.py -k expneg` passes on the NPU (one test).
  This is arithmetic smoke coverage, not model-quality or performance evidence.

Reproduction inventories: `target/npu-dev/conda-explicit.txt`,
`python-freeze.txt`, `peano-requirements.txt`, and `env.sh`. These local files
are not committed. The upstream userspace-only build guidance is in
[MLIR-AIE's non-Ubuntu guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLinNonUbuntu.md);
matching XRT/plugin source guidance is in
[AMD's README](https://github.com/amd/xdna-driver/blob/main/README.md).

### Runtime and selected model

Isolated `target/npu-dev/reference-runtime` starts at OFLM
`478dbea3f00f868ac58205803cc537b402c6cad5`. Reconciled runtime portions of
issue212 and PRs 223/193/194/213/207 there using three-way application and manual
conflict resolution. Retained shared BO views and Llama high/low probabilities;
extended their offsets across PR193's attention sub-blocks. Corrected the
low-query row stride from the old T to sub-block S. Synced the canonical stream
patch helper after removing its duplicate definitions in Phlegm.

This is a **development baseline runtime**, not a fully reconciled combined
runtime: it accepts manifest versions 1–3 and local DIT version 7, explicitly
refusing 4/5/6. BFP16/GLU/prep and Bonsai/Hadamard integration remain open and
must not be benchmarked through this checkout. The selected Llama set and its
GGUF twin both use version 1. DIT arithmetic and merged attention have not yet
passed hardware/model-quality gates.

The development source diff is `target/npu-dev/baseline-runtime.patch`, SHA-256
`4b5e57d0f92ec27f8b74e4132d477a006a7bec8ca93eef22fe43c1ad99600313`.
The original external runtime checkout remains unchanged. Do not interpret the
retained companion patches as a sequential series or these builds as acceptance.

Built the bridge with `benchmarks/reference/build.py`. The default output is
`target/reference-bench/phlegm-reference-bench` plus its Rust-verified sidecar.
Complete Llama 1B kernel export uses `--max-ctx 16384`, independent build root
and all streams; compiler program-memory checks completed. No new shape tuning.

Model: `QuantFactory/Llama-3.2-1B-Instruct-GGUF`, revision
`b3dd7c2b74b8666a5c9d365775bdeea3377ad0db`,
`Llama-3.2-1B-Instruct.Q4_1.gguf` (831745792 bytes), SHA-256
`6e35ca7e357c6c7f55a34d683e538614cb3e34b1155d92731da75c1685136add`.
Tokenizer/config: `unsloth/Llama-3.2-1B-Instruct`, revision
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`.
All are in `target/npu-dev/models/llama32-1b`; matching identity and tokenizer
behavior still need full-model validation. The GGUF is available to both engines.

### Commands and verification

Run from Phlegm after sourcing `target/npu-dev/env.sh`:

```bash
python benchmarks/reference/build.py --reference-root target/npu-dev/reference-runtime --xrt-include target/npu-dev/xrt/include --xrt-lib target/npu-dev/xrt/lib/libxrt_coreutil.so.2
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/llama32-1b.json --out target/npu-dev/kernels/llama32-1b --build-root target/npu-dev/kernel-builds/llama32-1b --max-ctx 16384 --force -j 2
python -m pytest -q --tb=short
PHLEGM_REFERENCE_BRIDGE="$PWD/target/reference-bench/phlegm-reference-bench" cargo test -p phlegm-core compiled_bridge_handshake_and_provenance -- --ignored
cargo test -p phlegm-core
cmake --build target/npu-dev/harness-build
ctest --test-dir target/npu-dev/reference-tests -R 'OPEN-MANIFEST|GGUF-PACK|BLOCK' --output-on-failure
python benchmarks/reference/smoke.py --bridge target/reference-bench/phlegm-reference-bench --model-dir target/npu-dev/models/llama32-1b --kernels target/npu-dev/kernels/llama32-1b
git diff --check
```

Results:

- Python: **1809 passed, 64 skipped, 8 deselected**. Initial run exposed
  missing SciPy plus seven native cases requiring absent in-tree OFLM sources.
  Installed SciPy and made those prerequisites explicit in `conftest.py`;
  skips remain untested coverage.
- Rust compiled bridge provenance/handshake: **1 passed**. Core suite outside
  sandbox: **33 passed, 2 ignored**, release provenance integration **1 passed**.
  Sandbox run had eight HTTP bind failures (`EPERM`); host rerun passes.
- Reference manifest/GGUF CTests: **2 passed** against the isolated checkout's
  fixtures. `block_host_test` compiled but requires generated fixture inputs;
  running it without those only prints usage and is **not a pass**.
- Native `run_kernel` build exposed duplicate `AttnRowPatch`, `attn_rows_table`
  and `attn_rows_apply` definitions in the imported header. Removed only the
  older duplicate; retained KV-slice support. Harness now builds. Added a
  compiler-backed host regression for row addresses, flags and slice lengths;
  it passes and is included in the Python total. Focused related checks:
  **77 passed, 8 skipped**, plus the new regression **1 passed**.
- Llama model smoke and `xrt-smi examine`: **blocked**, not passes. Both fail
  at `mmap(... len=67108864, flags=8209 ...)`, error `-11`, with soft/hard
  memlock both 8192 KiB. No Phlegm model inference or performance data obtained.

Raw local logs and artifact hashes are indexed by `target/npu-dev/evidence.json`
(SHA-256 `6bef8a88bd1e00bb0052b634e8db52e46cd17279455f0b9140c281ef988cb7de`).
Kernel `toolchain.json` retains per-stream hashes and build settings. Generated
scratch directories were moved under ignored `target/npu-dev/export-scratch`.

### Remaining privileged step and next action

Raise the execution service's memlock hard and soft limits. During this session
its PID was **5763**; an authenticated local terminal can run:

```bash
sudo prlimit --pid 5763 --memlock=unlimited:unlimited
```

Verify that PID still identifies the current Codex execution service before
reusing it in another session. New child commands inherit the updated limit.
For persistence across logins, configure an appropriate memlock policy and
restart the application session; that persistent system configuration has not
been changed. No credentials were requested or stored.

Then rerun XRT enumeration and model smoke, resolve runtime/manifest/quality
failures, and establish reset/serving gates before matched ArtLite trials.
Qwen3 0.6B weights/kernels and all performance matrix cells remain open.
