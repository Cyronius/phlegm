"""Run trit-embed-3l on the XDNA2 NPU: the host side of the trit_embed encoder graph.

    from trit_npu import TritNpu
    npu = TritNpu("path/to/trit-embed-3l")   # the model folder: config.json, model.safetensors, tokenizer, trit_embed.py
    vectors = npu.encode(["how do index funds work", "Index funds track a market index."])  # (2, 384), unit norm
    npu.close()

The NPU runs every encoder layer in one dispatch per two 128-token slots (eight tokens a core), and
short texts share a slot (best-fit packing; per-token segment IDs keep their attention apart). At load the host repacks the model
folder's 2-bit codes and bf16 row scales into the NPU's packets (weight_stream). Per dispatch it tokenizes,
does the int8 embedding lookup and input LayerNorm, mean-pools, and applies the output projection, all with
the model folder's own reference code, and keeps two buffer sets in flight so it prepares one dispatch while
the NPU runs the previous one. An instance owns mutable device buffers: use it from one thread.

In a phlegm checkout, build the graph (build.py) and the XRT host library (build_host.py) first. A model
folder can instead ship them prebuilt in npu/ (bundle.py), next to a copy of this file: then the runtime
needs only Linux, an XDNA2 NPU on the amdxdna driver, XRT and the Python packages of trit_embed.py.
check_npu() names a missing prerequisite (NpuUnavailable) before anything touches the device; run
`python trit_npu.py MODEL_DIR` to check them and compare a few vectors with the CPU reference.
"""
import copy
import ctypes as ct
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import queue
import resource
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Not resolve(): a Hugging Face snapshot links each file to a blob of its own, so the folder is the link's.
ROOT = Path(os.path.abspath(__file__)).parent
# A phlegm checkout builds into build/; a model folder ships the prebuilt files in npu/ (bundle.py).
BUILD = ROOT / 'build' if (ROOT / 'trit_design.py').exists() else ROOT / 'npu'
LIBRARY_NAME = 'libtd_npu2.so'


def default_library(build):
    """TRIT_NPU_LIBRARY, else the host library of a build: <build>/libtd_npu2.so (bundle) or <build>/host/ (phlegm);
    variant builds (build_<tag>/, bench.py) have none of their own and share build/host/'s."""
    if 'TRIT_NPU_LIBRARY' in os.environ:
        return Path(os.environ['TRIT_NPU_LIBRARY'])
    for path in (Path(build) / LIBRARY_NAME, Path(build) / 'host' / LIBRARY_NAME):
        if path.exists():
            return path
    return ROOT / 'build/host' / LIBRARY_NAME


LIBRARY = default_library(BUILD)
SLOT = 128
SLOTS = 2               # slots a dispatch
TOKENS = SLOTS * SLOT
OUT = 6912              # a core's output packet (trit_design.OSZ)
SETS = int(os.environ.get('TRIT_NPU_SETS', '3'))   # buffer sets in flight
PACKETS = 170
ELEMENT = 8192   # broadcast element bytes
CUT = 1000       # characters: tokenize cuts longer texts first (exact; see TritNpu.tokenize)
CHUNK = 128      # texts that encode tokenizes before the first dispatch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NpuUnavailable(RuntimeError):
    """A runtime prerequisite of the NPU path is missing; the message names it and the fix."""


XDNA2 = {'0x17f0'}          # PCI device IDs (vendor 0x1022) of XDNA2 NPUs; 0x1502 is XDNA1 (Phoenix, Hawk Point)
# Locked memory: XRT maps a 64 MiB device heap when it opens the device. Measured 2026-11-04 (XRT 2.27.0): 48 MiB
# fails (mmap EAGAIN), 64 MiB works. check_npu asks for MEMLOCK_WANT and fails below MEMLOCK_NEED.
MEMLOCK_NEED = 64 << 20
MEMLOCK_WANT = 128 << 20
XRT_DIRS = ('lib', 'lib64')


def find_npu():
    """(accel node, PCI device ID) of the first device on the amdxdna driver."""
    seen = []
    for dev in sorted(glob.glob('/sys/class/accel/accel*')):
        driver = os.path.basename(os.path.realpath(os.path.join(dev, 'device/driver')))
        if driver != 'amdxdna':
            continue
        with open(os.path.join(dev, 'device/device')) as f:
            pci = f.read().strip()
        seen.append(('/dev/accel/' + os.path.basename(dev), pci))
    if not seen:
        if not os.path.isdir('/sys/module/amdxdna'):
            raise NpuUnavailable('The amdxdna kernel driver is not loaded (no /sys/module/amdxdna). It is in mainline '
                                 'Linux from 6.14 (CONFIG_DRM_ACCEL_AMDXDNA); try `sudo modprobe amdxdna`, or install '
                                 "AMD's xdna-driver for older kernels.")
        raise NpuUnavailable('amdxdna is loaded but no NPU is bound to it (no /sys/class/accel/accel* with driver '
                             'amdxdna). Check `dmesg | grep -i amdxdna` for firmware or probe errors.')
    for node, pci in seen:
        if pci in XDNA2:
            if not os.access(node, os.R_OK | os.W_OK):
                raise NpuUnavailable(f'No read/write access to {node}; add your user to its group '
                                     f'(`ls -l {node}`, usually render) and log in again.')
            return node, pci
    raise NpuUnavailable(f'The NPU (PCI 1022:{seen[0][1][2:]}) is not XDNA2: these kernels need an XDNA2 NPU '
                         '(PCI 1022:17f0: Ryzen AI 300 / Strix Point, Strix Halo, Krackan Point); 1022:1502 is the '
                         'XDNA1 NPU of Phoenix and Hawk Point.')


def raise_memlock(need=MEMLOCK_WANT):
    """Raise the soft RLIMIT_MEMLOCK towards need (up to the hard limit); returns the soft limit in force.
    XRT pins its command and buffer memory when it opens the device; with the 8 MiB that systemd gives
    desktop sessions by default, opening the device fails."""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    inf = resource.RLIM_INFINITY
    if soft != inf and soft < need:
        target = need if hard == inf else min(need, hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (target, hard))
            soft = target
    return soft


def memlock_hint(soft):
    if soft == resource.RLIM_INFINITY:
        return ''
    return (f' The locked-memory limit is {soft >> 20} MiB (`ulimit -l`); XRT needs {MEMLOCK_NEED >> 20} MiB. Raise it, for example with '
            '`* - memlock unlimited` in /etc/security/limits.d/99-npu.conf (and DefaultLimitMEMLOCK=infinity in '
            '/etc/systemd/user.conf and system.conf for desktop sessions), then log in again; or run one command as '
            '`systemd-run --user --pipe --wait -p LimitMEMLOCK=infinity python ...`.')


def load_xrt():
    """Load XRT's libxrt_coreutil.so.2 globally, so the host library binds to it; returns (handle, path, version).
    Search order: $XILINX_XRT/lib[64], /opt/xilinx/xrt/lib[64] (AMD's packages), then the loader's default path."""
    roots = [os.environ['XILINX_XRT']] if os.environ.get('XILINX_XRT') else []
    candidates = [os.path.join(r, d, 'libxrt_coreutil.so.2') for r in roots + ['/opt/xilinx/xrt'] for d in XRT_DIRS]
    errors = []
    for path in [c for c in candidates if os.path.exists(c)] + ['libxrt_coreutil.so.2']:
        try:
            lib = ct.CDLL(path, mode=ct.RTLD_GLOBAL)
        except OSError as e:
            errors.append(str(e))
            continue
        real = loaded_path('libxrt_coreutil.so') or path
        version = os.path.basename(os.path.realpath(real)).partition('.so.')[2] or 'unknown'
        return lib, os.path.realpath(real), version
    raise NpuUnavailable('XRT is not installed (cannot load libxrt_coreutil.so.2: ' + '; '.join(errors[-2:]) + '). '
                         'Install XRT 2.27 or newer with its amdxdna plugin (Fedora: xrt-npu and xrt-plugin-amdxdna; '
                         "otherwise AMD's xdna-driver packages, under /opt/xilinx/xrt), or set XILINX_XRT.")


def loaded_path(name):
    """Path of a loaded shared object whose file name starts with name, from /proc/self/maps."""
    try:
        with open('/proc/self/maps') as f:
            for line in f:
                path = line.rstrip('\n').split(maxsplit=5)[-1]
                if os.path.basename(path).startswith(name):
                    return path
    except OSError:
        pass
    return None


def check_npu(library=None):
    """Check the runtime prerequisites without opening the device; returns what was found. Raises
    NpuUnavailable naming the first one missing: Linux on x86-64, an XDNA2 NPU on amdxdna, device access,
    XRT, the locked-memory limit (raised here when the hard limit allows), and the host library loading."""
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise NpuUnavailable(f'The prebuilt NPU host library is for Linux on x86-64; this is {sys.platform} '
                             f'{platform.machine()}. Use TritEmbed (CPU/GPU) here.')
    node, pci = find_npu()
    _, xrt_path, xrt_version = load_xrt()
    plugin = glob.glob(os.path.join(os.path.dirname(xrt_path), 'libxrt_driver_xdna.so*'))
    memlock = raise_memlock()
    info = {'device': node, 'pci_id': '1022:' + pci[2:], 'xrt': xrt_path, 'xrt_version': xrt_version,
            'xdna_plugin': plugin[0] if plugin else None,
            'memlock_mib': 'unlimited' if memlock == resource.RLIM_INFINITY else memlock >> 20}
    if memlock != resource.RLIM_INFINITY and memlock < MEMLOCK_NEED:
        raise NpuUnavailable('Too little lockable memory for XRT.' + memlock_hint(memlock))
    if library is not None:
        load_runtime(library, info)
    return info


def load_runtime(library=LIBRARY, info=None):
    library = Path(library)
    if not library.exists():
        raise NpuUnavailable(f'No NPU host library at {library}: in phlegm, run build_host.py; a model folder ships '
                             f'it as npu/{LIBRARY_NAME}.')
    try:
        dll = ct.CDLL(str(library))
    except OSError as e:
        message = str(e)
        if 'GLIBC' in message or 'CXXABI' in message:
            hint = ('The prebuilt library needs glibc 2.32 and libstdc++ from GCC 12 (GLIBCXX_3.4.30) or newer; a '
                    'Python environment that loads an older libstdc++ first (some conda environments) hides the '
                    "system's. Rebuild it with phlegm's bundle.py, or use a newer libstdc++.")
        elif 'undefined symbol' in message:
            hint = (f'Your XRT ({(info or {}).get("xrt_version", "unknown version")}) lacks an XRT symbol the library '
                    'was linked against (XRT 2.27.0); update XRT, or rebuild the library against yours.')
        else:
            hint = 'Check that XRT is installed (check_npu()).'
        raise NpuUnavailable(f'Cannot load {library}: {message}. {hint}') from e
    dll.tdn_open.argtypes, dll.tdn_open.restype = [ct.c_char_p, ct.c_char_p], ct.c_void_p
    dll.tdn_buf.argtypes, dll.tdn_buf.restype = [ct.c_void_p, ct.c_size_t], ct.c_void_p
    dll.tdn_sync.argtypes, dll.tdn_sync.restype = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t], None
    dll.tdn_bind.argtypes, dll.tdn_bind.restype = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_int)], ct.c_int
    dll.tdn_start.argtypes, dll.tdn_start.restype = [ct.c_void_p, ct.c_int], ct.c_int
    dll.tdn_wait.argtypes, dll.tdn_wait.restype = [ct.c_void_p, ct.c_int], ct.c_double
    dll.tdn_close.argtypes, dll.tdn_close.restype = [ct.c_void_p], None
    dll.tdn_last_error.argtypes, dll.tdn_last_error.restype = [], ct.c_char_p
    vp, ip = ct.c_void_p, ct.POINTER(ct.c_int)
    dll.tdn_host_init.argtypes = [vp, vp, vp, ct.c_int, vp, vp, ct.c_int, vp, vp, vp, ct.c_float, vp,
                                  ct.c_int, ip, ip, ip, ip, ct.c_int]
    dll.tdn_host_init.restype = ct.c_int
    dll.tdn_encode.argtypes = [vp, ct.c_int, vp, vp, ct.c_int, vp, ip]
    dll.tdn_encode.restype = ct.c_int

    def check(result, function, arguments):
        error = dll.tdn_last_error()
        if error:
            raise RuntimeError(f'{function.__name__}: {error.decode(errors="replace")}')
        return result
    for name in ('tdn_open', 'tdn_buf', 'tdn_sync', 'tdn_bind', 'tdn_start', 'tdn_wait', 'tdn_host_init', 'tdn_encode'):
        getattr(dll, name).errcheck = check
    return dll


def pack_weights(codes: np.ndarray) -> np.ndarray:
    """Ternary codes [N, K] -> [N/32, K/16, 128] bytes: 2-bit fields, four output channels per byte
    (packet formats 2 and 3, fused2.h): each block's channels 0-15 store code + 1, channels 16-31 code & 3."""
    codes = np.asarray(codes)
    n, k = codes.shape
    if n % 32 or k % 16 or not np.isin(codes, [-1, 0, 1]).all():
        raise ValueError('Weights require N%32=K%16=0 and codes in {-1,0,1}')
    c = (codes.astype(np.int16) & 3).astype(np.uint8)
    lo = (codes.astype(np.int16) + 1).astype(np.uint8)
    kb, lane = np.arange(k // 16)[:, None], np.arange(128)[None, :]
    ki, ni = kb * 16 + lane // 8, 2 * (lane % 8)
    blocks = []
    for start in range(0, n, 32):
        w, wl = c[start:start + 32], lo[start:start + 32]
        blocks.append(wl[ni, ki] | (w[ni + 16, ki] << 2) | (wl[ni + 1, ki] << 4) | (w[ni + 17, ki] << 6))
    return np.stack(blocks)


def bf16_bits(v):
    """bf16 bit patterns of fp32 values, rounded to nearest even."""
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def parameter_packet(weight, bias):
    """A LayerNorm packet: gamma as bf16 hi/lo halves (gamma = hi + lo to 16 bits; [384] each at bytes
    0 and 768, fused2.h norm_out), beta fp32 at byte 2048."""
    g = np.asarray(weight, np.float32).reshape(-1)
    gh = bf16_bits(g)
    gl = bf16_bits(g - (gh.astype(np.uint32) << 16).view(np.float32))
    p = np.zeros(1024, np.float32)
    p[512:512 + len(bias)] = np.asarray(bias, np.float32).reshape(-1)
    u = p.view(np.uint16)
    u[:len(g)], u[384:384 + len(g)] = gh, gl
    return p.view(np.uint8)


def weight_stream(model):
    """Per layer, 170 packets of 4096 bytes in consumption order, broadcast as 86 elements of 8192
    bytes: consecutive pairs, except that each LayerNorm packet has an element of its own. A weight
    packet holds one 32-output-channel block of packed ternary codes, then at byte 3072 the bias (fp32)
    and weight scale (bf16: scale for channels 0-15, scale/4 for 16-31) of its two 16-channel halves,
    each repeated for the 4 token rows (packet format 3, fused2.h)."""
    packets = []

    def linear(codes, scale, bias):
        assert len(scale) == len(bias) == codes.shape[0]
        for n, block in enumerate(pack_weights(codes)):
            p = np.zeros(4096, np.uint8)
            p[:block.size] = block.reshape(-1)
            b, s = bias[n * 32:(n + 1) * 32], scale[n * 32:(n + 1) * 32]
            s = np.concatenate([s[:16], s[16:] / 4]).astype(np.float32)          # bf16-exact
            rep = lambda v: np.tile(v.reshape(2, 1, 16), (1, 4, 1)).reshape(-1)  # [half][row][16]
            p[3072:3584] = rep(np.asarray(b, np.float32)).view(np.uint8)
            p[3584:3840] = rep((s.view(np.uint32) >> 16).astype(np.uint16)).view(np.uint8)
            packets.append(p)

    for layer in model.layers:
        start = len(packets)
        for name in ('k', 'v', 'q', 'o'):          # K and V first: their exchange overlaps Q (f2_step)
            m = getattr(layer, name)
            linear(m.codes.numpy(), m.scale.float().numpy(), m.bias.detach().numpy())
        packets.append(parameter_packet(layer.ln1.weight.detach().numpy(), layer.ln1.bias.detach().numpy()))
        uc, us = layer.fc1.codes.numpy(), layer.fc1.scale.float().numpy()
        dc, ds = layer.fc2.codes.numpy(), layer.fc2.scale.float().numpy()
        for g in range(6):
            linear(uc[g * 256:(g + 1) * 256], us[g * 256:(g + 1) * 256],
                   layer.fc1.bias.detach().numpy()[g * 256:(g + 1) * 256])
            bias = layer.fc2.bias.detach().numpy() if g == 0 else np.zeros(384, np.float32)
            linear(np.ascontiguousarray(dc[:, g * 256:(g + 1) * 256]), ds, bias)
        packets.append(parameter_packet(layer.ln2.weight.detach().numpy(), layer.ln2.bias.detach().numpy()))
        assert len(packets) - start == PACKETS
    pad, out = np.zeros(4096, np.uint8), []
    for layer in range(len(model.layers)):
        p = packets[layer * PACKETS:(layer + 1) * PACKETS]
        ln1 = 48                            # K, V, Q and O: 4 x 12 packets, then the first LayerNorm
        out += p[:ln1] + [p[ln1], pad] + p[ln1 + 1:PACKETS - 1] + [p[PACKETS - 1], pad]
    return np.stack(out).reshape(-1)


def pack(lengths, align=1):
    """Best-fit decreasing: assign each text to a 128-token slot; returns ((slot, offset) per text, slots).

    With align=8 each text starts on an 8-token boundary, so no 8-key block-floating
    group mixes two texts. Measured on the KJV, this did not change accuracy."""
    span = [-(-n // align) * align for n in lengths]
    by_room = {}                      # free tokens -> slots with exactly that much room
    room, place = [], [None] * len(lengths)
    for i in sorted(range(len(lengths)), key=lambda i: -span[i]):
        need = span[i]
        if need > SLOT:
            raise ValueError(f'text {i} has {lengths[i]} tokens; the slot holds {SLOT}')
        slot = next((by_room[c].pop() for c in range(need, SLOT + 1) if by_room.get(c)), None)
        if slot is None:
            slot = len(room)
            room.append(SLOT)
        place[i] = (slot, SLOT - room[slot])
        room[slot] -= need
        by_room.setdefault(room[slot], []).append(slot)
    return place, len(room)


def load_model(model_dir):
    """The model folder's reference model (trit-packed-v1); the host half runs through it."""
    spec = importlib.util.spec_from_file_location('trit_embed_reference', Path(model_dir) / 'trit_embed.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TritEmbed.from_pretrained(model_dir)


class TritNpu:
    def __init__(self, model_dir, build=None, library=None, profiling=False, native=True):
        """build: the folder with build.json, final.xclbin and insts.bin; by default this checkout's build/, or
        else the model folder's prebuilt npu/. library: the host library, by default the build's (default_library).
        profiling=True (bench.py) also accepts stub and variant builds, whose outputs are not vectors.
        native=True leaves everything after tokenization to the host library (tdn_encode: packing, embedding and
        input LayerNorm, the runs, pooling, output projection and normalization in C++); native=False keeps the
        Python loop below, with the model's own embed and output head.
        Raises NpuUnavailable when a runtime prerequisite is missing (check_npu)."""
        if build is None:
            build = BUILD if (BUILD / 'build.json').exists() else Path(model_dir) / 'npu'
        build = Path(build)
        if not (build / 'build.json').exists():
            raise NpuUnavailable(f'No NPU build at {build} (build.json): in phlegm, run build.py and build_host.py; '
                                 'a model folder ships it prebuilt in npu/.')
        library = Path(library) if library is not None else default_library(build)
        self.runtime = check_npu()
        self.model = load_model(model_dir)
        self.backend = copy.deepcopy(self.model.tokenizer.backend_tokenizer)
        self.backend.enable_truncation(SLOT)
        self.backend.no_padding()
        dims = json.loads((Path(model_dir) / 'config.json').read_text())['dims']
        self.layers = dims['layers']
        self.manifest = json.loads((Path(build) / 'build.json').read_text())
        expected = {'d': 384, 'emb': 384, 'heads': 12, 'ffn': 1536, 'max_pos': SLOT, 'pool': 'mean', 'loops': 1}
        if any(dims[k] != v for k, v in expected.items()):
            raise ValueError(f'The graph is built for {expected}; the model has {dims}')
        if (self.manifest['layers'] != self.layers or self.manifest['cap'] != SLOT
                or self.manifest.get('slots') != SLOTS
                or self.manifest.get('packets') != PACKETS or self.manifest.get('packet_format') != 3
                or self.manifest.get('element_bytes') != ELEMENT or (self.manifest.get('stub') and not profiling)):
            raise ValueError('Compiled encoder shape differs from the model (or is a profiling stub)')
        if not self.manifest.get('bundle'):   # a bundle's sources stay in phlegm; its artifact hashes still hold
            for name, sha in self.manifest['source_sha256'].items():
                if not profiling and digest(ROOT / name) != sha:
                    raise ValueError(f'Source {name} changed since the build; rebuild')
        for name, sha in self.manifest['artifact_sha256'].items():
            if digest(build / name) != sha:
                raise ValueError(f'Changed build artifact {name}')
        self.dll = load_runtime(library, self.runtime)
        try:
            self.handle = self.dll.tdn_open(os.fsencode(build / 'final.xclbin'), os.fsencode(build / 'insts.bin'))
        except RuntimeError as e:
            soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
            if 'mmap' in str(e) and soft != resource.RLIM_INFINITY:
                cause = 'XRT could not lock its device heap.' + memlock_hint(soft)
            else:
                cause = ('Another program may hold the NPU columns this full-array design needs (`xrt-smi examine -r '
                         f'aie-partitions`), or the XRT amdxdna plugin is missing (found: {self.runtime["xdna_plugin"]}).')
            raise NpuUnavailable(f'XRT could not open the NPU or load the encoder ({e}). {cause}') from e
        try:
            weights = weight_stream(self.model)
            for _, m in self.model.ternary_layers():   # the layers now live on the NPU; free the host copies
                m.codes = m.weight = torch.empty(0)
            self.buffers = [self.alloc(weights.nbytes)]
            self.buffers[0][:] = weights
            self.dll.tdn_sync(self.handle, 0, 1, self.buffers[0].nbytes, 0)
            # Per set: segment IDs, input states, K/V scratch, layer outputs.
            self.sets = []
            for _ in range(SETS):
                idx = []
                for size in (ELEMENT, TOKENS * 384 * 4, SLOTS * 12 * 4 * 4096, 2 * 32 * OUT):
                    idx.append(len(self.buffers))
                    self.buffers.append(self.alloc(size))
                for i in idx[:2]:       # submit leaves unused rows as they are: start them finite
                    self.buffers[i][:] = 0
                    self.dll.tdn_sync(self.handle, i, 1, self.buffers[i].nbytes, 0)
                run = self.dll.tdn_bind(self.handle, 5, (ct.c_int * 5)(0, *idx))
                self.sets.append((run, idx))
            self.native = native
            if native:
                self.init_native()
        except BaseException:
            self.close()
            raise
        self.dispatch_ms = []

    def init_native(self):
        """Hand the host library the embedding tables, output head and buffer sets (tdn_host_init; it copies them)."""
        m = self.model
        assert m.out.bias is None and m.out.weight.shape == (384, 384)
        tables = [np.ascontiguousarray(a) for a in (
            m.word.qweight.numpy(), m.word.scale.float().numpy(), m.pos.qweight.numpy(), m.pos.scale.float().numpy(),
            m.tok_type.weight[0].detach().float().numpy(), m.ln_e.weight.detach().float().numpy(),
            m.ln_e.bias.detach().float().numpy(), m.out.weight.detach().float().numpy())]
        word, wscale, pos, pscale, tok, gamma, beta, head = tables
        assert word.dtype == pos.dtype == np.int8 and word.shape[1] == pos.shape[1] == 384
        col = lambda j: (ct.c_int * SETS)(*[idx[j] for _, idx in self.sets])
        self.dll.tdn_host_init(self.handle, word.ctypes.data, wscale.ctypes.data, len(word), pos.ctypes.data,
                               pscale.ctypes.data, len(pos), tok.ctypes.data, gamma.ctypes.data, beta.ctypes.data,
                               float(m.ln_e.eps), head.ctypes.data, SETS, (ct.c_int * SETS)(*[r for r, _ in self.sets]),
                               col(0), col(1), col(3), OUT // 4)

    def run_native(self, token_ids, base, out):
        """Vectors of pre-tokenized texts into out ([len(token_ids), 384] fp32, C-contiguous) through
        tdn_encode; their segment IDs start at base + 1. Returns the dispatch count."""
        n = len(token_ids)
        if not n:
            return 0
        lengths = np.fromiter(map(len, token_ids), np.int64, n)
        offsets = np.zeros(n + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        ids = np.fromiter(itertools.chain.from_iterable(token_ids), np.int32, int(offsets[-1]))
        assert out.dtype == np.float32 and out.flags.c_contiguous and out.shape == (n, 384)
        runs = ct.c_int()
        self.dll.tdn_encode(self.handle, n, ids.ctypes.data, offsets.ctypes.data, base, out.ctypes.data, ct.byref(runs))
        return runs.value

    def alloc(self, size):
        ptr = self.dll.tdn_buf(self.handle, size)
        return np.ctypeslib.as_array((ct.c_uint8 * size).from_address(ptr))

    @torch.inference_mode()
    def start(self, s, ids, pos, seg):
        """Queue one dispatch on buffer set s: token ids, positions and segment IDs (0 = padding)."""
        self.submit(s, self.model.embed(ids, pos).numpy(), seg)

    def submit(self, s, states, seg, rows=None):
        """Queue one dispatch on buffer set s from its input states and segment IDs. states is [256][384]
        fp32, or with rows (token indexes) just those tokens' states; the other rows keep what an earlier
        dispatch left there, which the segment IDs mask out (0 = padding). The NPU takes the states in
        its cores' own layout, [core][k/8][token][8] (f2_pre)."""
        run, (mask, x, _, _) = self.sets[s]
        self.buffers[mask].view(np.int32)[:TOKENS] = seg
        xs = self.buffers[x].view(np.float32).reshape(32, 48, 8, 8).transpose(0, 2, 1, 3)   # core, token, k/8, 8
        if rows is None:
            np.copyto(xs, states.reshape(32, 8, 48, 8))
            cores = 32
        else:
            xs[rows // 8, rows % 8] = states.reshape(-1, 48, 8)
            cores = int(rows[-1]) // 8 + 1 if len(rows) else 1
        self.dll.tdn_sync(self.handle, mask, 1, TOKENS * 4, 0)
        self.dll.tdn_sync(self.handle, x, 1, cores * 8 * 384 * 4, 0)
        self.dll.tdn_start(self.handle, run)

    def wait(self, s, tokens=TOKENS):
        """Wait for set s and return its final-layer states of tokens 0 .. tokens - 1 (rounded up to a
        core's eight), shape (tokens, 384)."""
        run, _ = self.sets[s]
        ms = self.dll.tdn_wait(self.handle, run)
        if ms <= 0:
            raise RuntimeError(f'Incomplete trit_embed dispatch: {ms}')
        self.dispatch_ms.append(ms)
        cores = -(-tokens // 8)
        t = self.trace(s, cores).reshape(2, 32, OUT // 4)[:, :cores, :1536].reshape(2, cores, 4, 384)
        return t.transpose(1, 0, 2, 3).reshape(cores * 8, 384)         # core, half, token, channel

    def finish(self, s):
        """Wait for set s and return its final-layer token states, shape (256, 384)."""
        return torch.from_numpy(self.wait(s).copy())

    def trace(self, s, cores=32):
        """Set s's output packets, fp32 [half][column][row][OUT / 4]: half h of core 4 column + row holds its
        final states for tokens 4h .. 4h + 3 ([4][384]); profiling builds put the timer totals after them in
        half 1. Only cores 0 .. cores - 1 are read back from the device."""
        _, (_, _, _, out) = self.sets[s]
        if cores == 32:
            self.dll.tdn_sync(self.handle, out, 0, self.buffers[out].nbytes, 0)
        else:
            for half in range(2):
                self.dll.tdn_sync(self.handle, out, 0, cores * OUT, half * 32 * OUT)
        return self.buffers[out].view(np.float32).reshape(2, 8, 4, OUT // 4)

    def tokenize(self, texts):
        """Token ids ([CLS] ... [SEP], at most 128 a text), the same as the tokenizer's own
        truncation=True, max_length=128 call returns, through the backend's batch call: a text over CUT
        characters is first cut at the next space. If that prefix still fills the 128 tokens, its first
        126 word pieces are the whole text's (BERT splits words at spaces before word pieces, and a piece
        never crosses a word); texts whose prefix comes out shorter are tokenized whole."""
        cut = []
        for t in texts:
            if len(t) > CUT:
                k = t.find(' ', CUT)
                if k > 0:
                    t = t[:k]
            cut.append(t)
        ids = [e.ids for e in self.backend.encode_batch_fast(cut)]
        redo = [i for i, (t, c, e) in enumerate(zip(texts, cut, ids)) if len(c) < len(t) and len(e) < SLOT]
        for i, e in zip(redo, self.backend.encode_batch_fast([texts[i] for i in redo]) if redo else ()):
            ids[i] = e.ids
        return ids

    @staticmethod
    def layout(token_ids, base=0, align=1):
        """Pack texts (token id lists) into dispatches: the arrays dispatches() reads, for texts
        base .. base + len(token_ids) - 1. Segment ID = text index + 1, 0 = padding."""
        n = len(token_ids)
        lengths = np.fromiter(map(len, token_ids), np.int64, n)
        place, slots = pack(lengths.tolist(), align)
        runs = -(-slots // SLOTS)
        place = np.asarray(place, np.int64).reshape(n, 2)
        first = place[:, 0] * SLOT + place[:, 1]                 # flat index of each text's first token
        step = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        dst = np.repeat(first, lengths) + step
        ids = np.zeros(runs * TOKENS, np.int64)
        pos = np.zeros(runs * TOKENS, np.int64)
        seg = np.zeros(runs * TOKENS, np.int32)
        ids[dst] = np.fromiter(itertools.chain.from_iterable(token_ids), np.int64, len(dst))
        pos[dst] = step
        seg[dst] = np.repeat(np.arange(base + 1, base + n + 1, dtype=np.int32), lengths)
        order = np.argsort(first, kind='stable')
        bounds = np.searchsorted(first[order], np.arange(runs + 1) * TOKENS)
        return base, runs, first, lengths, ids, pos, seg, order, bounds

    @staticmethod
    def dispatches(layout):
        """Per dispatch of a layout: (texts, ids, pos, seg, rows, pooling), with texts the global indexes
        of its texts, ids/pos/seg [256], rows the token indexes in use and pooling the reduceat bounds of
        the texts' token ranges (each below the token count it needs back)."""
        base, runs, first, lengths, ids, pos, seg, order, bounds = layout
        for k in range(runs):
            texts = order[bounds[k]:bounds[k + 1]]
            starts = first[texts] - k * TOKENS
            ends = starts + lengths[texts]
            cuts = np.unique(np.concatenate([starts, ends[ends < TOKENS]]))
            span = slice(k * TOKENS, (k + 1) * TOKENS)
            yield (texts + base, ids[span], pos[span], seg[span], np.flatnonzero(seg[span]),
                   (np.searchsorted(cuts, starts), cuts, lengths[texts], min(TOKENS, int(ends.max()) + 1)))

    @torch.inference_mode()
    def run(self, layouts, n):
        """Unit-norm vectors (torch, [n, 384]) and the dispatch count for n texts laid out by layout()
        (one or more layouts, run in turn); two buffer sets stay in flight."""
        m = self.model
        pooled = np.empty((n, 384), np.float32)
        runs, pending = 0, None

        def pool(s, index, which, cuts, count, used):
            pooled[index] = np.add.reduceat(self.wait(s, used), cuts)[which] / count[:, None]

        for layout in layouts:
            for index, ids, pos, seg, rows, pooling in self.dispatches(layout):
                s = runs % SETS
                if len(rows) == TOKENS:
                    self.submit(s, m.embed(torch.from_numpy(ids), torch.from_numpy(pos)).numpy(), seg)
                else:
                    self.submit(s, m.embed(torch.from_numpy(ids[rows]), torch.from_numpy(pos[rows])).numpy(), seg, rows)
                if pending:
                    pool(*pending)
                pending = (s, index, *pooling)
                runs += 1
        if pending:
            pool(*pending)
        vectors = F.normalize(m.out(torch.from_numpy(pooled)), dim=-1)
        if not torch.isfinite(vectors).all():
            raise ValueError('Nonfinite trit_embed output')
        return vectors, runs

    def encode_ids(self, token_ids, align=1):
        """Unit-norm vectors (torch, [n, 384]) and the dispatch count, for pre-tokenized texts
        (lists of ids including [CLS]/[SEP], at most 128 each). align > 1 takes the Python loop."""
        if self.native and align == 1:
            out = np.empty((len(token_ids), 384), np.float32)
            return torch.from_numpy(out), self.run_native(token_ids, 0, out)
        return self.run([self.layout(token_ids, 0, align)], len(token_ids))

    def encode(self, texts):
        """Unit-norm float32 vectors [n, 384]; inputs are truncated to 128 tokens. Queries need no prefix.

        Past CHUNK texts a thread tokenizes and packs them in chunks (each four times the last) while
        the NPU encodes the chunks before (the tokenizer backend releases the GIL); a chunk's last
        dispatch may run part empty."""
        texts = list(texts)
        n = len(texts)
        if n <= CHUNK:
            if self.native:
                out = np.empty((n, 384), np.float32)
                self.run_native(self.tokenize(texts), 0, out)
                return out
            return self.run([self.layout(self.tokenize(texts))], n)[0].numpy()
        chunks = queue.Queue()

        def produce():
            try:
                start, size = 0, CHUNK
                while start < n:
                    ids = self.tokenize(texts[start:start + size])
                    chunks.put((start, ids) if self.native else self.layout(ids, start))
                    start, size = start + size, size * 4
                chunks.put(None)
            except BaseException as e:
                chunks.put(e)

        def consume():
            while (c := chunks.get()) is not None:
                if isinstance(c, BaseException):
                    raise c
                yield c

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        try:
            if self.native:
                out = np.empty((n, 384), np.float32)
                for start, ids in consume():
                    self.run_native(ids, start, out[start:start + len(ids)])
                return out
            return self.run(consume(), n)[0].numpy()
        finally:
            worker.join()

    def close(self):
        if getattr(self, 'handle', None):
            self.dll.tdn_close(self.handle)
            self.handle = None


SELFTEST = [
    "A man is playing a guitar on the street.",
    "how do index funds work",
    "Index funds track a market index, so their holdings change only when the index does.",
    "The mitochondria produce most of the chemical energy needed to power the cell's biochemical reactions.",
    "Who wrote the novel Pride and Prejudice?",
    "Jane Austen published Pride and Prejudice in 1813; it follows Elizabeth Bennet and her sisters. " * 4,
]


def main(argv=None):
    """python trit_npu.py [MODEL_DIR]: check the NPU prerequisites; with a model folder, also encode a few texts
    on the NPU, compare them with the CPU reference (minimum cosine about 0.998; below 0.995 fails) and time them."""
    import argparse
    import time
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('model_dir', nargs='?', type=Path)
    a = p.parse_args(argv)
    try:
        print(json.dumps(check_npu(), indent=2), flush=True)
        if a.model_dir is None:
            return
        npu = TritNpu(a.model_dir)
    except NpuUnavailable as e:
        sys.exit(f'NPU unavailable: {e}')
    try:
        texts = SELFTEST * 64
        vectors = npu.encode(texts)
        cosine = (vectors * load_model(a.model_dir).encode(texts)).sum(-1)
        seconds = []
        for _ in range(3):
            t = time.perf_counter()
            npu.encode(texts)
            seconds.append(time.perf_counter() - t)
        print(json.dumps({'texts': len(texts), 'cosine_to_cpu': {'min': float(cosine.min()), 'mean': float(cosine.mean())},
                          'texts_per_second': len(texts) / sorted(seconds)[1]}, indent=2))
    finally:
        npu.close()
    if not np.isfinite(vectors).all() or cosine.min() < 0.995:
        sys.exit('FAIL: NPU vectors differ from the CPU reference')
    print('OK')


if __name__ == '__main__':
    main()
