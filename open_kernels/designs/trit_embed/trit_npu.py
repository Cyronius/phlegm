"""Run trit-embed-3l on the XDNA2 NPU: the host side of the trit_embed encoder graph.

    from trit_npu import TritNpu
    npu = TritNpu("path/to/trit-embed-3l")   # the model folder: config.json, model.safetensors, tokenizer, trit_embed.py
    vectors = npu.encode(["how do index funds work", "Index funds track a market index."])  # (2, 384), unit norm
    npu.close()

The NPU runs every encoder layer in one dispatch per 128-token slot, and short texts share a slot
(best-fit packing; per-token segment IDs keep their attention apart). At load the host repacks the model
folder's 2-bit codes and bf16 row scales into the NPU's packets (weight_stream). Per dispatch it tokenizes,
does the int8 embedding lookup and input LayerNorm, mean-pools, and applies the output projection, all with
the model folder's own reference code, and keeps two buffer sets in flight so it prepares one dispatch while
the NPU runs the previous one. Build the graph (build.py) and the XRT host library (build_host.py) first.
An instance owns mutable device buffers: use it from one thread.
"""
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'
LIBRARY = Path(os.environ.get('TRIT_NPU_LIBRARY', str(BUILD / 'host/libtd_npu2.so')))
SLOT = 128
SETS = 2
PACKETS = 170
ELEMENT = 8192   # broadcast element bytes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runtime(library=LIBRARY):
    dll = ct.CDLL(str(library))
    dll.tdn_open.argtypes, dll.tdn_open.restype = [ct.c_char_p, ct.c_char_p], ct.c_void_p
    dll.tdn_buf.argtypes, dll.tdn_buf.restype = [ct.c_void_p, ct.c_size_t], ct.c_void_p
    dll.tdn_sync.argtypes, dll.tdn_sync.restype = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t], None
    dll.tdn_bind.argtypes, dll.tdn_bind.restype = [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_int)], ct.c_int
    dll.tdn_start.argtypes, dll.tdn_start.restype = [ct.c_void_p, ct.c_int], ct.c_int
    dll.tdn_wait.argtypes, dll.tdn_wait.restype = [ct.c_void_p, ct.c_int], ct.c_double
    dll.tdn_close.argtypes, dll.tdn_close.restype = [ct.c_void_p], None
    dll.tdn_last_error.argtypes, dll.tdn_last_error.restype = [], ct.c_char_p

    def check(result, function, arguments):
        error = dll.tdn_last_error()
        if error:
            raise RuntimeError(f'{function.__name__}: {error.decode(errors="replace")}')
        return result
    for name in ('tdn_open', 'tdn_buf', 'tdn_sync', 'tdn_bind', 'tdn_start', 'tdn_wait'):
        getattr(dll, name).errcheck = check
    return dll


def pack_weights(codes: np.ndarray) -> np.ndarray:
    """Ternary codes [N, K] -> [N/32, K/16, 128] bytes: 2-bit fields, four output channels per byte
    (packet format 2, fused2.h): each block's channels 0-15 store code + 1, channels 16-31 code & 3."""
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


def parameter_packet(bias, scale):
    p = np.zeros(1024, np.float32)
    p[:len(bias)] = np.asarray(bias).reshape(-1)
    p[512:512 + len(scale)] = np.asarray(scale).reshape(-1)
    return p.view(np.uint8)


def weight_stream(model):
    """Per layer, 170 packets of 4096 bytes in consumption order, broadcast as 86 elements of 8192
    bytes: consecutive pairs, except that each LayerNorm packet has an element of its own. A weight
    packet holds one 32-output-channel block of packed ternary codes, then at byte 3072 its 8-channel
    chunks' bias (fp32) and weight scale (bf16: scale for channels 0-15, scale/4 for 16-31), each
    repeated for the 4 token rows (packet format 2, fused2.h)."""
    packets = []

    def linear(codes, scale, bias):
        assert len(scale) == len(bias) == codes.shape[0]
        for n, block in enumerate(pack_weights(codes)):
            p = np.zeros(4096, np.uint8)
            p[:block.size] = block.reshape(-1)
            b, s = bias[n * 32:(n + 1) * 32], scale[n * 32:(n + 1) * 32]
            s = np.concatenate([s[:16], s[16:] / 4]).astype(np.float32)          # bf16-exact
            rep = lambda v: np.tile(v.reshape(4, 1, 8), (1, 4, 1)).reshape(-1)   # [chunk][row][8]
            p[3072:3584] = rep(np.asarray(b, np.float32)).view(np.uint8)
            p[3584:3840] = rep((s.view(np.uint32) >> 16).astype(np.uint16)).view(np.uint8)
            packets.append(p)

    for layer in model.layers:
        start = len(packets)
        for name in ('q', 'k', 'v', 'o'):
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
        ln1 = 48                            # Q, K, V and O: 4 x 12 packets, then the first LayerNorm
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
    def __init__(self, model_dir, build=BUILD, library=LIBRARY, profiling=False):
        """profiling=True (bench.py) also accepts stub and variant builds, whose outputs are not vectors."""
        self.model = load_model(model_dir)
        dims = json.loads((Path(model_dir) / 'config.json').read_text())['dims']
        self.layers = dims['layers']
        self.manifest = json.loads((Path(build) / 'build.json').read_text())
        expected = {'d': 384, 'emb': 384, 'heads': 12, 'ffn': 1536, 'max_pos': SLOT, 'pool': 'mean', 'loops': 1}
        if any(dims[k] != v for k, v in expected.items()):
            raise ValueError(f'The graph is built for {expected}; the model has {dims}')
        if (self.manifest['layers'] != self.layers or self.manifest['cap'] != SLOT
                or self.manifest.get('packets') != PACKETS or self.manifest.get('packet_format') != 2
                or self.manifest.get('element_bytes') != ELEMENT or (self.manifest.get('stub') and not profiling)):
            raise ValueError('Compiled encoder shape differs from the model (or is a profiling stub)')
        for name, sha in self.manifest['source_sha256'].items():
            if not profiling and digest(ROOT / name) != sha:
                raise ValueError(f'Source {name} changed since the build; rebuild')
        for name, sha in self.manifest['artifact_sha256'].items():
            if digest(Path(build) / name) != sha:
                raise ValueError(f'Changed build artifact {name}')
        self.dll = load_runtime(library)
        self.handle = self.dll.tdn_open(os.fsencode(Path(build) / 'final.xclbin'), os.fsencode(Path(build) / 'insts.bin'))
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
                for size in (ELEMENT, SLOT * 384 * 4, 32 * 8192, self.layers * 32 * 8192):
                    idx.append(len(self.buffers))
                    self.buffers.append(self.alloc(size))
                run = self.dll.tdn_bind(self.handle, 5, (ct.c_int * 5)(0, *idx))
                self.sets.append((run, idx))
        except BaseException:
            self.close()
            raise
        self.dispatch_ms = []

    def alloc(self, size):
        ptr = self.dll.tdn_buf(self.handle, size)
        return np.ctypeslib.as_array((ct.c_uint8 * size).from_address(ptr))

    @torch.inference_mode()
    def start(self, s, ids, pos, seg):
        """Queue one dispatch on buffer set s: token ids, positions and segment IDs (0 = padding)."""
        m = self.model
        run, (mask, x, _, _) = self.sets[s]
        states = m.embed(ids, pos)
        self.buffers[mask].view(np.int32)[:SLOT] = seg
        self.buffers[x].view(np.float32)[:] = states.numpy().reshape(32, 4, 48, 8).transpose(0, 2, 1, 3).reshape(-1)
        for i in (mask, x):
            self.dll.tdn_sync(self.handle, i, 1, self.buffers[i].nbytes, 0)
        self.dll.tdn_start(self.handle, run)

    def finish(self, s):
        """Wait for set s and return its final-layer token states, shape (128, 384)."""
        run, (_, _, _, out) = self.sets[s]
        ms = self.dll.tdn_wait(self.handle, run)
        if ms <= 0:
            raise RuntimeError(f'Incomplete trit_embed dispatch: {ms}')
        self.dispatch_ms.append(ms)
        layer = self.buffers[out].nbytes // self.layers
        self.dll.tdn_sync(self.handle, out, 0, layer, (self.layers - 1) * layer)
        raw = self.buffers[out][(self.layers - 1) * layer:].view(np.float32).reshape(32, 2048)[:, :1536]
        return torch.from_numpy(raw.reshape(32, 48, 4, 8).transpose(0, 2, 1, 3).reshape(SLOT, 384).copy())

    @torch.inference_mode()
    def encode_ids(self, token_ids, align=1):
        """Unit-norm vectors (torch, [n, 384]) and the dispatch count, for pre-tokenized texts
        (lists of ids including [CLS]/[SEP], at most 128 each)."""
        m = self.model
        place, slots = pack([len(t) for t in token_ids], align)
        members = [[] for _ in range(slots)]
        for i, (s, off) in enumerate(place):
            members[s].append((i, off, len(token_ids[i])))
        pooled = torch.empty((len(token_ids), 384))

        def pool(s, group):
            states = self.finish(s)
            for i, off, n in group:
                pooled[i] = states[off:off + n].mean(0)

        pending = None
        for k, group in enumerate(members):
            ids = np.zeros(SLOT, np.int64)
            pos = np.zeros(SLOT, np.int64)
            seg = np.zeros(SLOT, np.int32)
            for j, (i, off, n) in enumerate(group):
                ids[off:off + n] = token_ids[i]
                pos[off:off + n] = np.arange(n)
                seg[off:off + n] = j + 1
            self.start(k % SETS, torch.from_numpy(ids), torch.from_numpy(pos), seg)
            if pending:
                pool(*pending)
            pending = (k % SETS, group)
        if pending:
            pool(*pending)
        vectors = F.normalize(m.out(pooled), dim=-1)
        if not torch.isfinite(vectors).all():
            raise ValueError('Nonfinite trit_embed output')
        return vectors, slots

    def encode(self, texts):
        """Unit-norm float32 vectors [n, 384]; inputs are truncated to 128 tokens. Queries need no prefix."""
        ids = self.model.tokenizer(list(texts), truncation=True, max_length=SLOT)['input_ids']
        return self.encode_ids(ids)[0].numpy()

    def close(self):
        if getattr(self, 'handle', None):
            self.dll.tdn_close(self.handle)
            self.handle = None
