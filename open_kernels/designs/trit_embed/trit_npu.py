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
the NPU runs the previous one. Build the graph (build.py) and the XRT host library (build_host.py) first.
An instance owns mutable device buffers: use it from one thread.
"""
import copy
import ctypes as ct
import hashlib
import importlib.util
import itertools
import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'
LIBRARY = Path(os.environ.get('TRIT_NPU_LIBRARY', str(BUILD / 'host/libtd_npu2.so')))
SLOT = 128
SLOTS = 2               # slots a dispatch
TOKENS = SLOTS * SLOT
OUT = 6912              # a core's output packet (trit_design.OSZ)
SETS = 2
PACKETS = 170
ELEMENT = 8192   # broadcast element bytes
CUT = 1000       # characters: tokenize cuts longer texts first (exact; see TritNpu.tokenize)
CHUNK = 128      # texts that encode tokenizes before the first dispatch


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
    def __init__(self, model_dir, build=BUILD, library=LIBRARY, profiling=False):
        """profiling=True (bench.py) also accepts stub and variant builds, whose outputs are not vectors."""
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
                for size in (ELEMENT, TOKENS * 384 * 4, SLOTS * 12 * 4 * 4096, 2 * 32 * OUT):
                    idx.append(len(self.buffers))
                    self.buffers.append(self.alloc(size))
                for i in idx[:2]:       # submit leaves unused rows as they are: start them finite
                    self.buffers[i][:] = 0
                    self.dll.tdn_sync(self.handle, i, 1, self.buffers[i].nbytes, 0)
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
        (lists of ids including [CLS]/[SEP], at most 128 each)."""
        return self.run([self.layout(token_ids, 0, align)], len(token_ids))

    def encode(self, texts):
        """Unit-norm float32 vectors [n, 384]; inputs are truncated to 128 tokens. Queries need no prefix.

        Past CHUNK texts a thread tokenizes and packs them in chunks (each four times the last) while
        the NPU encodes the chunks before (the tokenizer backend releases the GIL); a chunk's last
        dispatch may run part empty."""
        texts = list(texts)
        n = len(texts)
        if n <= CHUNK:
            return self.run([self.layout(self.tokenize(texts))], n)[0].numpy()
        chunks = queue.Queue()

        def produce():
            try:
                start, size = 0, CHUNK
                while start < n:
                    chunks.put(self.layout(self.tokenize(texts[start:start + size]), start))
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
            return self.run(consume(), n)[0].numpy()
        finally:
            worker.join()

    def close(self):
        if getattr(self, 'handle', None):
            self.dll.tdn_close(self.handle)
            self.handle = None
