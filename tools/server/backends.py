"""Generation backends.

A backend turns (prompt_ids, params) into a stream of output token ids. Two
implementations:

  MockBackend  - no device, deterministic canned tokens. Default. Makes the whole
                 HTTP/SSE layer testable with zero hardware.
  NpuBackend   - holds ONE resident decode_driver.exe (serve mode) with the
                 interval-3 pools + prefill states resident, and generates by the
                 per-token step protocol, exactly as tools/kernel-interp/
                 generate_npu.py does. The needed pieces of that script are copied
                 here (not imported) so we don't trigger its 2 GB import-time
                 side effects and don't modify it.

Both are driven through the same `generate()` iterator so the server is backend
agnostic. Generation is serialized by a per-backend lock: the NPU is a single
shared device and the driver processes one step at a time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import threading
import os
import sys
import subprocess
import time

from . import config
from .sampler import Sampler


class DeviceBusyError(RuntimeError):
    """The NPU could not be acquired (driver failed to become ready). -> HTTP 503."""


@dataclass
class GenParams:
    max_tokens: int
    sampler: Sampler
    stop_token_ids: set[int] = field(default_factory=set)


class Backend:
    name = "base"

    def generate(self, prompt_ids: list[int], params: GenParams):
        """Yield output token ids one at a time until max_tokens or a stop id."""
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Mock backend                                                                 #
# --------------------------------------------------------------------------- #
class MockBackend(Backend):
    """Deterministic canned generation. Echoes a short, prompt-derived reply as a
    stream of UTF-8 byte ids (which the PlaceholderTokenizer decodes back to text),
    so streaming and non-streaming responses are both meaningful in tests."""

    name = "mock"

    def __init__(self):
        self._lock = threading.Lock()

    def generate(self, prompt_ids: list[int], params: GenParams):
        # Build a canned reply. prompt_ids are UTF-8 bytes under the placeholder
        # tokenizer; echo their length so the response varies with input.
        reply = f"[mock npu] received {len(prompt_ids)} prompt tokens. interval-3 is finite here."
        out_ids = list(reply.encode("utf-8"))[: params.max_tokens]
        with self._lock:
            for tid in out_ids:
                if tid in params.stop_token_ids:
                    return
                time.sleep(0.005)  # simulate token latency so SSE is observable
                yield tid


# --------------------------------------------------------------------------- #
# NPU backend                                                                  #
# --------------------------------------------------------------------------- #
class NpuBackend(Backend):
    """Drives the resident decode_driver.exe (serve mode).

    Lifecycle: lazy. The driver (and the ~2 GB dequantized lm_head cache) is only
    spun up on the first generate() call, so importing/instantiating the server is
    cheap and the shared NPU isn't grabbed until a request actually needs it.

    IMPORTANT LIMITATION (handoff): the resident prefill *states* baked into
    npu-engine/m3out/5li3/state_L*.bin correspond to a FIXED prompt (built offline
    by run_5li3_npu.py). Per-request prefill of arbitrary prompts is NOT yet wired.
    So today this backend generates a continuation of that pre-built prefill,
    seeded from first_token.npy, regardless of the request's messages. The prompt
    still flows through tokenizer + params so the plumbing is complete; only the
    NPU-side prefill is the missing piece. See README "Integration points".
    """

    name = "npu"

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._np = None
        self._m = None          # Q4NX model
        self._W = None          # dequantized lm_head [248320, 2048]
        self._NW = None         # model.norm.weight
        self._sched = None
        self._act_path = f"{config.NPU_OUT_DIR}/srv_act.bin"
        self._hidden_path = f"{config.NPU_OUT_DIR}/srv_hidden.bin"

    # ---- lazy heavy init ---------------------------------------------------
    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        # import numpy + q4nx lazily (only when the NPU path is actually used)
        import numpy as np
        sys.path.insert(0, config.KERNEL_INTERP_DIR)
        from q4nx import Q4NX, bf16_to_f32  # noqa: F401  (from tools/kernel-interp)

        self._np = np
        # Model file lives in the FLM model dir (q4nx.MODEL_DIR resolves it).
        from q4nx import MODEL_DIR
        self._m = Q4NX(os.path.join(MODEL_DIR, config.MODEL_Q4NX))
        self._NW = self._m.bf16("model.norm.weight")
        self._sched = [
            "full_attention"
            if f"model.layer.{l}.self_attn.q_proj.weight" in self._m.tensors
            else "linear_attention"
            for l in range(config.NUM_LAYERS)
        ]
        self._W = self._build_lmhead_matrix()

        cfg = self._write_serve_config()
        self._proc = subprocess.Popen(
            [config.DRIVER_EXE, cfg],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        deadline = time.time() + config.DRIVER_READY_TIMEOUT
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise DeviceBusyError("decode_driver exited before SERVE READY (device busy?)")
            ln = self._proc.stdout.readline()
            if not ln:
                raise DeviceBusyError("decode_driver produced no output (device busy?)")
            if "SERVE READY" in ln:
                return
        raise DeviceBusyError("decode_driver did not become ready in time (device busy?)")

    # ---- lm_head dequant (copied from generate_npu.py) --------------------
    def _build_lmhead_matrix(self):
        np = self._np
        cache = f"{config.NPU_OUT_DIR}/lmhead_W.f32.npy"
        if os.path.exists(cache):
            return np.load(cache, mmap_mode="r")
        from q4nx import bf16_to_f32
        lmb = np.frombuffer(self._m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
        d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
        qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
        r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
        p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
        j = bc * 32 + r + 0 * i
        W = np.zeros((248320, 2048), dtype=np.float32)
        for c0 in range(0, lmb.shape[0], 8192):
            ce = min(c0 + 8192, lmb.shape[0])
            vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce - c0, 32, 8, 32).astype(np.float32)
            dd = d[c0:ce][:, j.reshape(-1)].reshape(ce - c0, 32, 8, 32)
            w = (vals * dd).reshape(ce - c0, 32, 256)
            for cc in range(c0, ce):
                W[32 * (cc // 8):32 * (cc // 8) + 32, 256 * (cc % 8):256 * (cc % 8) + 256] = w[cc - c0]
        np.save(cache, W)
        return W

    # ---- per-token host math (copied from generate_npu.py) ----------------
    def _full_logits(self, hidden_row0):
        np = self._np
        hn = (hidden_row0 / np.sqrt((hidden_row0.astype(np.float64) ** 2).mean() + 1e-6) * self._NW).astype(np.float32)
        return self._W @ hn

    def _embed(self, tok):
        np = self._np
        t0 = self._m.tensors["model.embed_tokens.weight"]
        base = self._m.data_base + t0["data_offsets"][0]
        from q4nx import bf16_to_f32
        return bf16_to_f32(np.frombuffer(self._m.mm[base + tok * 4096:base + (tok + 1) * 4096], dtype=np.uint16))

    def _write_act(self, tok):
        np = self._np
        from q4nx import f32_to_bf16
        act = np.zeros(1048576, dtype=np.uint8)
        act[:4096] = f32_to_bf16(self._embed(tok)).view(np.uint8)
        act[4096:8192] = f32_to_bf16(self._NW).view(np.uint8)
        act.tofile(self._act_path)

    def _write_serve_config(self):
        XB = config.XCLBIN_DIR
        CAP = config.CAP_DIR
        D = config.NPU_OUT_DIR
        n = config.NUM_LAYERS
        lines = [
            "device",
            f"xclbin L {XB}/layer.xclbin",
            f"xclbin LM {XB}/lm_head.xclbin",
            f"kernel k0 L {CAP}/elf_000005.bin",
            f"kernel k1 L {CAP}/elf_000006.bin",
            f"kernel klm LM {CAP}/elf_000003.bin",
        ]
        for l in range(n):
            lines.append(f"buf pool{l} 536870912 {D}/pool_L{l}.bin")
            lines.append(f"buf pack{l} 2097152 {D}/pack_L{l}.bin")
            lines.append(f"buf side{l} 6291456 {D}/side_L{l}.bin")
            lines.append(f"buf state{l} 3145728 {D}/state_L{l}.bin")
        lines.append(f"buf lmpool 542113792 {D}/pool_lmhead.bin")
        lines.append("buf act 1048576")
        lines.append("buf logits 1048576")
        lines.append("serve")
        kern = lambda l: "k0" if l == 0 else "k1"
        chunk: list[int] = []
        for l in range(n):
            chunk.append(l)
            if len(chunk) == 3:
                lines.append("runlist L")
                for c in chunk:
                    lines.append(f"layer {kern(c)} pool{c} act pack{c} side{c} state{c}")
                lines.append("submit")
                lines.append("barrier klm logits lmpool act")
                chunk = []
        if chunk:
            lines.append("runlist L")
            for c in chunk:
                lines.append(f"layer {kern(c)} pool{c} act pack{c} side{c} state{c}")
            lines.append("submit")
        lines.append("barrier klm logits lmpool act")
        lines.append("endserve")
        cfg = f"{D}/srv_serve.txt"
        open(cfg, "w").write("\n".join(lines) + "\n")
        return cfg

    def _seed_token(self) -> int:
        np = self._np
        f = f"{config.NPU_OUT_DIR}/first_token.npy"
        if os.path.exists(f):
            return int(np.load(f))
        return 276

    # ---- generation --------------------------------------------------------
    def generate(self, prompt_ids: list[int], params: GenParams):
        np = None
        with self._lock:
            try:
                self._ensure_started()
            except DeviceBusyError:
                raise
            np = self._np
            tok = self._seed_token()
            for _ in range(params.max_tokens):
                self._write_act(tok)
                self._proc.stdin.write(f"step {self._act_path} {self._hidden_path}\n")
                self._proc.stdin.flush()
                # wait for STEP OK
                ok = False
                t_deadline = time.time() + config.DRIVER_STEP_TIMEOUT
                while time.time() < t_deadline:
                    ln = self._proc.stdout.readline()
                    if not ln:
                        raise DeviceBusyError("driver died mid-step")
                    if "STEP OK" in ln:
                        ok = True
                        break
                    if "STEP" in ln and ("ERR" in ln or "FAILED" in ln):
                        raise RuntimeError("driver step failed: " + ln.strip())
                if not ok:
                    raise DeviceBusyError("driver step timed out")
                from q4nx import bf16_to_f32
                hidden = bf16_to_f32(np.fromfile(self._hidden_path, dtype=np.uint16))[:2048]
                if not np.isfinite(hidden).all():
                    raise RuntimeError("non-finite hidden (interval-3 blowup) — should not happen")
                logits = self._full_logits(hidden)
                nxt = params.sampler.sample(logits)
                if nxt in params.stop_token_ids:
                    return
                yield nxt
                tok = nxt

    def close(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write("quit\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
            self._proc = None


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
_INSTANCE: Backend | None = None
_INSTANCE_LOCK = threading.Lock()


def get_backend() -> Backend:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        name = config.resolve_backend()
        if name == "npu":
            _INSTANCE = NpuBackend()
        else:
            _INSTANCE = MockBackend()
        return _INSTANCE
