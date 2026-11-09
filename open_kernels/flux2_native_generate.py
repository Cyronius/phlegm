r"""flux2_native_generate: FLUX.2 [klein] text-to-image and one-reference edits, every op on
the NPU, for ``phlegm image`` / ``phlegm serve-images`` (backend ``flux2``).

Runs ``klein_pipeline.plan(R, steps, edit)``'s schedule (text encoder, conditioning, the
denoising steps, the VAE; plus the VAE encoder for an edit) through the pyxrt host
(``harness/npu_host.py``), one hardware context per kernel set. Ported from OFLM's
``utilities/dit-chain/generate.py``; weights are read straight from the published
diffusers checkpoint (``transformer/``, ``text_encoder/``, ``vae/``, ``tokenizer/``), no
q4nx bundle. The reference is ``flux2_klein_reference.py`` (the unchanged diffusers
``Flux2KleinPipeline``).

    python -m open_kernels.flux2_native_generate --model <checkpoint> --kernels <set dir> \
        --prompt "a red fox in fresh snow" --size 512 --seed 1 --out <dir> \
        [--image ref.png] [--noise noise.npy] [--steps 4]

Kernel directory: ``export_dit_kernels.py --resolutions ... [--edits ...] --out <dir>``.
Packed weights are cached in ``<kernels>/packed/`` (dit_gemm's bfp16 layout, ~8 GB for
4B) and ``<kernels>/vae_packed.npz`` / ``venc_packed.npz``; each carries a fingerprint of
the checkpoint it came from.

Noise: ``--noise`` takes the reference's ``noise.npy`` ([128, R/16, R/16]); otherwise
``torch.randn``-compatible noise is not available without torch, so the seed drives
numpy's PCG64 (images are then not comparable with the reference's for the same seed).

Writes ``<out>/image.png``, ``<out>/latents.npy`` (packed, bf16 bits) and
``<out>/summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "harness"))
import klein_pipeline as kp  # noqa: E402

SETS = {"gemm": ".", "fa": "fa", "ew": "ew", "conv": "conv", "conv1": "conv1", "vew": "vew"}
_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64,
       "I32": np.int32, "U8": np.uint8}


class SafeTensors:
    """Tensors of a (sharded) safetensors directory, numpy only; bf16 -> float32."""

    def __init__(self, directory: Path):
        d = Path(directory)
        idx = d / "model.safetensors.index.json"
        files = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
                 if idx.exists() else [p.name for p in sorted(d.glob("*.safetensors"))])
        self._where, self._maps = {}, {}
        for f in files:
            with open(d / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self._maps[f] = (np.memmap(d / f, dtype=np.uint8, mode="r"), 8 + n)
            for name, meta in hdr.items():
                if name != "__metadata__":
                    self._where[name] = (f, meta)

    def names(self) -> list[str]:
        return list(self._where)

    def raw(self, name: str) -> np.ndarray:
        f, meta = self._where[name]
        mm, base = self._maps[f]
        a, b = meta["data_offsets"]
        return np.asarray(mm[base + a:base + b]).view(_DT[meta["dtype"]]).reshape(meta["shape"])

    def get(self, name: str) -> np.ndarray:
        raw = self.raw(name)
        if self._where[name][1]["dtype"] == "BF16":
            return (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(np.float32)


def fingerprint(model: Path, subs: tuple[str, ...], code: str) -> str:
    """The checkpoint's weight files (name, size, head/middle/tail samples) plus the packing
    code: a packed cache from another checkpoint or packer is refused, not reused."""
    h = hashlib.sha256(code.encode())
    for sub in subs:
        for p in sorted((model / sub).glob("*.safetensors")):
            n = p.stat().st_size
            h.update(f"{sub}/{p.name}:{n}".encode())
            with open(p, "rb") as f:
                for at in (0, max(0, n // 2 - (1 << 19)), max(0, n - (1 << 20))):
                    f.seek(at)
                    h.update(f.read(1 << 20))
    return h.hexdigest()


def _pack_one(sub: str, name: str, cache: str, model: str) -> tuple[str, float]:
    t0 = time.time()
    specs = kp.dit_weight_specs() if sub == "transformer" else kp.te_weight_specs()
    K, N, build = specs[name]
    b = build(SafeTensors(Path(model) / sub).get)
    assert b.shape == (K, N), (name, b.shape, (K, N))
    out = Path(cache) / f"{name}.bin"
    tmp = out.with_suffix(".tmp")
    kp.pack_b_cols(b).tofile(tmp)
    tmp.replace(out)
    return name, time.time() - t0


def ensure_packed(cache: Path, model: Path, jobs: int) -> dict[str, tuple[Path, int]]:
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / "provenance.txt"
    fp = fingerprint(model, ("transformer", "text_encoder"), inspect.getsource(kp.pack_b_cols))
    if stamp.exists() and stamp.read_text().strip() != fp:
        print(f"{cache}: packed from another checkpoint or packer; repacking", flush=True)
        for p in cache.glob("*.bin"):
            p.unlink()
    stamp.write_text(fp + "\n")
    want = {n: ("transformer", K * N * 9 // 8) for n, (K, N, _) in kp.dit_weight_specs().items()}
    want |= {n: ("text_encoder", K * N * 9 // 8) for n, (K, N, _) in kp.te_weight_specs().items()}
    todo = [n for n, (_, nb) in want.items()
            if not (cache / f"{n}.bin").exists() or (cache / f"{n}.bin").stat().st_size != nb]
    if todo:
        print(f"packing {len(todo)} weights -> {cache} ({jobs} processes)", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(jobs) as ex:
            for i, (n, s) in enumerate(ex.map(_pack_one, [want[n][0] for n in todo], todo,
                                              [str(cache)] * len(todo),
                                              [str(model)] * len(todo))):
                if i % 20 == 0 or i == len(todo) - 1:
                    print(f"  [{i + 1}/{len(todo)}] {n} {s:.1f} s", flush=True)
        print(f"packed in {time.time() - t0:.0f} s", flush=True)
    return {n: (cache / f"{n}.bin", nb) for n, (_, nb) in want.items()}


def packed_coder(cache: Path, model: Path, coder):
    """vae_decoder / vae_encoder pack_weights output, cached with the checkpoint's VAE
    fingerprint: (weight bytes, table, group-norm blocks, index)."""
    fp = fingerprint(model, ("vae",), inspect.getsource(coder.pack_weights))
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["fingerprint"]) == fp:
            return z["w"], z["table"].item(), z["blocks"].view(bfloat16), z["index"].item()
    t0 = time.time()
    st = SafeTensors(model / "vae")
    sd = {k: st.get(k) for k in st.names() if not k.endswith("num_batches_tracked")}
    w, table, blocks, index = coder.pack_weights(sd)
    np.savez(cache, w=w, table=np.array(table, dtype=object), blocks=blocks.view(np.uint16),
             index=np.array(index, dtype=object), fingerprint=np.array(fp))
    print(f"packed {coder.__name__} in {time.time() - t0:.0f} s -> {cache}", flush=True)
    return w, table, blocks, index


def embed_rows(st: SafeTensors, ids: np.ndarray) -> np.ndarray:
    """model.embed_tokens rows for ids (bf16 bits), zero-padded to TE_PAD: XT. A BF16 table
    is copied bit for bit; any other dtype (an F16 re-save) is rounded to bf16, not
    reinterpreted."""
    name = "model.embed_tokens.weight"
    rows = st.raw(name)[ids]
    if st._where[name][1]["dtype"] != "BF16":
        rows = rows.astype(np.float32).astype(bfloat16).view(np.uint16)
    x = np.zeros((len(ids), kp.TE_PAD), np.uint16)
    x[:, :rows.shape[1]] = rows
    return x


def check_checkpoint(model: Path) -> dict:
    """The schedule's constants against the checkpoint's configs: refuse a different
    geometry (another klein size) instead of running it with the wrong layout."""
    tc = json.loads((model / "transformer" / "config.json").read_text())
    te = json.loads((model / "text_encoder" / "config.json").read_text())
    want_t = {"num_layers": kp.N_DBL, "num_single_layers": kp.N_SGL,
              "num_attention_heads": kp.HEADS, "attention_head_dim": kp.HD,
              "joint_attention_dim": 3 * kp.TE_HID, "in_channels": kp.LAT_CH}
    want_e = {"hidden_size": kp.TE_HID, "intermediate_size": kp.TE_MLP}
    bad = [f"transformer.{k}={tc.get(k)} (schedule {v})" for k, v in want_t.items() if tc.get(k) != v]
    bad += [f"text_encoder.{k}={te.get(k)} (schedule {v})" for k, v in want_e.items() if te.get(k) != v]
    if bad:
        raise SystemExit("checkpoint geometry differs from klein_pipeline's: " + "; ".join(bad))
    mi = json.loads((model / "model_index.json").read_text())
    if not mi.get("is_distilled", False):
        raise SystemExit("undistilled (Base) checkpoints need CFG, which this route does not run")
    return {"transformer": {k: tc[k] for k in want_t}, "is_distilled": True}


class Runner:
    def __init__(self, model: Path, kdir: Path, R: int, steps: int, edit: bool, pack_jobs: int):
        from npu_host import Npu
        import vae_decoder as vd

        t0 = time.time()
        self.R, self.steps = R, steps
        self.pl = kp.plan(R, steps=steps, edit=edit)
        self.dit = SafeTensors(model / "transformer")
        self.te = SafeTensors(model / "text_encoder")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model / "tokenizer" / "tokenizer.json"))
        packed = ensure_packed(kdir / "packed", model, pack_jobs)

        self.npu = npu = Npu()
        self.sets = {n: npu.kernel_set(n, kdir / d) for n, d in SETS.items()}
        fa_meta = json.loads((kdir / "fa" / "dit_fa.json").read_text())
        self.vl_words = fa_meta["patch"]["te_attn"]["valid_len"]

        self.bufs = {n: npu.buf(n, nb) for n, nb in self.pl.buffers.items()}
        for b in self.bufs.values():
            b.zero()
        self.bufs["PARAMS"].write(kp.fill_params(self.dit.get, self.te.get))
        qin = self.pl.vae.buffers["QIN"]
        q = np.zeros((qin.H * qin.W, qin.C), bfloat16)
        q[:, 512] = 1                                  # the VAE attention GEMM's bias row
        self.bufs["v_QIN"].write(q)
        wbytes, self.vtable, blocks, self.vindex = packed_coder(kdir / "vae_packed.npz", model, vd)
        self.vW, self.vS = npu.buf("vae_W", wbytes.size), npu.buf("vae_S", blocks.size * 2)
        self.vW.write(wbytes)
        self.vS.write(blocks)
        self.blk_bytes = vd.BLOCK * vd.EL * 2
        if edit:
            import vae_encoder as ve
            wbytes, self.etable, blocks, self.eindex = packed_coder(kdir / "venc_packed.npz",
                                                                    model, ve)
            self.eW, self.eS = npu.buf("venc_W", wbytes.size), npu.buf("venc_S", blocks.size * 2)
            self.eW.write(wbytes)
            self.eS.write(blocks)

        t1 = time.time()
        self.w = {}
        for n, (path, nb) in packed.items():
            self.w[n] = npu.buf(f"w:{n}", nb)
            self.w[n].load(path)
        gb = sum(nb for _, nb in packed.values()) / 2 ** 30
        print(f"weights: {gb:.2f} GiB in {time.time() - t1:.1f} s", flush=True)

        self.sig = kp.sigmas(R, steps)
        self.bufs["TF"].write(kp.timestep_features(self.sig, steps))
        self.bufs["DT"].write(kp.dt_params(self.sig, steps))
        self.ops = []
        for o in self.pl.ops:
            st = self.sets[o["set"]].stream(o["stream"])
            self.ops.append((o["set"], st, [self._arg(a) for a in o["args"]], o["phase"], o))
        self.vl_stream = self.sets["fa"].stream("te_attn")
        self.load_s = time.time() - t0
        print(f"loaded {R}x{R} ({'edit' if edit else 't2i'}, {steps} steps): "
              f"{len(self.ops)} dispatches in {self.load_s:.1f} s", flush=True)

    def _arg(self, ref):
        kind = ref[0]
        if kind == "buf":
            _, n, off, nb = ref
            return self.bufs[n].view(off, nb)
        if kind == "w":
            return self.w[ref[1]].bo
        if kind == "vae_w":
            off, n = self.vtable[ref[1]]
            return self.vW.view(off, n)
        if kind == "vae_gn":
            return self.vS.view(self.vindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        if kind == "venc_w":
            off, n = self.etable[ref[1]]
            return self.eW.view(off, n)
        if kind == "venc_gn":
            return self.eS.view(self.eindex[ref[1]] * self.blk_bytes, self.blk_bytes)
        raise ValueError(ref)

    def set_prompt(self, prompt: str) -> tuple[np.ndarray, int]:
        ids, n_real = kp.token_ids(self.tok, prompt)
        self.bufs["XT"].write(embed_rows(self.te, ids))
        self.vl_stream.patch(self.vl_words, n_real)
        return ids, n_real

    def set_noise(self, lat_bits: np.ndarray) -> None:
        self.bufs["LAT"].write(np.ascontiguousarray(lat_bits, np.uint16))

    def set_reference(self, rgb: np.ndarray) -> None:
        """uint8 [R, R, 3] as bf16 2 (x / 255) - 1 in channels 0-2 of the encoder's
        zero-bordered input (diffusers' float32 arithmetic, then bf16)."""
        b = self.pl.enc.buffers["IN"]
        x = np.zeros((b.H + 2, b.pitch, b.C), bfloat16)
        x[1:b.H + 1, 1:b.W + 1, :3] = (np.float32(2) * (rgb.astype(np.float32) / np.float32(255))
                                       - np.float32(1)).astype(bfloat16)
        self.bufs[self.pl.enc_buffers["IN"]].write(x)

    def run(self) -> dict[str, float]:
        from npu_host import COMPLETED

        def wait(h, what):
            st = h.wait()
            if st != COMPLETED:
                raise RuntimeError(f"{what}: {st}")

        # Every queued run's handle stays alive until it has completed: Linux XRT aborts
        # the process when an xrt::run is destroyed while its command is in flight (the
        # Windows runner overwrote all but the last handle).
        phases: dict[str, float] = {}
        cur_set, cur_phase = None, None
        inflight: list = []

        def drain(what):
            for h in inflight:
                wait(h, what)
            inflight.clear()

        t_phase = t0 = time.perf_counter()
        for kset, st, bos, phase, o in self.ops:
            if kset != cur_set or phase != cur_phase:
                drain(f"{cur_set} before {o['what']}")
                if phase != cur_phase:
                    now = time.perf_counter()
                    if cur_phase is not None:
                        phases[cur_phase] = now - t_phase
                    t_phase, cur_phase = now, phase
                cur_set = kset
            if len(inflight) >= 64:                    # same context: in order, so the
                wait(inflight.pop(0), f"{kset}/{st.name}")   # oldest finishes first
            inflight.append(st.start(*bos))
        drain("last ops")
        now = time.perf_counter()
        phases[cur_phase] = now - t_phase
        phases["total"] = now - t0
        return phases

    def image(self) -> np.ndarray:
        R = self.R
        return self.bufs["v_RGBA"].read(np.uint8).reshape(-1, 8192)[:, :4096] \
            .reshape(R, R, 4)[..., :3].copy()

    def latents(self) -> np.ndarray:
        T = kp.image_tokens(self.R)
        return self.bufs["LAT"].read(np.uint16, 0, T * kp.LAT_CH).reshape(T, kp.LAT_CH)


def load_reference(path: Path, R: int) -> np.ndarray:
    """uint8 [R, R, 3]: EXIF orientation, RGB, centre crop to a square, LANCZOS to R."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != R:
        img = img.resize((R, R), Image.Resampling.LANCZOS)
    return np.asarray(img, np.uint8)


def pack_noise(noise: np.ndarray) -> np.ndarray:
    """[128, h, w] -> diffusers' _pack_latents order [h*w, 128], as bf16 bits."""
    c, h, w = noise.shape
    return np.ascontiguousarray(noise.reshape(c, h * w).T).astype(bfloat16).view(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="diffusers checkpoint directory")
    ap.add_argument("--kernels", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=kp.STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=Path, help="initial latents [128, R/16, R/16] (.npy)")
    ap.add_argument("--image", type=Path, help="reference image: a one-reference edit")
    ap.add_argument("--pack-jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    why = kp.check_resolution(a.size) or (kp.check_edit(a.size, a.size) if a.image else None)
    if why:
        raise SystemExit(f"unsupported size: {why}")
    if not 1 <= a.steps <= 50:
        raise SystemExit("steps must be between 1 and 50")
    geometry = check_checkpoint(a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    T = kp.image_tokens(a.size)
    if a.noise:
        noise = np.load(a.noise).astype(np.float32)
        if noise.shape != (kp.LAT_CH, a.size // 16, a.size // 16):
            raise SystemExit(f"noise shape {noise.shape} does not fit {a.size}")
        lat = pack_noise(noise)
    else:
        # An edit draws from its own stream: with the text-to-image noise of the same seed
        # (the server's default seed is shared), editing an image generated with that seed
        # reproduces the reference instead of applying the edit (measured 2026-11-08).
        rng = np.random.default_rng([a.seed, 1] if a.image else a.seed)
        lat = rng.standard_normal((T, kp.LAT_CH), np.float32).astype(bfloat16).view(np.uint16)

    r = Runner(a.model, a.kernels, a.size, a.steps, a.image is not None, a.pack_jobs)
    ids, n_real = r.set_prompt(a.prompt)
    r.set_noise(lat)
    if a.image:
        r.set_reference(load_reference(a.image, a.size))
    phases = r.run()
    from PIL import Image
    image_path = a.out / "image.png"
    Image.fromarray(r.image()).save(image_path)
    np.save(a.out / "latents.npy", r.latents())
    summary = {"image": str(image_path), "prompt": a.prompt, "size": a.size,
               "steps": a.steps, "seed": a.seed, "noise": str(a.noise) if a.noise else None,
               "edit": a.image is not None, "reference": str(a.image) if a.image else None,
               "tokens": n_real, "model": str(a.model), "kernels": str(a.kernels),
               "checkpoint": geometry, "dispatches": len(r.ops),
               "load_s": round(r.load_s, 2),
               "npu_phases_s": {k: round(v, 4) for k, v in phases.items()}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{phases['total']:.2f} s on the NPU -> {image_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
