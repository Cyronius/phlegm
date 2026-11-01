"""Qwen Image 2.1 first-frame VAE encoder on the NPU: plan, weight pack, kernel export, run.

The published edit pipeline encodes each condition image with
``AutoencoderKLQwenImage21.encode`` (RGBA in [-1, 1], one frame, argmax latents) and
normalizes the latents with ``latents_mean`` / ``latents_std``. For one frame:

* every ``CausalConv3d`` is a 2-D convolution (3x3 pad 1, or 1x1);
* ``downsample2d`` / ``downsample3d`` is ``conv3x3(pad(x, right 1, bottom 1), stride 2)``;
  on the first frame ``downsample3d`` only caches its input, so ``time_conv`` is inactive;
* each residual down block adds ``AvgDown3D(block input)``. With one frame its temporal
  factor 2 pads one zero frame in front, so the map is linear and fixed: output channel o
  is the mean of ``group`` consecutive entries of the (channel, frame, row parity, column
  parity) flattening, where the zero frame contributes zeros;
* the mid block is resnet, single-head 768-wide attention over the 32x32 grid, resnet.

Native mapping (all arithmetic on the NPU; the host only copies layouts):

* 3x3 and 1x1 convolutions: ``dit_conv`` (channels padded to 64 in / 128 out);
* stride-2 convolutions: the host writes the input's space-to-depth grid (a layout copy,
  channel (2p + q)C + c holds pixel (2i + p, 2j + q)); a 3x3 ``dit_conv`` with
  ``vae_encoder.s2d_weights`` (zero on the -1 taps, so the right/bottom pad lands on the
  grid's zero border) computes the downsample;
* ``AvgDown3D``: a 1x1 ``dit_conv`` on the space-to-depth grid of the block input with
  fixed weights 1/group (exact in BF16);
* RMSNorm (+ SiLU) and residual adds: ``qwen_vae_ew`` at physical widths, the gamma scaled
  by sqrt(logical / physical) as in the decoder;
* attention: ``dit_fa`` single-head VAE variant at 768 (``DF_VAE_DIM=768``);
* ``conv_out``, ``quant_conv``'s mean half and the latent normalization fold into one 3x3
  convolution (float64 fold of the FP32 checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen-Image-2.1"
DIMS = [96, 96, 192, 384, 768, 768]          # base_dim * [1] + dim_mult
TEMPORAL = [False, True, True, True]          # temperal_downsample (blocks 0..3)
Z = 64
ATTN = 768
EW_ELEMENT = {128: 1024, 256: 1024, 384: 1536, 768: 1536}


def phys_out(c: int) -> int:
    return -(-c // 128) * 128


def phys_in(c: int) -> int:
    return -(-c // 64) * 64


# --------------------------------------------------------------------------- plan

def plan(resolution: int = 512) -> list[dict]:
    """The encode, in order. Convs carry logical (ci, co) and physical (Cin, Cout)."""
    if resolution not in (512, 1024):
        raise ValueError("the Qwen VAE encoder plan covers 512- and 1024-pixel squares")
    ops: list[dict] = []

    def conv(name, h, ci, co, taps, *, cin=None, source="checkpoint", **extra):
        ops.append({"kind": "conv", "name": name, "h": h, "ci": ci, "co": co, "taps": taps,
                    "Cin": cin or phys_in(phys_out(ci) if ci > 4 else ci),
                    "Cout": phys_out(co), "source": source, **extra})

    def rms(name, h, c, silu=True):
        ops.append({"kind": "rms_silu" if silu else "rms", "name": name, "h": h, "c": c,
                    "C": phys_out(c)})

    def resnet(base, h, ci, co):
        ops.append({"kind": "save", "name": base + ".input"})
        if ci != co:
            conv(base + ".conv_shortcut", h, ci, co, 1)
        rms(base + ".norm1", h, ci)
        conv(base + ".conv1", h, ci, co, 3)
        rms(base + ".norm2", h, co)
        conv(base + ".conv2", h, co, co, 3)
        ops.append({"kind": "add", "name": base + ".residual", "h": h, "c": co,
                    "C": phys_out(co), "other": base + (".conv_shortcut" if ci != co else ".input")})

    h = resolution
    conv("encoder.conv_in", h, 4, DIMS[0], 3, cin=64)
    for i in range(5):
        ci, co = DIMS[i], DIMS[i + 1]
        base = f"encoder.down_blocks.{i}"
        ops.append({"kind": "save", "name": base + ".input"})
        for j in range(2):
            resnet(f"{base}.resnets.{j}", h, ci if j == 0 else co, co)
        if i < 4:
            ops.append({"kind": "s2d", "name": base + ".downsampler.s2d", "h": h, "c": co,
                        "source": "x"})
            conv(base + ".downsampler.resample.1", h // 2, co, co, 3, cin=phys_in(4 * co),
                 source="s2d")
            ops.append({"kind": "s2d", "name": base + ".avg_shortcut.s2d", "h": h, "c": ci,
                        "source": base + ".input"})
            conv(base + ".avg_shortcut", h // 2, ci, co, 1, cin=phys_in(4 * ci),
                 source="avgdown", factor_t=2 if TEMPORAL[i] else 1)
            h //= 2
            ops.append({"kind": "add", "name": base + ".shortcut_add", "h": h, "c": co,
                        "C": phys_out(co), "other": base + ".avg_shortcut"})
        else:
            ops.append({"kind": "add", "name": base + ".shortcut_add", "h": h, "c": co,
                        "C": phys_out(co), "other": base + ".input"})
        ops.append({"kind": "stage", "name": f"down_blocks.{i}", "c": co})
    mid = "encoder.mid_block"
    resnet(mid + ".resnets.0", h, ATTN, ATTN)
    ops.append({"kind": "save", "name": mid + ".attentions.0.input"})
    rms(mid + ".attentions.0.norm", h, ATTN, silu=False)
    conv(mid + ".attentions.0.to_qkv", h, ATTN, 3 * ATTN, 1)
    ops.append({"kind": "attention", "name": mid + ".attentions.0", "h": h, "c": ATTN})
    conv(mid + ".attentions.0.proj", h, ATTN, ATTN, 1)
    ops.append({"kind": "add", "name": mid + ".attentions.0.residual", "h": h, "c": ATTN,
                "C": ATTN, "other": mid + ".attentions.0.input"})
    resnet(mid + ".resnets.1", h, ATTN, ATTN)
    ops.append({"kind": "stage", "name": "mid_block", "c": ATTN})
    rms("encoder.norm_out", h, ATTN)
    conv("latent_out", h, ATTN, Z, 3, source="latent_out")
    ops.append({"kind": "stage", "name": "latents", "c": Z})
    for op in ops:
        if op["kind"] == "conv" and op["source"] != "s2d" and op["name"] != "encoder.conv_in":
            op["Cin"] = phys_out(op["ci"]) if op["source"] != "avgdown" else op["Cin"]
    return ops


def conv_spec(op: dict) -> dict:
    return {"H": op["h"], "W": op["h"], "Cin": op["Cin"], "Cout": op["Cout"], "up": False}


def conv_stream(op: dict) -> str:
    return f"conv_h{op['h']}_ci{op['Cin']}_co{op['Cout']}_k{op['taps']}"


def ew_spec(op: dict) -> dict:
    h, c = op["h"], op["C"]
    kind = {"rms_silu": "rms_silu", "rms": "rms", "add": "add"}[op["kind"]]
    pitch = max(1, 128 // h) * (h + 2)
    size = (h + 2) * pitch * c
    el = EW_ELEMENT[c]
    spec = {"op": kind, "C": c, "H": h, "W": h, "el": el,
            "a": {"pitch": pitch}, "y": {"pitch": pitch},
            "sizes": {"X": size, "B": size if kind == "add" else 1,
                      "P": 1 if kind == "add" else 2 * el, "Y": size}}
    if kind == "add":
        spec["b"] = {"pitch": pitch}
    return spec


def ew_stream(op: dict) -> str:
    return f"ew_{ew_spec(op)['op']}_h{op['h']}_c{op['C']}"


def streams(resolution: int) -> tuple[dict, dict]:
    """({conv stream: {spec, taps}}, {ew stream: spec}) for the plan."""
    conv, ew = {}, {}
    for op in plan(resolution):
        if op["kind"] == "conv":
            conv[conv_stream(op)] = {"spec": conv_spec(op), "taps": 9 if op["taps"] == 3 else 1}
        elif op["kind"] in ("rms", "rms_silu", "add"):
            ew[ew_stream(op)] = ew_spec(op)
    return conv, ew


# --------------------------------------------------------------------------- weights

def _tensor(path: Path, header: dict, name: str) -> np.ndarray:
    item = header[name]
    if item["dtype"] != "F32":
        raise ValueError(f"{name} is not FP32")
    with path.open("rb") as stream:
        data = 8 + struct.unpack("<Q", stream.read(8))[0]
    return np.array(np.memmap(path, mode="r", dtype="<f4", offset=data + item["data_offsets"][0],
                              shape=tuple(item["shape"])))


def avgdown_weights(ci: int, co: int, factor_t: int) -> np.ndarray:
    """AvgDown3D(ci -> co, factor_t, factor_s=2) on one frame as a 1x1 conv on the
    space-to-depth grid (channel (2p + q) ci + c): [co, 4 ci, 1, 1]."""
    factor = factor_t * 4
    group = ci * factor // co
    w = np.zeros((co, 4 * ci, 1, 1), np.float32)
    real = factor_t - 1                          # the zero pad frame comes first
    for c in range(ci):
        for p in range(2):
            for q in range(2):
                f = ((c * factor_t + real) * 2 + p) * 2 + q
                w[f // group, (2 * p + q) * ci + c, 0, 0] = 1.0 / group
    return w


def latent_out_weights(get, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """conv_out, quant_conv's mean half and (mu - mean) / std as one 3x3 conv [64, 768]."""
    wc, bc = get("encoder.conv_out.weight").astype(np.float64), get("encoder.conv_out.bias").astype(np.float64)
    wq = get("quant_conv.weight").astype(np.float64)[:Z, :, 0, 0]
    bq = get("quant_conv.bias").astype(np.float64)[:Z]
    w = np.einsum("ck,kiyx->ciyx", wq, wc)
    b = wq @ bc + bq
    mean = np.asarray(config["latents_mean"], np.float64)
    std = np.asarray(config["latents_std"], np.float64)
    return (w / std[:, None, None, None]).astype(np.float32), ((b - mean) / std).astype(np.float32)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(model: Path, out: Path, resolution: int = 512) -> dict:
    from open_kernels.designs.dit_conv.conv_pack import pack_conv
    from open_kernels.designs.dit_conv.dit_conv import resolve_spec
    from open_kernels.image_safetensors import read_header
    from open_kernels.vae_encoder import s2d_weights
    file = model / "vae/diffusion_pytorch_model.safetensors"
    header = read_header(file)
    config = json.loads((model / "vae/config.json").read_text())
    for key, value in {"base_dim": 96, "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 2,
                       "z_dim": Z, "is_residual": True, "in_channels": 4,
                       "temperal_downsample": TEMPORAL, "attn_scales": []}.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported Qwen VAE encoder {key}: {config.get(key)!r}")
    get = lambda name: _tensor(file, header, name)  # noqa: E731
    out.mkdir(parents=True, exist_ok=False)
    records, used = {}, set()
    for index, op in enumerate(o for o in plan(resolution) if o["kind"] in ("conv", "rms", "rms_silu")):
        name = op["name"]
        if op["kind"] == "conv":
            if op["source"] == "avgdown":
                w = avgdown_weights(op["ci"], op["co"], op["factor_t"])
                b = np.zeros(op["co"], np.float32)
            elif op["source"] == "latent_out":
                w, b = latent_out_weights(get, config)
                used.update(("encoder.conv_out.weight", "encoder.conv_out.bias",
                             "quant_conv.weight", "quant_conv.bias"))
            else:
                w, b = get(name + ".weight"), get(name + ".bias")
                used.update((name + ".weight", name + ".bias"))
                if op["source"] == "s2d":
                    w = s2d_weights(w)
            taps = 9 if op["taps"] == 3 else 1
            spec = resolve_spec(conv_spec(op), taps)
            blob = pack_conv(w, b, taps=taps, cin=op["Cin"])
            if blob.nbytes != spec["sizes"]["W"] * 9:
                raise ValueError(f"packed size mismatch for {name}")
            meta = {"format": "dit_conv_v8bfp16ebs8", "source": op["source"],
                    "logical": [op["ci"], op["co"]], "physical": [op["Cin"], op["Cout"]],
                    "taps": op["taps"]}
        else:
            gamma = get(name + ".gamma").reshape(-1).astype(np.float32)
            used.add(name + ".gamma")
            padded = np.zeros(op["C"], bfloat16)
            padded[:op["c"]] = (gamma * np.sqrt(op["c"] / op["C"])).astype(bfloat16)
            blob = padded.view(np.uint8)
            meta = {"format": "bf16_rms_gamma_padded", "channels": op["c"],
                    "physical_channels": op["C"], "scale": "sqrt(channels / physical_channels)"}
        target = out / f"{index:03d}.bin"
        blob.tofile(target)
        records[name] = {"file": target.name, "bytes": target.stat().st_size,
                         "sha256": digest(target), **meta}
    expected = {k for k in header if k.startswith("encoder.") or k.startswith("quant_conv.")}
    inactive = sorted(expected - used)
    if not all(".downsampler.time_conv." in k for k in inactive):
        raise ValueError(f"unplanned Qwen VAE encoder tensors: {inactive}")
    result = {"model": MODEL, "scope": "first-frame VAE encoder parameters",
              "resolution": resolution, "complete": True, "records": records,
              "active_checkpoint_tensors": len(used), "inactive_first_frame_tensors": inactive}
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"packed {len(records)} Qwen VAE encoder parameter groups ({len(used)} tensors, "
          f"{len(inactive)} inactive time_conv tensors) -> {out}")
    return result


# --------------------------------------------------------------------------- export

def export(out: Path, resolution: int, attention: Path | None, jobs: int = 2) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from open_kernels.designs.dit_conv.dit_conv import resolve_spec, tiles
    from open_kernels.designs.qwen_vae_ew.qwen_vae_ew import check_spec
    conv, ew = streams(resolution)
    for name, item in conv.items():
        tiles(resolve_spec(item["spec"], item["taps"]))
    for name, spec in ew.items():
        if why := check_spec(spec):
            raise ValueError(f"{name}: {why}")
    out.mkdir(parents=True, exist_ok=True)

    def build(item):
        name, (design, env) = item
        target = out / name
        if (target / "final.xclbin").is_file() and (target / "insts.bin").is_file():
            return name
        e = os.environ.copy()
        e.update(env)
        r = subprocess.run((sys.executable, str(ROOT / "build_design.py"), str(ROOT / design),
                            str(target)), env=e, capture_output=True, text=True)
        if r.returncode or "BUILD_OK" not in r.stdout:
            raise RuntimeError(f"{name} build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        print(f"built {name}", flush=True)
        return name

    jobs_ = {n: ("designs/dit_conv/dit_conv.py",
                 {"DC_TAPS": str(i["taps"]), "DC_SPEC": json.dumps(i["spec"])}) for n, i in conv.items()}
    jobs_ |= {n: ("designs/qwen_vae_ew/qwen_vae_ew.py", {"QVE_SPEC": json.dumps(s)}) for n, s in ew.items()}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(build, jobs_.items()))
    files = {}
    for name in jobs_:
        for part in ("final.xclbin", "insts.bin"):
            p = out / name / part
            files[f"{name}/{part}"] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    (out / "manifest.json").write_text(json.dumps({
        "model": MODEL, "scope": "first-frame VAE encoder kernels", "resolution": resolution,
        "complete": True, "conv": conv, "ew": ew,
        "attention": str(attention) if attention else None, "files": files}, indent=2) + "\n")
    print(f"exported {len(conv)} convolution and {len(ew)} elementwise Qwen VAE encoder streams")


# --------------------------------------------------------------------------- run

def space_to_depth(x: np.ndarray, c: int) -> np.ndarray:
    """[H, W, >=c] -> [H/2, W/2, 4c]: channel (2p + q)c + k holds x[2i + p, 2j + q, k]."""
    h, w = x.shape[:2]
    v = x[..., :c].reshape(h // 2, 2, w // 2, 2, c)
    return np.ascontiguousarray(v.transpose(0, 2, 1, 3, 4).reshape(h // 2, w // 2, 4 * c))


class NativeEncoder:
    def __init__(self, kernels: Path, weights: Path, attention: Path, scratch: Path):
        from open_kernels.qwen21_vae_run import FileNpu
        self.kernels, self.weight_dir, self.attention_kernel = kernels, weights, attention
        self.km = json.loads((kernels / "manifest.json").read_text())
        self.wm = json.loads((weights / "manifest.json").read_text())
        if not (self.km.get("complete") and self.wm.get("complete")
                and self.km["model"] == self.wm["model"] == MODEL
                and self.km["resolution"] == self.wm["resolution"]):
            raise ValueError("incomplete or mismatched Qwen VAE encoder package")
        self.resolution = self.km["resolution"]
        self.npu = FileNpu(scratch)
        self.streams = {}

    def stream(self, name: str, directory: Path):
        from open_kernels.qwen21_vae_run import FileStream
        if name not in self.streams:
            if not (directory / "final.xclbin").is_file():
                raise ValueError(f"missing native kernel {directory}")
            self.streams[name] = FileStream(self.npu, name, directory)
        return self.streams[name]

    def weight(self, name: str):
        record = self.wm["records"][name]
        buf = self.npu.buf(f"w:{name}", record["bytes"])
        buf.load(self.weight_dir / record["file"])
        return buf

    def convolution(self, op: dict, x: np.ndarray) -> np.ndarray:
        from open_kernels.designs.dit_conv.dit_conv import resolve_spec
        name = conv_stream(op)
        item = self.km["conv"][name]
        s = resolve_spec(item["spec"], item["taps"])
        h, ci, co = s["H"], s["Cin"], s["Cout"]
        data = np.zeros(s["sizes"]["X"], bfloat16)
        view = data[:(h + 2) * s["x"]["pitch"] * ci].reshape(h + 2, s["x"]["pitch"], ci)
        n = min(ci, x.shape[2])
        view[1:h + 1, 1:h + 1, :n] = x[..., :n]
        xb = self.npu.buf(f"x:{op['name']}", data.nbytes)
        xb.write(data)
        yb = self.npu.buf(f"y:{op['name']}", s["sizes"]["Y"] * 2)
        yb.zero()
        self.stream(name, self.kernels / name).run(xb, self.weight(op["name"]), yb)
        y = yb.read()[:(h + 2) * s["y"]["pitch"] * co].reshape(h + 2, s["y"]["pitch"], co)
        return np.ascontiguousarray(y[1:h + 1, 1:h + 1])

    def elementwise(self, op: dict, x: np.ndarray, other: np.ndarray | None = None) -> np.ndarray:
        name = ew_stream(op)
        s = self.km["ew"][name]
        h, c = s["H"], s["C"]

        def bordered(v, role):
            out = np.zeros(s["sizes"][role], bfloat16)
            if v is not None:
                out[:(h + 2) * s["a"]["pitch"] * c].reshape(h + 2, s["a"]["pitch"], c)[
                    1:h + 1, 1:h + 1, :v.shape[2]] = v[..., :c]
            return out

        if s["op"] == "add":
            b, p = bordered(other, "B"), np.zeros(1, bfloat16)
        else:
            b = np.zeros(1, bfloat16)
            rec = self.wm["records"][op["name"]]
            gamma = np.fromfile(self.weight_dir / rec["file"], bfloat16)
            p = np.zeros((2, s["el"]), bfloat16)
            p[:, :c] = gamma
        bufs = []
        for role, value in (("a", bordered(x, "X")), ("b", b), ("p", p)):
            buf = self.npu.buf(f"{role}:{op['name']}", value.nbytes)
            buf.write(value)
            bufs.append(buf)
        yb = self.npu.buf(f"y:{op['name']}", s["sizes"]["Y"] * 2)
        yb.zero()
        self.stream(name, self.kernels / name).run(*bufs, yb)
        y = yb.read()[:(h + 2) * s["y"]["pitch"] * c].reshape(h + 2, s["y"]["pitch"], c)
        return np.ascontiguousarray(y[1:h + 1, 1:h + 1])

    def attention(self, qkv: np.ndarray) -> np.ndarray:
        h = qkv.shape[0]
        flat = np.ascontiguousarray(qkv[..., :3 * ATTN].reshape(h * h, 3 * ATTN))
        qb = self.npu.buf("attn-qkv", flat.nbytes)
        qb.write(flat)
        ob = self.npu.buf("attn-out", h * h * ATTN * 2)
        ob.zero()
        self.stream("attention", self.attention_kernel).run(qb, qb, qb, ob)
        return ob.read().reshape(h, h, ATTN)

    def encode(self, pixels: np.ndarray, out: Path | None = None, trace: bool = False,
               dump_ops: int = 0) -> np.ndarray:
        """pixels [R, R, 4] in [-1, 1] -> normalized latents [R/16, R/16, 64] (float32)."""
        r = self.resolution
        if pixels.shape != (r, r, 4):
            raise ValueError(f"Qwen VAE encoder input must be {r}x{r}x4")
        x = pixels.astype(bfloat16)
        saved: dict[str, np.ndarray] = {}
        pending_s2d: np.ndarray | None = None
        ops = plan(r)
        for index, op in enumerate(ops):
            kind, name = op["kind"], op["name"]
            if kind == "save":
                saved[name] = x
                continue
            if kind == "stage":
                if out is not None:
                    x[..., :op["c"]].astype("<f4").tofile(out / f"stage-{name}.f32")
                continue
            if kind == "s2d":
                src = x if op["source"] == "x" else saved[op["source"]]
                pending_s2d = space_to_depth(src, op["c"])
                continue
            if kind == "conv":
                if op["source"] in ("s2d", "avgdown"):
                    y = self.convolution(op, pending_s2d)
                    pending_s2d = None
                    if op["source"] == "avgdown":
                        saved[name] = y
                    else:
                        x = y
                elif name.endswith(".conv_shortcut"):
                    saved[name] = self.convolution(op, x)
                else:
                    x = self.convolution(op, x)
                if name == "encoder.conv_in" and out is not None:
                    x[..., :DIMS[0]].astype("<f4").tofile(out / "stage-conv_in.f32")
            elif kind in ("rms", "rms_silu"):
                x = self.elementwise(op, x)
            elif kind == "add":
                x = self.elementwise(op, x, saved.pop(op["other"]))
            elif kind == "attention":
                x = self.attention(x)
            else:
                raise ValueError(f"unhandled encoder op {kind}")
            if not np.isfinite(x.astype(np.float32)).all():
                raise ValueError(f"nonfinite Qwen VAE encoder output after {name}")
            if out is not None and index < dump_ops:          # debugging: every op's output
                y = saved.get(name, x)
                np.save(out / f"op-{index:03d}.npy", y.astype(np.float32))
            if trace:
                v = x.astype(np.float32)
                print(f"{index + 1:3d}/{len(ops)} {name}: {x.shape} rms {np.sqrt(np.mean(v * v)):.5g}",
                      flush=True)
        return x[..., :Z].astype(np.float32)


def load_pixels(image: Path, resolution: int) -> np.ndarray:
    from PIL import Image
    with Image.open(image) as source:
        img = source.convert("RGBA")
    if img.size != (resolution, resolution):
        raise ValueError(f"the native Qwen VAE encoder covers {resolution}x{resolution} condition images")
    return np.asarray(img, np.float32) / 127.5 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--resolution", type=int, default=512)
    p = sub.add_parser("pack")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--attention", type=Path)
    p.add_argument("--jobs", type=int, default=2)
    p = sub.add_parser("run")
    for name in ("kernels", "weights", "attention", "image", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--dump-ops", type=int, default=0, help="save the first N ops' outputs")
    args = parser.parse_args()
    if args.command == "plan":
        ops = plan(args.resolution)
        conv, ew = streams(args.resolution)
        print(json.dumps({"operations": len(ops),
                          "npu": sum(o["kind"] in ("conv", "rms", "rms_silu", "add", "attention") for o in ops),
                          "conv_streams": len(conv), "ew_streams": len(ew)}))
    elif args.command == "pack":
        pack(args.model, args.out, args.resolution)
    elif args.command == "export":
        export(args.out.resolve(), args.resolution, args.attention, args.jobs)
    else:
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        enc = NativeEncoder(args.kernels.resolve(), args.weights.resolve(),
                            args.attention.resolve(), out / "scratch")
        latents = enc.encode(load_pixels(args.image, enc.resolution), out, args.trace,
                             args.dump_ops)
        latents.astype("<f4").tofile(out / "latents.f32")
        latents.astype(bfloat16).tofile(out / "latents.bf16")
        (out / "run.json").write_text(json.dumps({
            "model": MODEL, "scope": "native NPU first-frame VAE encoder",
            "image": str(args.image), "resolution": enc.resolution,
            "latents_shape": list(latents.shape), "npu_runs": len(enc.npu.log),
            "npu_ms": sum(t for *_, t in enc.npu.log)}, indent=2) + "\n")
        import shutil
        shutil.rmtree(out / "scratch")
        print(f"Qwen VAE native encoder: {latents.shape}, {len(enc.npu.log)} NPU runs")


if __name__ == "__main__":
    main()
