"""Replay one Qwen Image 2.1 transformer block with the C++ NPU harness.

This is a real-weight numerical integration gate for 15 text and 1024 image
tokens at 512 pixels. Inputs and shared modulation rows may be supplied by a
later sampler. The host only pads and rearranges rows at the two attention
boundaries; every model operation executes in a native NPU stream.

--kernels is a JSON mapping from norm, res, qk, qkv, attn_out, mlp_in,
mlp_out, fa_text, fa_image to single-stream build directories. Each directory
contains final.xclbin and insts.bin, with the exact shapes in the image plan.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.image_safetensors import read_header
from open_kernels.image_weight_pack import source_paths
from open_kernels.qwen21_weight_layout import HIDDEN, HEAD_DIM

TEXT, IMAGE, CHUNK, JOINT = 15, 1024, 512, 1536
MLP = 12288
REQUIRED = {"norm", "res", "qk", "qkv", "attn_out", "mlp_in", "mlp_out",
            "fa_text", "fa_image"}


def bf(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(bfloat16)


def tensor(model: Path, name: str) -> np.ndarray:
    path = source_paths("qwen21", model)[name]
    entry = read_header(path)[name]
    if entry["dtype"] != "BF16":
        raise ValueError(f"{name} is not BF16")
    with path.open("rb") as source:
        offset = 8 + struct.unpack("<Q", source.read(8))[0] + entry["data_offsets"][0]
    return np.asarray(np.memmap(path, mode="r", dtype=bfloat16,
                                offset=offset, shape=tuple(entry["shape"]))).copy()


def params(*rows: np.ndarray) -> np.ndarray:
    zero = np.zeros(HIDDEN, bfloat16)
    return np.concatenate((zero, *(bf(row) for row in rows), zero))


def qk_params(wq: np.ndarray, wk: np.ndarray) -> np.ndarray:
    """4096-wide q/k weights plus compact three-axis theta=10000 RoPE tables."""
    frequencies = np.concatenate([10000.0 ** (-np.arange(0, dim, 2) / dim)
                                  for dim in (16, 56, 56)])
    tables = []
    for step in (1, 32):
        angle = np.arange(32)[:, None] * step * frequencies[None]
        tables.append(bf(np.stack((np.cos(angle), np.sin(angle)), -1)).reshape(-1))
    row = np.zeros(HIDDEN, bfloat16)
    row[:HEAD_DIM], row[HEAD_DIM:2 * HEAD_DIM] = bf(wq), bf(wk)
    return params(row, *tables)


def modulation(path: Path | None, rng: np.random.Generator) -> np.ndarray:
    if path is None:
        value = rng.normal(0, 0.1, (4, HIDDEN)).astype(np.float32)
        value[[1, 3]] += 0.8
        return bf(value)
    value = np.fromfile(path, dtype=bfloat16)
    if value.size != 4 * HIDDEN:
        raise ValueError(f"{path}: expected four 4096-wide BF16 rows")
    return value.reshape(4, HIDDEN)


class HarnessStage:
    def __init__(self, root: Path, name: str, kernel_dirs: dict[str, Path],
                 harness: Path):
        self.root = root / name
        self.root.mkdir(parents=True, exist_ok=False)
        self.harness = harness
        self.lines = ["device"]
        for key, directory in kernel_dirs.items():
            self.lines.extend((f"xclbin {key} {directory / 'final.xclbin'}",
                               f"kernelx {key} {key} {directory / 'insts.bin'}"))
        self.names = set()
        self.dumps = {}

    def buf(self, name: str, value: np.ndarray | None = None, *,
            shape: tuple[int, ...] | None = None, path: Path | None = None) -> str:
        if name in self.names:
            raise ValueError(f"duplicate harness buffer: {name}")
        self.names.add(name)
        if path is not None:
            size = path.stat().st_size
            self.lines.append(f"buf {name} {size} {path}")
        elif value is not None:
            value = np.ascontiguousarray(value, dtype=bfloat16)
            local = self.root / f"{name}.bf16"
            value.tofile(local)
            self.lines.append(f"buf {name} {value.nbytes} {local}")
        elif shape is not None:
            self.lines.append(f"buf {name} {int(np.prod(shape)) * 2}")
        else:
            raise ValueError("buffer needs a value, shape or path")
        return name

    def run(self, kernel: str, *names: str) -> None:
        if any(name not in self.names for name in names):
            raise ValueError(f"{kernel}: run uses an undeclared buffer")
        self.lines.append(f"run {kernel} {' '.join(names)}")

    def dump(self, name: str, shape: tuple[int, ...]) -> None:
        if name not in self.names:
            raise ValueError(f"unknown output buffer: {name}")
        path = self.root / f"{name}.out.bf16"
        self.lines.append(f"dump {name} {path} {int(np.prod(shape)) * 2}")
        self.dumps[name] = (path, shape)

    def execute(self) -> dict[str, np.ndarray]:
        config = self.root / "run.cfg"
        config.write_text("\n".join(self.lines + [""]))
        with (self.root / "harness.log").open("w") as log:
            subprocess.run((str(self.harness), str(config)), cwd=self.root,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return {name: np.fromfile(path, bfloat16).reshape(shape)
                for name, (path, shape) in self.dumps.items()}


class NativeBlock:
    def __init__(self, kernels: Path, weights: Path, model: Path,
                 layer: int, harness: Path, out: Path):
        named = json.loads(kernels.read_text())
        if set(named) != REQUIRED:
            raise ValueError(f"kernel manifest needs exactly {sorted(REQUIRED)}")
        self.kernels = {name: Path(path).resolve() for name, path in named.items()}
        for name, path in self.kernels.items():
            if not (path / "final.xclbin").is_file() or not (path / "insts.bin").is_file():
                raise ValueError(f"{name}: incomplete kernel directory {path}")
        packed = json.loads((weights / "packed_weights.json").read_text())
        if packed["model"] != "Qwen-Image-2.1":
            raise ValueError("wrong packed transformer family")
        self.weights = {}
        for name in ("qkv", "attn_out", "mlp_in", "mlp_out"):
            slot = packed["slots"][f"block.{layer}.{name}"]
            path = weights / slot["file"]
            if path.stat().st_size != slot["bytes"]:
                raise ValueError(f"{name}: packed weight byte count differs")
            self.weights[name] = path
        wq = tensor(model, f"transformer_blocks.{layer}.attn.norm_q.weight")
        wk = tensor(model, f"transformer_blocks.{layer}.attn.norm_k.weight")
        if wq.shape != wk.shape or wq.shape != (HEAD_DIM,):
            raise ValueError("Qwen q/k norm weights differ from the expected width")
        self.qk_params = qk_params(wq, wk)
        self.out, self.harness = out, harness

    def stage(self, name: str, *kernels: str) -> HarnessStage:
        return HarnessStage(self.out, name,
                            {key: self.kernels[key] for key in kernels}, self.harness)

    def run(self, text: np.ndarray, image: np.ndarray,
            text_mod: np.ndarray, image_mod: np.ndarray) -> dict[str, np.ndarray]:
        text_tokens = text.shape[0]
        if not 1 <= text_tokens <= 64 or text.shape != (text_tokens, HIDDEN) or image.shape != (IMAGE, HIDDEN):
            raise ValueError("Qwen block needs 1..64 text and 1024 image tokens")
        chunks = [np.zeros((CHUNK, HIDDEN), bfloat16) for _ in range(3)]
        chunks[0][:text_tokens] = text
        chunks[1][:], chunks[2][:] = image[:CHUNK], image[CHUNK:]
        mods = (text_mod, image_mod, image_mod)
        s = self.stage("01-qkv", "norm", "qkv")
        s.buf("w_qkv", path=self.weights["qkv"])
        for i, (x, mod) in enumerate(zip(chunks, mods)):
            s.buf(f"x{i}", x)
            s.buf(f"p{i}", params(np.zeros(HIDDEN), mod[0]))
            s.buf(f"dummy{i}", shape=(HIDDEN,))
            s.buf(f"norm{i}", shape=(CHUNK, HIDDEN))
            s.buf(f"qkv{i}", shape=(CHUNK, 3 * HIDDEN))
            s.run("norm", f"x{i}", f"dummy{i}", f"p{i}", f"norm{i}", f"dummy{i}")
            s.run("qkv", f"norm{i}", "w_qkv", f"qkv{i}")
            s.dump(f"qkv{i}", (CHUNK, 3 * HIDDEN))
        projected = s.execute()
        joint = np.zeros((JOINT, 3 * HIDDEN), bfloat16)
        joint[:text_tokens] = projected["qkv0"][:text_tokens]
        joint[text_tokens:text_tokens + IMAGE] = np.concatenate((projected["qkv1"], projected["qkv2"]))

        s = self.stage("02-qk", "qk")
        s.buf("qkv", joint)
        s.buf("p", self.qk_params)
        s.buf("q", shape=(JOINT, HIDDEN))
        s.buf("k", shape=(JOINT, HIDDEN))
        s.run("qk", "qkv", "qkv", "p", "q", "k")
        s.dump("q", (JOINT, HIDDEN))
        s.dump("k", (JOINT, HIDDEN))
        rotated = s.execute()
        q, k, v = rotated["q"], rotated["k"], joint[:, 2 * HIDDEN:]

        s = self.stage("03-attention", "fa_text", "fa_image")
        for suffix, name, qs, ks, vs in (
            ("text", "fa_text", q[:CHUNK], k[:64], v[:64]),
            ("image", "fa_image", q[text_tokens:text_tokens + IMAGE], k[:1088], v[:1088]),
        ):
            s.buf(f"q_{suffix}", qs)
            s.buf(f"k_{suffix}", ks)
            s.buf(f"v_{suffix}", vs)
            s.buf(f"o_{suffix}", shape=qs.shape)
            s.run(name, f"q_{suffix}", f"k_{suffix}", f"v_{suffix}", f"o_{suffix}")
            s.dump(f"o_{suffix}", qs.shape)
        outputs = s.execute()
        attention = np.zeros((JOINT, HIDDEN), bfloat16)
        attention[:text_tokens] = outputs["o_text"][:text_tokens]
        attention[text_tokens:text_tokens + IMAGE] = outputs["o_image"]
        attn_chunks = [np.zeros((CHUNK, HIDDEN), bfloat16) for _ in range(3)]
        attn_chunks[0][:text_tokens] = attention[:text_tokens]
        attn_chunks[1][:], attn_chunks[2][:] = (attention[text_tokens:text_tokens + CHUNK],
                                                 attention[text_tokens + CHUNK:text_tokens + IMAGE])

        s = self.stage("04-residual-mlp", "attn_out", "res", "mlp_in", "mlp_out")
        for name in ("attn_out", "mlp_in", "mlp_out"):
            s.buf(f"w_{name}", path=self.weights[name])
        for i, (x, a, mod) in enumerate(zip(chunks, attn_chunks, mods)):
            s.buf(f"x{i}", x)
            s.buf(f"attention{i}", a)
            s.buf(f"p_first{i}", params(mod[1], np.zeros(HIDDEN), mod[2]))
            s.buf(f"p_second{i}", params(mod[3], np.zeros(HIDDEN), np.zeros(HIDDEN)))
            for name, shape in (("aout", (CHUNK, HIDDEN)), ("norm2", (CHUNK, HIDDEN)),
                                ("x1", (CHUNK, HIDDEN)), ("mlp", (CHUNK, 2 * MLP)),
                                ("mout", (CHUNK, HIDDEN)), ("unused", (CHUNK, HIDDEN)),
                                ("x2", (CHUNK, HIDDEN))):
                s.buf(f"{name}{i}", shape=shape)
            s.run("attn_out", f"attention{i}", "w_attn_out", f"aout{i}")
            s.run("res", f"x{i}", f"aout{i}", f"p_first{i}", f"norm2{i}", f"x1{i}")
            s.run("mlp_in", f"norm2{i}", "w_mlp_in", f"mlp{i}")
            s.run("mlp_out", f"mlp{i}", "w_mlp_out", f"mout{i}")
            s.run("res", f"x1{i}", f"mout{i}", f"p_second{i}", f"unused{i}", f"x2{i}")
            s.dump(f"x2{i}", (CHUNK, HIDDEN))
        result = s.execute()
        return {"text": result["x20"][:text_tokens],
                "image": np.concatenate((result["x21"], result["x22"])),
                "attention": attention[:text_tokens + IMAGE],
                "q": q[:text_tokens + IMAGE], "k": k[:text_tokens + IMAGE]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--harness", type=Path, default=(
        Path(__file__).resolve().parents[1] / "target/npu-dev/harness-build/run_kernel"))
    parser.add_argument("--text-x", type=Path)
    parser.add_argument("--image-x", type=Path)
    parser.add_argument("--text-mod", type=Path)
    parser.add_argument("--image-mod", type=Path)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    if not 0 <= args.layer < 32:
        parser.error("layer must be in 0..31")
    if bool(args.text_x) != bool(args.image_x):
        parser.error("supply both text and image hidden buffers")
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    text = (np.fromfile(args.text_x, bfloat16).reshape(TEXT, HIDDEN) if args.text_x else
            bf(rng.normal(0, 0.6, (TEXT, HIDDEN))))
    image = (np.fromfile(args.image_x, bfloat16).reshape(IMAGE, HIDDEN) if args.image_x else
             bf(rng.normal(0, 0.6, (IMAGE, HIDDEN))))
    text_mod, image_mod = modulation(args.text_mod, rng), modulation(args.image_mod, rng)
    for name, data in (("input-text", text), ("input-image", image),
                       ("text-mod", text_mod), ("image-mod", image_mod)):
        data.tofile(args.out / f"{name}.bf16")
    runner = NativeBlock(args.kernels.resolve(), args.weights.resolve(),
                         args.model.resolve(), args.layer, args.harness.resolve(),
                         args.out.resolve())
    outputs = runner.run(text, image, text_mod, image_mod)
    for name, data in outputs.items():
        if not np.isfinite(data.astype(np.float32)).all():
            raise ValueError(f"{name}: NPU output contains NaN or Inf")
        data.tofile(args.out / f"native-{name}.bf16")
        print(f"{name}: {data.shape} RMS {np.sqrt(np.mean(data.astype(np.float32)**2)):.6g}",
              flush=True)
    (args.out / "fixture.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "layer": args.layer,
        "text_tokens": TEXT, "image_tokens": IMAGE, "seed": args.seed,
        "kernels": str(args.kernels.resolve()), "weights": str(args.weights.resolve()),
        "operations": 24}, indent=2) + "\n")
    print("NPU streams completed: 24")


if __name__ == "__main__":
    main()
