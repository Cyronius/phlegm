"""Run Qwen Image 2.1's first-frame VAE decoder with native NPU kernels.

Convolution, RMSNorm/SiLU, attention, and residual addition run on the NPU.
The host performs only layout copies for the different native tensor pitches,
the published DupUp3D first-frame rearrangement, and PNG serialization.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.designs.dit_conv.dit_conv import resolve_spec
from open_kernels.moss_vae_run import write_png
from open_kernels.qwen21_vae_pack import physical
from open_kernels.qwen21_vae_plan import plan
from open_kernels.qwen21_vae_shortcut import first_frame


class FileBuf:
    def __init__(self, root: Path, name: str, size: int):
        self.path = root / f"{name.split(':')[0]}.bin"
        self.size = size
        self.initialized = False
        self.bo = self

    def write(self, values: np.ndarray) -> None:
        if values.nbytes > self.size:
            raise ValueError("input exceeds harness buffer")
        values.tofile(self.path)
        self.initialized = True

    def load(self, path: Path) -> None:
        self.path = path
        self.initialized = True

    def zero(self) -> None:
        self.initialized = False

    def read(self, dtype=bfloat16) -> np.ndarray:
        return np.fromfile(self.path, dtype)


class FileNpu:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.harness = Path(__file__).resolve().parents[1] / "target/npu-dev/harness-build/run_kernel"
        if not self.harness.is_file():
            raise ValueError(f"missing native NPU harness: {self.harness}")
        self.log: list[tuple[str, str, float]] = []

    def buf(self, name: str, size: int) -> FileBuf:
        return FileBuf(self.root, name, size)


class FileStream:
    def __init__(self, npu: FileNpu, name: str, directory: Path):
        self.npu, self.name, self.directory = npu, name, directory

    def run(self, *buffers: FileBuf) -> None:
        names = [f"a{i}" for i in range(len(buffers))]
        output = buffers[-1]
        lines = ["device", f"xclbin G {self.directory / 'final.xclbin'}",
                 f"kernelx k G {self.directory / 'insts.bin'}"]
        for name, buf in zip(names, buffers):
            lines.append(f"buf {name} {buf.size}" +
                         (f" {buf.path}" if buf.initialized else ""))
        lines.extend((f"run k {' '.join(names)}",
                      f"dump {names[-1]} {output.path} {output.size}", ""))
        config = self.npu.root / "run.cfg"
        config.write_text("\n".join(lines))
        result = subprocess.run((str(self.npu.harness), str(config)),
                                cwd=self.npu.root, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Qwen VAE NPU {self.name} failed:\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        timing = re.search(r"run k \[\d+ bufs\] -> state 4 \(([0-9.]+) ms\)", result.stdout)
        if timing is None:
            raise RuntimeError(f"Qwen VAE NPU {self.name} did not complete: {result.stdout[-2000:]}")
        self.npu.log.append(("vae", self.name, float(timing.group(1))))


class NativeDecoder:
    def __init__(self, model: Path, conv_kernels: Path, ew_kernels: Path,
                 attention_kernel: Path, weights: Path, scratch: Path):
        self.model = model
        self.plan = plan(model)
        if self.plan["resolution"] != 512:
            raise ValueError("this decoder requires the compiled 512-pixel streams")
        self.conv = json.loads((conv_kernels / "manifest.json").read_text())
        self.ew = json.loads((ew_kernels / "manifest.json").read_text())
        self.weights = json.loads((weights / "manifest.json").read_text())
        if not all(m.get("complete") and m.get("model") == "Qwen/Qwen-Image-2.1"
                   for m in (self.conv, self.ew, self.weights)):
            raise ValueError("incomplete or wrong-family Qwen VAE package")
        if not self.weights["records"]["post_quant_conv"].get("latents_mean_std_fused"):
            raise ValueError("the VAE package must fuse latent mean/std into post_quant_conv")
        self.conv_kernels, self.ew_kernels = conv_kernels, ew_kernels
        self.attention_kernel, self.weight_dir = attention_kernel, weights
        self.npu = FileNpu(scratch)
        self.streams = {}
        self.weight_bos = {}

    def stream(self, name: str, directory: Path) -> FileStream:
        if name not in self.streams:
            if not (directory / "final.xclbin").is_file() or not (directory / "insts.bin").is_file():
                raise ValueError(f"missing native kernel {directory}")
            self.streams[name] = FileStream(self.npu, name, directory)
        return self.streams[name]

    def weight(self, name: str):
        if name not in self.weight_bos:
            record = self.weights["records"][name]
            path = self.weight_dir / record["file"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"wrong packed weight length for {name}")
            buf = self.npu.buf(f"weight:{name}", record["bytes"])
            buf.load(path)
            self.weight_bos[name] = buf
        return self.weight_bos[name]

    def convolution(self, name: str, x: np.ndarray) -> np.ndarray:
        stream_name = self.conv["operations"][name]
        item = self.conv["streams"][stream_name]
        s = resolve_spec(item["spec"], item["taps"])
        h, w, ci, co = (s[k] for k in ("H", "W", "Cin", "Cout"))
        if x.shape[:2] != (h, w) or x.shape[2] < ci:
            raise ValueError(f"{name} input shape {x.shape} != {(h, w, ci)}")
        if x.shape[2] != ci:
            x = np.ascontiguousarray(x[..., :ci])
        input_data = np.zeros(s["sizes"]["X"], bfloat16)
        xview = input_data[:(h + 2) * s["x"]["pitch"] * ci].reshape(
            h + 2, s["x"]["pitch"], ci)
        xview[1:h + 1, 1:w + 1] = x
        oh, ow = h * (2 if s["up"] else 1), w * (2 if s["up"] else 1)
        xbo = self.npu.buf(f"x:{name}", input_data.nbytes)
        xbo.write(input_data)
        ybo = self.npu.buf(f"y:{name}", s["sizes"]["Y"] * 2)
        ybo.zero()
        kernel = self.stream(stream_name, self.conv_kernels / stream_name)
        kernel.run(xbo.bo, self.weight(name).bo, ybo.bo)
        output = ybo.read(dtype=bfloat16)
        yview = output[:(oh + 2) * s["y"]["pitch"] * co].reshape(
            oh + 2, s["y"]["pitch"], co)
        return np.ascontiguousarray(yview[1:oh + 1, 1:ow + 1])

    def elementwise(self, name: str, x: np.ndarray,
                    other: np.ndarray | None = None) -> np.ndarray:
        stream_name = self.ew["operations"][name]
        s = self.ew["streams"][stream_name]
        h, c = s["H"], s["C"]
        if x.shape != (h, h, c):
            raise ValueError(f"{name} input shape {x.shape} != {(h, h, c)}")
        def border(data: np.ndarray | None, role: str) -> np.ndarray:
            size = s["sizes"][role]
            result = np.zeros(size, bfloat16)
            if data is not None:
                view = result[:(h + 2) * s["a"]["pitch"] * c].reshape(
                    h + 2, s["a"]["pitch"], c)
                view[1:h + 1, 1:h + 1] = data
            return result

        if s["op"] == "add":
            if other is None or other.shape != x.shape:
                raise ValueError(f"{name} needs matching residual input")
            b = border(other, "B")
            p = np.zeros(1, bfloat16)
        else:
            if other is not None:
                raise ValueError(f"{name} has an unexpected second input")
            b = np.zeros(1, bfloat16)
            gamma = np.fromfile(self.weight_dir / self.weights["records"][name]["file"],
                                bfloat16)
            if gamma.size != c:
                raise ValueError(f"{name} packed gamma width changed")
            p = np.zeros((2, s["el"]), bfloat16)
            p[:, :c] = gamma
        a = border(x, "X")
        abo = self.npu.buf(f"a:{name}", a.nbytes)
        bbo = self.npu.buf(f"b:{name}", b.nbytes)
        pbo = self.npu.buf(f"p:{name}", p.nbytes)
        ybo = self.npu.buf(f"y:{name}", s["sizes"]["Y"] * 2)
        abo.write(a)
        bbo.write(b)
        pbo.write(p)
        ybo.zero()
        kernel = self.stream(stream_name, self.ew_kernels / stream_name)
        kernel.run(abo.bo, bbo.bo, pbo.bo, ybo.bo)
        output = ybo.read(dtype=bfloat16)
        view = output[:(h + 2) * s["y"]["pitch"] * c].reshape(
            h + 2, s["y"]["pitch"], c)
        return np.ascontiguousarray(view[1:h + 1, 1:h + 1])

    def attention(self, x: np.ndarray) -> np.ndarray:
        if x.shape != (32, 32, 3456):
            raise ValueError("Qwen VAE attention requires the packed 32² QKV projection")
        qkv = np.ascontiguousarray(x.reshape(1024, 3456))
        qbo = self.npu.buf("vae-qkv", qkv.nbytes)
        obo = self.npu.buf("vae-attention-out", 1024 * 1152 * 2)
        qbo.write(qkv)
        obo.zero()
        kernel = self.stream("attention", self.attention_kernel)
        kernel.run(qbo.bo, qbo.bo, qbo.bo, obo.bo)
        return obo.read(dtype=bfloat16).reshape(32, 32, 1152)

    def decode(self, latents: np.ndarray, *, max_ops: int | None = None,
               trace: bool = False) -> np.ndarray:
        if latents.shape != (32, 32, 64):
            raise ValueError("Qwen VAE latents must be 32x32x64")
        x = np.ascontiguousarray(latents, dtype=bfloat16)
        residuals: dict[str, np.ndarray] = {}
        stage_inputs: dict[int, np.ndarray] = {}
        attn_identity: np.ndarray | None = None
        pending_shortcut: np.ndarray | None = None
        for index, op in enumerate(self.plan["operations"]):
            if max_ops is not None and index >= max_ops:
                break
            kind, name = op["kind"], op["name"]
            if name.startswith("decoder.up_blocks.") and ".resnets.0." in name:
                stage = int(name.split(".")[2])
                stage_inputs.setdefault(stage, x)
            if kind == "latent_denormalize":
                pass  # folded into the native post_quant_conv weight and bias
            elif kind == "conv":
                if name.endswith(".conv_shortcut"):
                    residuals[name.rsplit(".conv_shortcut", 1)[0]] = self.convolution(name, x)
                else:
                    x = self.convolution(name, x)
            elif kind == "rmsnorm":
                if name.endswith(".norm1"):
                    base = name.rsplit(".norm1", 1)[0]
                    residuals.setdefault(base, x)
                if name == "decoder.mid_block.attentions.0.norm":
                    attn_identity = x
                x = self.elementwise(name, x)
            elif kind == "silu" or kind == "nearest2x":
                pass  # SiLU is fused with RMSNorm; upsample is fused with convolution
            elif kind == "attention":
                x = self.attention(x)
            elif kind == "dup_up3d_first_frame":
                stage = int(name.split(".")[2])
                source = stage_inputs[stage][..., :op["input_channels"]]
                shortcut = first_frame(source, op["channels"], op["temporal_factor"])
                c = physical(op["channels"], False)
                pending_shortcut = np.zeros((op["height"], op["width"], c), bfloat16)
                pending_shortcut[..., :op["channels"]] = shortcut
            elif kind == "add":
                if name == "decoder.mid_block.attentions.0.residual":
                    if attn_identity is None:
                        raise ValueError("missing attention residual input")
                    other, attn_identity = attn_identity, None
                elif name.endswith(".upsample_residual"):
                    if pending_shortcut is None:
                        raise ValueError("missing DupUp3D shortcut")
                    other, pending_shortcut = pending_shortcut, None
                else:
                    base = name.rsplit(".residual", 1)[0]
                    other = residuals.pop(base)
                x = self.elementwise(name, x, other)
            elif kind == "clamp":
                # Pixel clamp and serialization are image postprocessing on the host.
                x = np.clip(x[..., :4].astype(np.float32), -1.0, 1.0)
            else:
                raise ValueError(f"unhandled Qwen VAE operation: {kind} {name}")
            if trace:
                values = x.astype(np.float32)
                print(f"{index + 1:3d}/{len(self.plan['operations'])} {name}: "
                      f"shape={x.shape} rms={np.sqrt(np.mean(values * values)):.6g}",
                      flush=True)
            if not np.isfinite(x).all():
                raise ValueError(f"nonfinite Qwen VAE output after {name}")
        return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conv-kernels", type=Path, required=True)
    parser.add_argument("--ew-kernels", type=Path, required=True)
    parser.add_argument("--attention-kernel", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-ops", type=int)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    latents = np.fromfile(args.latents, bfloat16)
    if latents.size != 32 * 32 * 64:
        raise ValueError("latent file must hold 1024x64 BF16 values")
    decoder = NativeDecoder(args.model, args.conv_kernels, args.ew_kernels,
                            args.attention_kernel, args.weights, args.out / "scratch")
    output = decoder.decode(latents.reshape(32, 32, 64),
                            max_ops=args.max_ops, trace=args.trace)
    output_name = "output.f32" if output.dtype == np.float32 else "output.bf16"
    output.tofile(args.out / output_name)
    if args.max_ops is None:
        pixels = np.clip((output.astype(np.float32) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        write_png(args.out / "native-qwen21.png", pixels)
    (args.out / "run.json").write_text(json.dumps({
        "model": "Qwen/Qwen-Image-2.1", "operations": args.max_ops or 150,
        "output_shape": list(output.shape), "npu_runs": len(decoder.npu.log),
        "npu_ms": sum(item[2] for item in decoder.npu.log),
        "output_file": output_name,
    }, indent=2) + "\n")
    print(f"Qwen VAE native output shape={output.shape}, NPU runs={len(decoder.npu.log)}")


if __name__ == "__main__":
    main()
