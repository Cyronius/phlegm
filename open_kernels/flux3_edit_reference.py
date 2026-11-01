"""Published FLUX 3 reference checks for native one-reference editing.

Run with the reference virtual environment (PyTorch, transformers), not the NPU
environment. Subcommands:

  noise      the published sampler's BF16-rounded starting noise for a seed
  generate   unchanged published ``Pipeline.generate`` with one reference
             (the ``pup=False`` path of ``flux3.generate``), on the CPU
  onepass    one published ``DiT.forward`` on a native step's own inputs,
             compared stage by stage with that step's native NPU outputs
  pixels     MAE / PSNR between two RGB images

None of this runs on the NPU; it is the oracle for the native route.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GRID = 32
TOKENS = GRID * GRID
EL = 3072


def flux3(reference: Path):
    sys.path.insert(0, str(reference.resolve() / "src"))
    import flux3.dit as dit
    import flux3.pipeline as pipeline
    import flux3.text_encoder as text_encoder
    from transformers import AutoTokenizer

    class TextOnlyProcessor:
        """The text encoder uses its processor only for the chat template and the
        tokenizer. transformers' Qwen3-VL processor also builds a video processor,
        which needs torchvision (absent from this CPU environment), so load the
        checkpoint's own tokenizer, whose chat template the processor applies."""

        def __init__(self, path):
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

        @classmethod
        def from_pretrained(cls, path, **_):
            return cls(path)

        def apply_chat_template(self, *args, **kwargs):
            return self.tokenizer.apply_chat_template(*args, **kwargs)

    text_encoder.AutoProcessor = TextOnlyProcessor
    return dit, pipeline


def metrics(native: np.ndarray, expected: np.ndarray) -> dict:
    native, expected = native.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = native - expected
    return {"relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(expected)),
            "cosine": float(native @ expected / (np.linalg.norm(native) * np.linalg.norm(expected))),
            "max_abs": float(np.max(np.abs(delta)))}


def bf16_rows(path: Path, rows: int, width: int):
    import torch
    values = np.fromfile(path, "<u2")
    if values.size < rows * width:
        raise ValueError(f"{path}: fewer than {rows} x {width} BF16 values")
    return torch.from_numpy(values[:rows * width].reshape(rows, width).copy()).view(torch.bfloat16)


def noise(args) -> None:
    import torch
    tokens = (args.resolution // 16) ** 2
    value = torch.randn((tokens, 128), generator=torch.Generator().manual_seed(args.seed))
    value = value.to(torch.bfloat16).float().numpy()
    value.astype("<f4").tofile(args.out)
    print(f"published FLUX 3 noise seed {args.seed} -> {args.out}")


def generate(args) -> None:
    import torch
    from PIL import Image
    dit, pipeline = flux3(args.reference)
    torch.set_num_threads(args.threads)
    weights = args.weights.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pipe = pipeline.Pipeline({"distilled": str(weights)}, str(weights / "text_encoder"),
                             str(weights / "ae.safetensors"), device="cpu")
    references = []
    if args.image:
        with Image.open(args.image) as source:
            prepared, crop = pipeline.prepare_reference(source)
        if prepared.size != (args.resolution,) * 2 or crop is not None:
            raise ValueError("this oracle needs a reference that prepares to the output square")
        references = [prepared]
    model = pipe.dits["distilled"]
    captures: dict[str, list[np.ndarray]] = {"cond_early": [], "shared": [], "velocity": []}
    model.content_mode_blocks["image_cond"][4].register_forward_hook(
        lambda _m, _i, o: captures["cond_early"].append(o.detach().float().numpy()))
    model.single_blocks[27].register_forward_hook(
        lambda _m, _i, o: captures["shared"].append(o.detach().float().numpy()))
    model.final_layer["image"].register_forward_hook(
        lambda _m, _i, o: captures["velocity"].append(o.detach().float().numpy()))
    if references:
        tokens, _ = pipe.encode_references(references)
        tokens.float().numpy().astype("<f4").tofile(out / "reference-tokens.f32")
    text, _ = pipe.text_encoder.encode(args.prompt)
    text.view(torch.int16).numpy().astype("<i2").tofile(out / "text-tokens.bf16")
    # The published sampler autocasts to BF16 on CUDA; on the CPU the same
    # autocast region has to be CPU autocast or BF16 weights meet FP32 latents.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        image = pipe.generate("distilled", args.prompt, references, args.resolution,
                              args.resolution, args.steps, None, args.seed, args.alpha)
    image.save(out / "published.png")
    for name, values in captures.items():
        if values:
            np.stack(values).astype("<f4").tofile(out / f"{name}.f32")
    (out / "fixture.json").write_text(json.dumps({
        "model": "black-forest-labs/flux-3-image-klein",
        "scope": "published CPU Pipeline.generate, pup=False, "
                 + ("one reference" if references else "text to image"),
        "prompt": args.prompt, "image": str(args.image), "steps": args.steps,
        "resolution": args.resolution,
        "seed": args.seed, "alpha": args.alpha, "text_length": int(text.shape[0]),
        "captured_steps": len(captures["velocity"]),
    }, indent=2) + "\n")
    print(f"published FLUX 3 edit -> {out / 'published.png'}")


def onepass(args) -> None:
    import torch
    dit, _ = flux3(args.reference)
    torch.set_num_threads(args.threads)
    from safetensors.torch import load_file
    step = args.step_dir.resolve()
    meta = json.loads((step / "conditioning" / "fixture.json").read_text())
    text_length = args.text_length
    GRID = args.resolution // 16
    TOKENS = GRID * GRID
    model = dit.DiT(flow_map=True).to(torch.bfloat16).eval()
    model.load_state_dict(load_file(str(args.weights.resolve() / "model.safetensors")), strict=True)
    captures = {}
    model.content_mode_blocks["image"][4].register_forward_hook(
        lambda _m, _i, o: captures.__setitem__("image_early", o.detach().float().numpy()))
    model.content_mode_blocks["image_cond"][4].register_forward_hook(
        lambda _m, _i, o: captures.__setitem__("cond_early", o.detach().float().numpy()))
    model.txt_mode_blocks[4].register_forward_hook(
        lambda _m, _i, o: captures.__setitem__("text_early", o.detach().float().numpy()))
    model.single_blocks[27].register_forward_hook(
        lambda _m, _i, o: captures.__setitem__("shared", o.detach().float().numpy()))
    image = bf16_rows(step / "latent-input.bf16", TOKENS, 128)
    refs = int(args.reference_tokens is not None)
    cond = (bf16_rows(args.reference_tokens, TOKENS, 128) if refs
            else torch.zeros((0, 128), dtype=torch.bfloat16))
    text = bf16_rows(args.text_tokens, text_length, 20480)
    p = torch.arange(TOKENS, dtype=torch.float32)
    zeros = torch.zeros_like(p)
    image_ids = torch.stack((zeros, (p // GRID) * 0.5, (p % GRID) * 0.5, zeros), -1)
    cond_ids = image_ids.clone()[:TOKENS * refs]
    cond_ids[:, 3] = -1
    t = torch.arange(text_length, dtype=torch.float32)
    tz = torch.zeros_like(t)
    text_ids = torch.stack((tz, tz, tz, t), -1)
    packed = dit.Packed(
        data={"image": image, "image_cond": cond, "text": text},
        ids={"image": image_ids, "image_cond": cond_ids, "text": text_ids},
        timesteps={"image": torch.full((TOKENS,), meta["image_time"], dtype=torch.bfloat16),
                   "image_cond": torch.zeros(TOKENS * refs, dtype=torch.bfloat16),
                   "text": torch.zeros(text_length, dtype=torch.bfloat16)},
        mode_indices={"image": [(0, TOKENS)], "image_cond": [(0, TOKENS * refs)],
                      "text": [(0, text_length)]})
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        velocity = model(packed, timesteps_ref=torch.tensor([meta["target_time"]])).float().numpy()
    native_rows = lambda path, rows: bf16_rows(path, rows, EL).float().numpy()
    image_rows = TOKENS * (1 + refs)
    shared = native_rows(step / "shared" / "native-x.bf16", text_length + image_rows)
    reference_shared = captures["shared"]  # published order: image, image_cond, text
    shared_ref = np.concatenate((reference_shared[image_rows:], reference_shared[:image_rows]))
    native_velocity = bf16_rows(step / "final" / "native-output.bf16", TOKENS, 1024).float().numpy()[:, :128]
    report = {
        "step_dir": str(step), "image_time": meta["image_time"], "target_time": meta["target_time"],
        "text_length": text_length,
        "image_early": metrics(native_rows(step / "image-early" / "native-x.bf16", TOKENS),
                               captures["image_early"]),
        **({"cond_early": metrics(native_rows(step / "cond-early" / "native-x.bf16", TOKENS),
                                  captures["cond_early"])} if refs else {}),
        "text_early": metrics(native_rows(step / "text-early" / "native-x.bf16", text_length),
                              captures["text_early"]),
        "shared": metrics(shared, shared_ref),
        "shared_image_rows": metrics(shared[text_length:text_length + TOKENS],
                                     reference_shared[:TOKENS]),
        "velocity": metrics(native_velocity, velocity),
    }
    (step / "reference-onepass.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def pixels(args) -> None:
    from PIL import Image
    with Image.open(args.a) as a, Image.open(args.b) as b:
        x = np.asarray(a.convert("RGB"), np.float64)
        y = np.asarray(b.convert("RGB"), np.float64)
    if x.shape != y.shape:
        raise ValueError("images differ in size")
    mse = np.mean((x - y) ** 2)
    print(json.dumps({"mae": float(np.mean(np.abs(x - y))),
                      "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else float("inf")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("noise")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    for name in ("generate", "onepass"):
        p = sub.add_parser(name)
        p.add_argument("--reference", type=Path, required=True, help="flux3-reference checkout")
        p.add_argument("--weights", type=Path, required=True, help="flux-3-image-klein package")
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--resolution", type=int, default=512, choices=(512, 1024))
        if name == "generate":
            p.add_argument("--prompt", required=True)
            p.add_argument("--image", type=Path, help="one reference; omit for text-to-image")
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--steps", type=int, default=16)
            p.add_argument("--alpha", type=float, default=16.0)
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--step-dir", type=Path, required=True)
            p.add_argument("--text-tokens", type=Path, required=True)
            p.add_argument("--text-length", type=int, required=True)
            p.add_argument("--reference-tokens", type=Path,
                           help="native reference-image tokens; omit for text-to-image")
    p = sub.add_parser("pixels")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = parser.parse_args()
    {"noise": noise, "generate": generate, "onepass": onepass, "pixels": pixels}[args.command](args)


if __name__ == "__main__":
    main()
