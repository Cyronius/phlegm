"""General q4nx layer slicer: keep an explicit list of OLD layer indices, in
order, renumber them 0..N-1, and write a matching config. Weights stay
bit-identical (byte-copied), so the closed engine sees known-good packing.

Used for the interval-3 NaN depth-bisection: build minimal models that each
isolate one variable (depth / interval / presence of a full-attention layer).

Usage:
  python slice_keep.py <model_dir> <name> <interval> <old_idx,old_idx,...>
    e.g. python slice_keep.py <dir> 3LiF 3 0,1,3   -> model_3LiF.q4nx / config_3LiF.json

layer_types for the output are derived from the original config: a kept old
layer that was 'full_attention' stays full at its new position, else linear.
(This matches how the working 6Li3 / 8Li4 variants were built.)
"""
import json, re, struct, sys
from pathlib import Path

CHUNK = 64 * 1024 * 1024

def main(model_dir, name, interval, keep_csv):
    model_dir = Path(model_dir)
    keep = [int(x) for x in keep_csv.split(",")]
    interval = int(interval)
    # Read the PRISTINE 40L weights, never the live model.q4nx (a capture run may
    # have swapped a small variant into it).
    src = model_dir / "model.q4nx.orig"
    if not src.exists():
        src = model_dir / "model.q4nx"
    dst = model_dir / f"model_{name}.q4nx"

    orig_cfg = json.loads((model_dir / "config.json.orig").read_text()
                          if (model_dir / "config.json.orig").exists()
                          else (model_dir / "config.json").read_text())
    orig_types = orig_cfg["layer_types"]
    old_to_new = {old: i for i, old in enumerate(keep)}
    new_types = [None] * len(keep)
    for old, new in old_to_new.items():
        new_types[new] = orig_types[old]
    print(f"keep old {keep} -> new 0..{len(keep)-1}")
    print(f"new layer_types: {new_types}")

    with open(src, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    meta = header.pop("__metadata__", None)

    layer_re = re.compile(r"^model\.layer\.(\d+)\.(.+)$")
    kept = []  # (old_name, new_name, offsets)
    for nm, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        m = layer_re.match(nm)
        if m:
            old = int(m.group(1))
            if old not in old_to_new:
                continue
            new_name = f"model.layer.{old_to_new[old]}.{m.group(2)}"
        else:
            new_name = nm  # embeddings / lm_head / final norm etc. kept as-is
        kept.append((nm, new_name, info["dtype"], info["shape"], info["data_offsets"]))

    new_header = {}
    if meta is not None:
        new_header["__metadata__"] = meta
    pos = 0
    for _, new_name, dtype, shape, (b, e) in kept:
        size = e - b
        new_header[new_name] = {"dtype": dtype, "shape": shape,
                                "data_offsets": [pos, pos + size]}
        pos += size
    hbytes = json.dumps(new_header, separators=(",", ":")).encode()
    hbytes += b" " * ((8 - len(hbytes) % 8) % 8)

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(hbytes)))
        fout.write(hbytes)
        data_base = 8 + hlen
        for old_name, _, _, _, (b, e) in kept:
            fin.seek(data_base + b)
            rem = e - b
            while rem:
                buf = fin.read(min(CHUNK, rem))
                if not buf:
                    raise IOError(f"short read in {old_name}")
                fout.write(buf); rem -= len(buf)
    print(f"wrote {dst} ({dst.stat().st_size/1e9:.2f} GB, {len(kept)} tensors)")

    cfg = dict(orig_cfg)
    cfg["num_hidden_layers"] = len(keep)
    cfg["full_attention_interval"] = interval
    cfg["layer_types"] = new_types
    (model_dir / f"config_{name}.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote config_{name}.json (layers={len(keep)} interval={interval})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
