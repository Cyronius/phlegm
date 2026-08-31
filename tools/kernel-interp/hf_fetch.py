"""Ranged-download individual tensors from the HF reference model (Qwen/Qwen3.6-35B-A3B).

Caches to ./hf_ref/<tensor_name>.npy (float32). Only fetches the byte range of
each tensor from its shard - no full-shard downloads.
Usage: python hf_fetch.py <tensor-name> [...]   (or import fetch())
"""
import urllib.request, json, struct, os, sys
import numpy as np

BASE = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/"
HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "hf_ref")
os.makedirs(REFDIR, exist_ok=True)
_wm = json.load(open(os.path.join(HERE, "hf_weight_map.json")))
_hdr_cache = {}

def _shard_header(shard):
    if shard in _hdr_cache:
        return _hdr_cache[shard]
    req = urllib.request.Request(BASE + shard, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read())[0]
    req = urllib.request.Request(BASE + shard, headers={"Range": f"bytes=8-{7+n}"})
    hdr = json.loads(urllib.request.urlopen(req, timeout=120).read())
    _hdr_cache[shard] = (8 + n, hdr)
    return _hdr_cache[shard]

def fetch(name):
    cache = os.path.join(REFDIR, name + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    shard = _wm[name]
    base_off, hdr = _shard_header(shard)
    t = hdr[name]
    o0, o1 = t["data_offsets"]
    req = urllib.request.Request(
        BASE + shard, headers={"Range": f"bytes={base_off+o0}-{base_off+o1-1}"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    assert len(raw) == o1 - o0, (len(raw), o1 - o0)
    if t["dtype"] == "BF16":
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif t["dtype"] == "F32":
        a = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(t["dtype"])
    a = a.reshape(t["shape"]).copy()
    np.save(cache, a)
    print(f"fetched {name} {t['dtype']} {t['shape']} -> {cache}")
    return a

if __name__ == "__main__":
    for n in sys.argv[1:]:
        fetch(n)
