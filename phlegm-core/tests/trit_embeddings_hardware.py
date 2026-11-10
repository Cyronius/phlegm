"""Hardware acceptance: direct TritNpu versus the server, never run by cargo test.

Run under the NPU and port Dibs leases; see specs/server/spec.md for the command.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--server", required=True)
parser.add_argument("--port", type=int, default=8767)
parser.add_argument("--direct", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / "open_kernels/designs/trit_embed"
texts = ["how do index funds work", "A man is playing a guitar on the street.", "Rain is expected tomorrow."]
if args.direct:
    sys.path.insert(0, str(source))
    from trit_npu import TritNpu  # noqa: E402
    encoder = TritNpu(args.model_dir)
    try:
        ids = encoder.tokenize(texts)
        direct, _ = encoder.encode_ids(ids)
        print(json.dumps({"ids": ids, "vectors": direct.tolist()}))
    finally:
        encoder.close()
    sys.exit(0)

# Exit the direct reference process before the server opens its XRT context.
direct = subprocess.run([sys.executable, "-I", __file__, "--direct", "--model-dir", args.model_dir,
    "--server", args.server, "--port", str(args.port)], check=True, stdout=subprocess.PIPE, text=True)
reference = json.loads(direct.stdout)
ids, expected = reference["ids"], reference["vectors"]

model = "trit-acceptance"
base = f"http://127.0.0.1:{args.port}"
server = subprocess.Popen([args.server, "serve", "--embed-model", args.model_dir,
    "--embed-model-name", model, "--embed-python", sys.executable, "--embed-source", str(source),
    "--port", str(args.port)], stdout=subprocess.DEVNULL)
try:
    for _ in range(600):
        if server.poll() is not None:
            raise RuntimeError(f"server exited {server.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as response:
                if json.load(response).get("backend") == "trit-embed":
                    break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        raise RuntimeError("server did not become ready")
    for inputs in (texts, ids):
        payload = json.dumps({"model": model, "input": inputs, "encoding_format": "float"}).encode()
        req = urllib.request.Request(base + "/v1/embeddings", payload, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.load(response)
        assert result["usage"]["total_tokens"] == sum(map(len, ids))
        assert len(result["data"]) == len(expected)
        for item, vector in zip(result["data"], expected):
            assert len(item["embedding"]) == len(vector)
            assert all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in zip(item["embedding"], vector))
    print("PASS: text and token-ID server vectors match direct TritNpu binary32 values")
finally:
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
