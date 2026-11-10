"""Read-only CLI/HTTP smoke: discovery and rejection routes never load a model.

Wrap in dibs run port-8767 (or the selected port). No NPU lease is needed.
"""
import argparse
from collections import Counter
import json
import os
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--image-config")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    command = [args.server, "serve", "--models-dir", args.models_dir,
               "--host", "127.0.0.1", "--port", str(args.port)]
    if args.image_config:
        command.extend(["--image-config", args.image_config])
    process = subprocess.Popen(command)

    def call(path, payload=None, expected=200):
        headers = {"Content-Type": "application/json"}
        if os.environ.get("PHLEGM_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["PHLEGM_API_KEY"]
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}" + path,
            headers=headers,
            data=None if payload is None else json.dumps(payload).encode())
        try:
            response = urllib.request.urlopen(request, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            assert response.code == expected, (response.code, expected)
            return json.load(response)

    try:
        deadline = time.monotonic() + 10
        while True:
            assert process.poll() is None, "server exited during startup"
            try:
                call("/health")
                break
            except urllib.error.URLError:
                assert time.monotonic() < deadline, "server startup timed out"
                time.sleep(0.05)
        models = call("/v1/models")["data"]
        assert models and all(not m["loaded"] and m["status"] == "unloaded" for m in models)
        embed = next(m["id"] for m in models if m["capabilities"] == ["embedding"])
        call("/v1/chat/completions", {"model": embed, "messages": []}, expected=400)
        call("/v1/chat/completions", {"model": "__u7_absent_model__", "messages": []}, expected=404)
        call("/v1/embeddings", {"input": "missing model"}, expected=400)
        call("/v1/images/generations", {"model": embed, "prompt": "wrong kind"}, expected=400)
        assert models == call("/v1/models")["data"], "rejected requests changed the unloaded catalog"
        assert process.poll() is None, "server stopped"
        counts = Counter(m["capabilities"][0] for m in models)
        print(f"PASS: {len(models)} models {dict(counts)}; all unloaded; 4 rejection routes; health")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
