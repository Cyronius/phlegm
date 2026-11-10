"""Explicit, opt-in Linux NPU switch gate. Run against a freshly started U7 server.

Never invoked by cargo/pytest. The caller owns the NPU/port lease and starts the
server with --keep-alive 5. Uses only the Python standard library.
"""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.request


def processes():
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm can contain spaces/parentheses; fields after its closing ')' start at 3.
            fields = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            result[pid] = (int(fields[1]), fields[19])  # ppid, starttime
        except (OSError, ValueError, IndexError):
            pass
    return result


def descendants(pid):
    table = processes()
    parents = {pid}
    found = {}
    while True:
        new = {child: (child, start) for child, (parent, start) in table.items()
               if parent in parents and child not in parents}
        if not new:
            return found
        found.update(new)
        parents.update(new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8767")
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--kat", default="KAT-Coder-V2.5-Dev-Q4K-NPU2")
    parser.add_argument("--embed", default="trit-embed-3l")
    parser.add_argument("--gemma", default="gemma-4-26B-A4B-it-NPU2")
    parser.add_argument("--image", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--check-edits", action="store_true")
    parser.add_argument("--lock-file", type=Path,
                        default=Path(os.environ.get("PHLEGM_NPU_LOCK", "/tmp/phlegm-npu.lock")))
    parser.add_argument("--api-key", default=os.environ.get("PHLEGM_API_KEY"))
    args = parser.parse_args()
    assert Path(f"/proc/{args.server_pid}").is_dir(), "server PID does not exist"
    assert not descendants(args.server_pid), "start with a fresh, unloaded server"
    observed = set()
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            observed.update(descendants(args.server_pid).values())
            stop.wait(0.1)

    thread = threading.Thread(target=monitor)
    thread.start()

    def call(path, payload=None):
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        request = urllib.request.Request(args.base_url + path, headers=headers,
                                         data=None if payload is None else json.dumps(payload).encode())
        with urllib.request.urlopen(request, timeout=3600) as response:
            data = response.read()
        return json.loads(data)

    def snapshot():
        return set(descendants(args.server_pid).values())

    def old_workers_gone(old):
        table = processes()
        assert not [(pid, start) for pid, start in old
                    if pid in table and table[pid][1] == start], "previous model worker survived swap"

    try:
        catalog = call("/v1/models")["data"]
        ids = {m["id"] for m in catalog}
        assert {args.kat, args.embed, args.gemma, args.image} <= ids
        assert all(not m["loaded"] for m in catalog)
        previous = set()
        for model in (args.kat, args.embed, args.gemma, args.image):
            if model == args.embed:
                response = call("/v1/embeddings", {"model": model, "input": ["hello", "goodbye"]})
                assert len(response["data"]) == 2
                assert all(len(row["embedding"]) == 384 and
                           all(math.isfinite(x) for x in row["embedding"])
                           for row in response["data"])
            elif model == args.image:
                response = call("/v1/images/generations", {"model": model,
                    "prompt": "A red cube on a white table", "size": "512x512", "seed": 41})
                png64 = response["data"][0]["b64_json"]
                assert base64.b64decode(png64, validate=True).startswith(b"\x89PNG\r\n\x1a\n")
                if args.check_edits:
                    edited = call("/v1/images/edits", {"model": model, "image": png64,
                        "prompt": "Make the cube blue", "size": "512x512", "seed": 41})
                    assert base64.b64decode(edited["data"][0]["b64_json"], validate=True).startswith(b"\x89PNG\r\n\x1a\n")
            else:
                response = call("/v1/chat/completions", {"model": model,
                    "messages": [{"role": "user", "content": "Name one primary color."}],
                    "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
                assert response["model"] == model
                assert response["choices"][0]["finish_reason"] in ("stop", "length")
            old_workers_gone(previous)
            previous = snapshot()
            if model == args.image:
                assert not previous, "image subprocess survived its response"
                assert not args.lock_file.exists(), "image NPU lock survived its response"
            else:
                assert previous, "resident model has no worker"
                assert args.lock_file.read_text().strip() == str(args.server_pid)
            print(f"PASS switch to {model}; old worker reaped", flush=True)
        time.sleep(7)  # --keep-alive 5 plus the 250 ms server idle sweep
        assert all(not m["loaded"] for m in call("/v1/models")["data"])
        assert not args.lock_file.exists(), "NPU lock is not free"
    finally:
        stop.set()
        thread.join()
    old_workers_gone(observed)
    print("PASS no observed bridge/Python descendants remain; NPU lock free")


if __name__ == "__main__":
    main()
