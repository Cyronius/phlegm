"""Probe the layer.xclbin hw_context's "~3 consecutive submissions" cap
directly: submit increasing runlist chunk sizes on a SINGLE context (no
ping-pong) and see where/how it actually fails.

Safety: reads driver stdout on a background thread so a genuine hang (XRT
wait() never returning, nothing printed) can't block this script forever —
main thread times out on a queue.get() and force-kills the subprocess
instead of blocking on a raw readline(). A clean STEP FAILED/RUNLIST FAILED
line (the driver already catches XRT exceptions around wait()) is the good
outcome; a forced kill after timeout means real hardware wedge risk and the
escalation stops immediately.

See docs/lm-head-npu-bottleneck-instrumentation.md.
Usage: python test_chunk_limit.py [max_chunk_size]
"""
import os, sys, subprocess, threading, queue, time

D = os.environ.get("L40_DIR", "C:/code/FastFlowLM/npu-engine/m3out/l40")
DRIVER = "C:/code/FastFlowLM/npu-engine/m0/out/decode_driver_nobarrier.exe"
XB = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"
ELF = "C:/caps/m0c/elf_000005.bin"
NLAYERS = 40
STEP_TIMEOUT_S = 30   # generous: let a real XRT-side timeout exception surface
                       # cleanly rather than preempting it with a forced kill
INIT_TIMEOUT_S = 120   # loading 40x512MB pools (~20GB) to device can be slow


def make_config(chunk_size, qos=""):
    L = ["device", f"xclbin L {XB}/layer.xclbin {qos}".rstrip(), f"kernel kL L {ELF}"]
    for l in range(NLAYERS):
        L.append(f"buf pool{l} 536870912 {D}/pool_L{l}.bin")
        L.append(f"buf pack{l} 2097152 {D}/pack_L{l}.bin")
        L.append(f"buf side{l} 6291456 {D}/side_L{l}.bin")
        L.append(f"buf state{l} 3145728 {D}/state_L{l}.bin")
    L.append("buf act 1048576")
    L.append("serve")
    for c0 in range(0, NLAYERS, chunk_size):
        L.append("runlist L")
        for l in range(c0, min(c0 + chunk_size, NLAYERS)):
            L.append(f"layer kL pool{l} act pack{l} side{l} state{l}")
        L.append("submit")
    L.append("endserve")
    cfgp = f"{D}/test_chunk_{chunk_size}.txt"
    open(cfgp, "w").write("\n".join(L) + "\n")
    return cfgp


def reader_thread(proc, q):
    for ln in proc.stdout:
        q.put(ln)
    q.put(None)  # EOF


def readline_with_timeout(q, timeout_s):
    try:
        return q.get(timeout=timeout_s)
    except queue.Empty:
        return "__TIMEOUT__"


def try_chunk(chunk_size, qos=""):
    cfg = make_config(chunk_size, qos)
    n_chunks = (NLAYERS + chunk_size - 1) // chunk_size
    tag = f" qos={qos}" if qos else ""
    print(f"chunk_size={chunk_size} ({n_chunks} chunks/step, single context 'L', no ping-pong){tag}")
    proc = subprocess.Popen([DRIVER, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    q = queue.Queue()
    th = threading.Thread(target=reader_thread, args=(proc, q), daemon=True)
    th.start()

    ready = False
    t0 = time.time()
    while time.time() - t0 < INIT_TIMEOUT_S:
        ln = readline_with_timeout(q, INIT_TIMEOUT_S - (time.time() - t0))
        if ln in (None, "__TIMEOUT__"):
            break
        if "SERVE READY" in ln:
            ready = True
            break
        print("  init:", ln.rstrip())
    if not ready:
        print("  -> driver never reached SERVE READY; killing")
        proc.kill()
        return "SETUP-FAILED"

    import numpy as np
    np.zeros(1048576, np.uint8).tofile(f"{D}/test_act.bin")
    proc.stdin.write(f"step {D}/test_act.bin {D}/test_hidden.bin\n")
    proc.stdin.flush()

    result = None
    t1 = time.time()
    while time.time() - t1 < STEP_TIMEOUT_S:
        ln = readline_with_timeout(q, STEP_TIMEOUT_S - (time.time() - t1))
        if ln == "__TIMEOUT__":
            result = "HANG (no output within timeout — forcing kill)"
            break
        if ln is None:
            result = "DRIVER-DIED (stdout closed mid-step)"
            break
        print("  ", ln.rstrip())
        if "STEP OK" in ln:
            result = "OK"
            break
        if "STEP ERR" in ln or "FAILED" in ln:
            result = f"FAILED: {ln.strip()}"
            break
    if result is None:
        result = "HANG (loop exhausted)"

    if result == "OK":
        proc.stdin.write("quit\n")
        try:
            proc.stdin.flush()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print(f"  => {result}\n")
    return result


def main():
    sizes = [3, 4, 5, 6, 8, 10, 20, 40]
    qos = ""
    if len(sys.argv) > 1:
        sizes = [int(sys.argv[1])]
    if len(sys.argv) > 2:
        qos = sys.argv[2]
    for cs in sizes:
        r = try_chunk(cs, qos)
        if r != "OK":
            print(f"stopping escalation at chunk_size={cs} (first non-OK result)")
            break


if __name__ == "__main__":
    main()
