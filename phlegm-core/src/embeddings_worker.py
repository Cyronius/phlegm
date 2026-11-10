"""JSON-lines adapter for TritNpu. stdout is reserved for protocol messages."""
import contextlib
import json
import sys
import traceback

sys.dont_write_bytecode = True
protocol = sys.stdout

def reply(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), file=protocol, flush=True)

try:
    with contextlib.redirect_stdout(sys.stderr):
        sys.path.insert(0, sys.argv[1])
        from trit_npu import TritNpu
        encoder = TritNpu(sys.argv[2])
    reply({"ok": True, "protocol": 1})
except Exception as exc:
    reply({"ok": False, "error": str(exc)})
    sys.exit(1)

try:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                ids = encoder.tokenize(request["texts"]) if "texts" in request else request["token_ids"]
                vocab = encoder.model.word.qweight.shape[0]
                if any(not row or len(row) > 128 or any(type(i) is not int or i < 0 or i >= vocab for i in row) for row in ids):
                    reply({"ok": False, "invalid": True, "error": "token IDs must be in vocabulary; each sequence must contain 1 to 128 tokens"})
                    continue
                vectors, _ = encoder.encode_ids(ids)
            reply({"ok": True, "vectors": vectors.tolist(), "tokens": sum(map(len, ids))})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "error": str(exc)})
finally:
    encoder.close()
