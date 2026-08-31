"""OpenAI-compatible HTTP server (stdlib only).

Endpoints:
  GET  /health                 -> {"status":"ok","backend":...}
  GET  /v1/models              -> OpenAI model list
  POST /v1/chat/completions    -> chat completion; stream=true gives SSE deltas

No third-party deps: uses http.server.ThreadingHTTPServer so it runs anywhere
Python does. Generation is serialized inside the backend (single shared NPU).
"""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import uuid

from . import config
from .backends import get_backend, GenParams, DeviceBusyError
from .tokenizer import get_tokenizer
from .sampler import sampler_from_request


def _now() -> int:
    return int(time.time())


def _cmpl_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # quieter logging
    def log_message(self, fmt, *args):
        pass

    # ---- helpers -----------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        # chunked so we can stream without knowing length up front
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_write(self, data: str):
        # HTTP chunked transfer encoding frame
        payload = data.encode("utf-8")
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _error(self, status, message, err_type="invalid_request_error"):
        self._send_json({"error": {"message": message, "type": err_type}}, status=status)

    # ---- routing -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "backend": config.resolve_backend()})
        elif self.path == "/v1/models":
            self._send_json(self._models_payload())
        else:
            self._error(404, f"unknown path {self.path}", "not_found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._error(404, f"unknown path {self.path}", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw or b"{}")
        except Exception as e:
            self._error(400, f"invalid JSON body: {e}")
            return
        try:
            self._chat_completion(req)
        except DeviceBusyError as e:
            # NPU is a single shared device — tell the client to retry.
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            body = json.dumps({"error": {"message": str(e), "type": "device_busy"}}).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self._error(500, f"generation error: {e}", "internal_error")

    # ---- payloads ----------------------------------------------------------
    def _models_payload(self):
        return {
            "object": "list",
            "data": [{
                "id": config.MODEL_ID,
                "object": "model",
                "created": _now(),
                "owned_by": "fastflowlm-open-npu",
            }],
        }

    def _chat_completion(self, req: dict):
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "'messages' must be a non-empty array")
            return

        model = req.get("model", config.MODEL_ID)
        stream = bool(req.get("stream", False))
        max_tokens = int(req.get("max_tokens") or config.DEFAULT_MAX_TOKENS)
        max_tokens = max(1, min(max_tokens, config.MAX_TOKENS_CAP))
        temperature = req.get("temperature")
        top_p = req.get("top_p")
        seed = req.get("seed")

        tok = get_tokenizer()
        backend = get_backend()
        sampler = sampler_from_request(temperature, top_p, seed)

        prompt = tok.apply_chat_template(messages)
        prompt_ids = tok.encode(prompt)
        stop_ids = set()
        if tok.eos_id is not None:
            stop_ids.add(tok.eos_id)
        params = GenParams(max_tokens=max_tokens, sampler=sampler, stop_token_ids=stop_ids)

        if stream:
            self._stream_response(backend, tok, prompt_ids, params, model)
        else:
            self._full_response(backend, tok, prompt_ids, params, model)

    def _full_response(self, backend, tok, prompt_ids, params, model):
        out_ids = list(backend.generate(prompt_ids, params))
        text = tok.decode(out_ids)
        finish = "length" if len(out_ids) >= params.max_tokens else "stop"
        self._send_json({
            "id": _cmpl_id(),
            "object": "chat.completion",
            "created": _now(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        })

    def _stream_response(self, backend, tok, prompt_ids, params, model):
        cid = _cmpl_id()
        created = _now()

        def chunk(delta: dict, finish=None):
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        self._send_sse_headers()
        # opening role delta
        self._sse_write("data: " + json.dumps(chunk({"role": "assistant"})) + "\n\n")

        out_ids: list[int] = []
        prev_text = ""
        count = 0
        for tid in backend.generate(prompt_ids, params):
            out_ids.append(tid)
            count += 1
            # incremental decode: emit only the newly-revealed suffix
            full = tok.decode(out_ids)
            if full.startswith(prev_text):
                piece = full[len(prev_text):]
            else:
                piece = full  # decode not monotonic (shouldn't happen with placeholder)
            prev_text = full
            if piece:
                self._sse_write("data: " + json.dumps(chunk({"content": piece})) + "\n\n")

        finish = "length" if count >= params.max_tokens else "stop"
        self._sse_write("data: " + json.dumps(chunk({}, finish=finish)) + "\n\n")
        self._sse_write("data: [DONE]\n\n")
        self._sse_end()


def serve():
    backend_name = config.resolve_backend()
    httpd = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    print(f"FastFlowLM open-NPU server on http://{config.HOST}:{config.PORT}")
    print(f"  backend: {backend_name}   model: {config.MODEL_ID}")
    print(f"  POST /v1/chat/completions | GET /v1/models | GET /health")
    if backend_name == "npu":
        print("  NPU backend: driver starts lazily on first request (~7s, loads pools).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        try:
            get_backend().close()
        except Exception:
            pass
        httpd.server_close()


if __name__ == "__main__":
    serve()
