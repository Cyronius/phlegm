"""OpenAI/FLM-compatible HTTP server wrapping the open NPU inference engine.

Exposes /v1/chat/completions (streaming + non-streaming), /v1/models, /health
on top of a pluggable generation backend (mock, or the resident NPU decode
driver). See README.md.
"""
