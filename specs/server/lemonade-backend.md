# SERVER-LEMONADE: Lemonade backend integration

Phlegm supplies a Lemonade main backend through
`integrations/lemonade/lemonade-main.patch`. A local-path `main` checkpoint
names a Phlegm model directory. The backend starts the installed `phlegm`
binary with that directory, Lemonade's registered model ID, an ephemeral
loopback port, and optional kernel/bridge paths. `ctx_size` becomes Phlegm's
`--max-ctx` when positive. Phlegm must load its tokenizer and C++ engine
before its `/health` endpoint answers; Lemonade waits for that endpoint.

The descriptor declares Linux XDNA2 NPU support and `ExclusiveNpu` slot
policy. Lemonade owns process lifetime, unloading the child on eviction.
The backend forwards chat and legacy completions. It does not claim
`/v1/responses`, embedding, image or audio support. Both the `user.NAME`
model ID and its bare `NAME` alias refer to the same single loaded model;
other IDs remain 404. Model weights and kernels are never copied into
Lemonade's cache or downloaded by this backend.

Validation requires: Phlegm CLI/server alias tests; Lemonade backend C++
compilation and registry checks against the pinned main commit; a live
Lemonade load/chat/stream/unload on an XDNA2 host with a compatible model and
kernel set. A compile or mock pass does not establish device inference.
