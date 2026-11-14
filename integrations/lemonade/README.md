# Lemonade backend for Phlegm

The [patch](lemonade-main.patch) adds a `phlegm` backend to Lemonade main at
commit `8560a810db0b6c4cdf92c0b88133e12213b12e8b` (2026-11-11). It
uses Lemonade's backend descriptor, NPU slot policy, process watchdog and
OpenAI chat/completion routing. Phlegm remains the inference server and owns
its C++ engine. The patch has no model download or Phlegm binary installer:
both the model directory and `phlegm` executable are local.

Apply it in a checkout of that Lemonade commit or a compatible later main:

```sh
git apply --check /path/to/phlegm/integrations/lemonade/lemonade-main.patch
git apply /path/to/phlegm/integrations/lemonade/lemonade-main.patch
cmake --preset default
cmake --build --preset default --target lemond lemonade
```

Build this Phlegm revision with `cargo build -p phlegm-cli --features npu`
and put `target/debug/phlegm` on Lemonade's `PATH`. A model directory must
already work with `phlegm serve --model-dir DIR`;
its tokenizer, compatible bridge and kernel set are required. If the
kernels are not under `DIR/open_kernels`, set `phlegm_kernels` when registering
the model. The optional `phlegm_bridge` selects an explicit bridge executable.

Register a local model through Lemonade's API (replace paths and model name):

```sh
curl http://127.0.0.1:13305/v1/models/register \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name": "user.MyPhlegmModel",
    "recipe": "phlegm",
    "source": "local_path",
    "checkpoint": "/absolute/path/to/model-dir",
    "recipe_options": {
      "phlegm_kernels": "/absolute/path/to/kernels",
      "ctx_size": 8192
    }
  }'
```

Then `lemonade load user.MyPhlegmModel` and send a chat request to Lemonade's
`/v1/chat/completions`. Lemonade launches `phlegm serve` on loopback and
unloads it with the model. The model ID works with both `user.MyPhlegmModel`
and the exposed `MyPhlegmModel` alias. The backend supports chat and legacy
completions, including Phlegm's streaming responses. Lemonade's `/v1/responses`
is outside this integration until Phlegm serves that endpoint.

The patch targets Linux XDNA2 hosts. It is a source integration for Lemonade
main; no Lemonade release package is modified by this repository.
