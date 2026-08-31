"""Golden CPU/GPU reference for the FLM interval-3 overflow.

Loads the real Qwen3.6-35B-A3B (Qwen3.5-MoE hybrid: DeltaNet linear layers +
periodic full-attention + MoE), then reproduces the EXACT layer slices the FLM
engine mis-executes, running the official transformers forward — the correct
behavior the closed engine should produce.

Engine results to compare against (captured on the NPU):
  3LiF [L,L,F]      -> healthy, max|logit| ~10.6
  4Li3 [L,L,F,L]    -> ran but max|logit| ~3.4e38 (fp32 edge)
  5Li3 [L,L,F,L,L]  -> NaN logits, "////////"
If the reference is finite/sane for all three, the overflow is purely the
engine's interval-3 handling, and these logits/norms are the oracle.
"""
import json, sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = "Qwen/Qwen3.6-35B-A3B"
# keep-lists over ORIGINAL layer indices, matching tools/seq-capture/slice_keep.py
VARIANTS = {
    "3LiF": [0, 1, 3],
    "4Li3": [0, 1, 3, 4],
    "5Li3": [0, 1, 3, 4, 5],
}
UNION = sorted(set(sum(VARIANTS.values(), [])))  # [0,1,3,4,5]

print("== loading", REPO, "(bf16, CPU) ==", flush=True)
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()

# locate the text decoder: the submodule with an embed_tokens + a 40-long layers list
decoder = None
for name, mod in model.named_modules():
    if hasattr(mod, "layers") and hasattr(mod, "embed_tokens") and isinstance(getattr(mod, "layers"), nn.ModuleList):
        if len(mod.layers) >= 30:
            decoder = mod; decoder_name = name; break
assert decoder is not None, "could not find decoder"
lm_head = model.get_output_embeddings()
orig_layers = list(decoder.layers)
tcfg = getattr(model.config, "text_config", model.config)
orig_lt = list(tcfg.layer_types)
print(f"decoder='{decoder_name}' n_layers={len(orig_layers)} lm_head={type(lm_head).__name__}", flush=True)
print("orig layer_types[0:8]:", orig_lt[:8], flush=True)

# move only the union layers + embed/norm/rotary/lm_head to GPU
dev = "cuda"
decoder.embed_tokens.to(dev)
if getattr(decoder, "norm", None) is not None: decoder.norm.to(dev)
for extra in ("rotary_emb",):
    m = getattr(decoder, extra, None)
    if m is not None: m.to(dev)
for i in UNION: orig_layers[i].to(dev)
lm_head.to(dev)

enc = tok.apply_chat_template([{"role": "user", "content": "Say hi."}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev)
print("input_ids shape:", tuple(ids.shape), "tokens:", ids[0].tolist(), flush=True)

results = {}
for vname, keep in VARIANTS.items():
    decoder.layers = nn.ModuleList([orig_layers[i] for i in keep])
    tcfg.num_hidden_layers = len(keep)
    tcfg.layer_types = [orig_lt[i] for i in keep]
    if hasattr(decoder, "config"):
        decoder.config.num_hidden_layers = len(keep)
        decoder.config.layer_types = [orig_lt[i] for i in keep]

    norms = []
    hooks = []
    for li, layer in enumerate(decoder.layers):
        def mk(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                norms.append((idx, float(h.float().norm(dim=-1).mean()),
                              float(h.float().abs().max())))
            return hook
        hooks.append(layer.register_forward_hook(mk(li)))

    with torch.no_grad():
        dout = decoder(input_ids=ids, use_cache=False)
        hidden = dout.last_hidden_state if hasattr(dout, "last_hidden_state") else dout[0]
        logits = lm_head(hidden)[0, -1].float()  # last-position logits
    for h in hooks: h.remove()

    fin = torch.isfinite(logits)
    r = {
        "layer_types": tcfg.layer_types,
        "final_hidden_absmax": float(hidden.float().abs().max()),
        "final_hidden_norm": float(hidden.float().norm(dim=-1).mean()),
        "logits_finite": bool(fin.all()),
        "logits_nan": int(torch.isnan(logits).sum()),
        "logits_absmax": (float(logits[fin].abs().max()) if fin.any() else float("nan")),
        "per_layer": [{"i": i, "hid_norm": round(n, 3), "hid_absmax": round(a, 3)} for (i, n, a) in norms],
    }
    if fin.all():
        top = torch.topk(logits, 5)
        r["argmax_id"] = int(top.indices[0])
        r["argmax_tok"] = tok.decode([int(top.indices[0])])
        r["top5"] = [(int(i), tok.decode([int(i)]), round(float(v), 2)) for i, v in zip(top.indices, top.values)]
    results[vname] = r
    print(f"\n=== {vname} {tcfg.layer_types} ===", flush=True)
    print(f"  per-layer hidden norm/absmax:", [(x['i'], x['hid_norm'], x['hid_absmax']) for x in r['per_layer']], flush=True)
    print(f"  final_hidden absmax={r['final_hidden_absmax']:.3g} norm={r['final_hidden_norm']:.3g}", flush=True)
    print(f"  logits finite={r['logits_finite']} nan={r['logits_nan']} absmax={r['logits_absmax']:.3g}", flush=True)
    if fin.all():
        print(f"  argmax={r['argmax_id']} '{r['argmax_tok']}'  top5={r['top5']}", flush=True)

json.dump(results, open("/root/reference_results.json", "w"), indent=2)
print("\nWROTE /root/reference_results.json", flush=True)
