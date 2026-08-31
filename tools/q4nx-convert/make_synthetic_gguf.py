"""Build a small qwen35moe GGUF from real HF reference tensors (hf_ref/) so the
full converter pipeline can be exercised + byte-compared against Josh's
model_3LiF.q4nx.  Emulates llama.cpp conventions: zero-centered RMSNorms carry a
baked +1, tensors in HF row order, weights stored F32/Q4_K.

Layer 0 = linear-attn (from HF layer 0), layer 1 = full-attn (from HF layer 3),
so both code paths (linear + full q_proj deinterleave) run.  Experts: 8-expert
stub placing the 5 available real experts at their true indices.

Not a faithful full model -- a pipeline+layout test harness.  For a real run,
use Josh's published GGUF directly with convert.py.
"""
import sys, numpy as np
from gguf import GGUFWriter

HF = "c:/code/FastFlowLM/tools/kernel-interp/hf_ref"
L = lambda n: np.load(f"{HF}/{n}.npy").astype(np.float32)


def build(path):
    w = GGUFWriter(path, "qwen35moe")
    w.add_uint32("qwen35moe.block_count", 2)
    w.add_uint32("qwen35moe.embedding_length", 2048)
    w.add_uint32("qwen35moe.full_attention_interval", 2)
    w.add_uint32("qwen35moe.attention.value_length", 256)
    w.add_uint32("qwen35moe.expert_count", 8)
    w.add_uint32("qwen35moe.expert_used_count", 8)
    w.add_uint32("qwen35moe.vocab_size", 248320)

    def add_f32(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, np.float32))

    def add_q4k(name, arr):   # store F32 (gguf 0.19 cannot encode k-quants); layout test
        w.add_tensor(name, np.ascontiguousarray(arr, np.float32))

    # ---- globals ----
    # embed / lm_head: use lm_head rows we have, padded (vocab stub 32 rows for speed)
    lmh = L("lm_head_rows0-1023")               # [1024,2048]
    add_f32("token_embd.weight", lmh[:32])       # tiny embed stub
    add_f32("output_norm.weight", np.zeros(2048, np.float32) + 0.6)  # already-effective norm
    add_q4k("output.weight", lmh)                # lm_head -> q8 in q4nx

    # ---- layer 0: linear-attn (HF layer 0) ----
    p = "model.language_model.layers.0"
    add_q4k("blk.0.attn_qkv.weight", L(f"{p}.linear_attn.in_proj_qkv.weight"))
    add_q4k("blk.0.attn_gate.weight", L(f"{p}.linear_attn.in_proj_z.weight"))
    add_q4k("blk.0.ssm_out.weight", L(f"{p}.linear_attn.out_proj.weight"))
    add_f32("blk.0.ssm_a", L(f"{p}.linear_attn.A_log"))
    add_f32("blk.0.ssm_dt.bias", L(f"{p}.linear_attn.dt_bias"))
    add_f32("blk.0.ssm_alpha.weight", L(f"{p}.linear_attn.in_proj_a.weight"))
    add_f32("blk.0.ssm_beta.weight", L(f"{p}.linear_attn.in_proj_b.weight"))
    add_f32("blk.0.ssm_conv1d.weight", L(f"{p}.linear_attn.conv1d.weight").reshape(8192, 4))
    add_f32("blk.0.ssm_norm.weight", L(f"{p}.linear_attn.norm.weight"))         # not zero-centered
    add_f32("blk.0.attn_norm.weight", L(f"{p}.input_layernorm.weight") + 1.0)    # baked +1
    add_f32("blk.0.post_attention_norm.weight", np.zeros(2048, np.float32) + 0.9)
    add_f32("blk.0.ffn_gate_inp.weight", L(f"{p}.mlp.gate.weight"))              # router [256,2048]
    add_f32("blk.0.ffn_gate_inp_shexp.weight", L(f"{p}.mlp.shared_expert_gate.weight").reshape(1, 2048))
    add_q4k("blk.0.ffn_gate_shexp.weight", L(f"{p}.mlp.shared_expert.gate_proj.weight"))
    add_q4k("blk.0.ffn_up_shexp.weight", L(f"{p}.mlp.shared_expert.up_proj.weight"))
    add_q4k("blk.0.ffn_down_shexp.weight", L(f"{p}.mlp.shared_expert.down_proj.weight"))

    # experts: 8-expert stub, real experts at indices 0,7 (down only have 7)
    gate = np.zeros((8, 512, 2048), np.float32)
    up = np.zeros((8, 512, 2048), np.float32)
    down = np.random.default_rng(0).standard_normal((8, 2048, 512)).astype(np.float32) * 0.02
    for e in (0, 7, 11, 16, 21):
        if e < 8:
            GU = L(f"l0_expert{e}_gate_up_proj")   # [1024,2048]
            gate[e] = GU[:512]; up[e] = GU[512:]
    down[7] = L("l0_expert7_down_proj")
    add_q4k("blk.0.ffn_gate_exps.weight", gate.reshape(8, 512, 2048))
    add_q4k("blk.0.ffn_up_exps.weight", up)
    add_q4k("blk.0.ffn_down_exps.weight", down)

    # ---- layer 1: full-attn (HF layer 3) ----
    q = "model.language_model.layers.3"
    add_q4k("blk.1.attn_q.weight", L(f"{q}.self_attn.q_proj.weight"))    # [8192,2048] interleaved
    add_q4k("blk.1.attn_k.weight", L(f"{q}.self_attn.k_proj.weight"))
    add_q4k("blk.1.attn_v.weight", L(f"{q}.self_attn.v_proj.weight"))
    add_q4k("blk.1.attn_output.weight", L(f"{q}.self_attn.o_proj.weight"))
    add_f32("blk.1.attn_q_norm.weight", L(f"{q}.self_attn.q_norm.weight") + 1.0)
    add_f32("blk.1.attn_k_norm.weight", L(f"{q}.self_attn.k_norm.weight") + 1.0)
    add_f32("blk.1.attn_norm.weight", np.zeros(2048, np.float32) + 0.7)
    add_f32("blk.1.post_attention_norm.weight", np.zeros(2048, np.float32) + 0.9)
    add_f32("blk.1.ffn_gate_inp.weight", L(f"{p}.mlp.gate.weight"))
    add_f32("blk.1.ffn_gate_inp_shexp.weight", L(f"{p}.mlp.shared_expert_gate.weight").reshape(1, 2048))
    add_q4k("blk.1.ffn_gate_shexp.weight", L(f"{p}.mlp.shared_expert.gate_proj.weight"))
    add_q4k("blk.1.ffn_up_shexp.weight", L(f"{p}.mlp.shared_expert.up_proj.weight"))
    add_q4k("blk.1.ffn_down_shexp.weight", L(f"{p}.mlp.shared_expert.down_proj.weight"))
    add_q4k("blk.1.ffn_gate_exps.weight", gate)
    add_q4k("blk.1.ffn_up_exps.weight", up)
    add_q4k("blk.1.ffn_down_exps.weight", down)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "synthetic_qwen35moe.gguf")
