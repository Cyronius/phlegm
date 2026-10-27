"""The family recipes, by ModelSpec.family. Each module exposes the same
surface: recipe(spec, max_ctx), layout(spec, max_ctx), pack_plan(spec),
programs(spec), builds(spec), hf_config_check(spec), manifest_layout(spec,
max_ctx), KERNEL_SOURCES, GEN_KERNELS."""
from __future__ import annotations

from types import ModuleType

from .spec import ModelSpec


def family_module(name: str) -> ModuleType:
    if name == "qwen36moe":
        from . import qwen36moe
        return qwen36moe
    if name == "qwen35":
        from . import qwen35
        return qwen35
    if name == "lfm2":
        from . import lfm2
        return lfm2
    if name == "dflash2":
        from . import dflash2
        return dflash2
    if name in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2", "k2"):
        from . import dense
        return dense
    if name in NOT_IMPLEMENTED:
        raise NotImplementedError(f"the open kernels have no recipe for {name!r} yet: "
                                  f"{NOT_IMPLEMENTED[name]}")
    raise ValueError(f"no recipe for family {name!r} (have {', '.join(FAMILIES)})")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2",
            "lfm2", "k2", "dflash2")

# Families whose ModelSpec derives but whose kernels do not exist. Naming the gap here is
# the point: routing such a model to the nearest recipe would emit kernels that drop a whole
# stage of the layer and then report parity against a replica making the same mistake.
NOT_IMPLEMENTED = {
    "gptoss": "the arithmetic is settled and tested (model/replica_gptoss.py); standalone "
              "gptoss_gate, gptoss_mxfp4, gptoss_router, gptoss_expert_gemv, gptoss_mix and "
              "gptoss_residual, gptoss_attention, gptoss_attention_state, gptoss_rope and "
              "gptoss_qkv_gemv "
              "kernels cover clamped activation, MXFP4 expert projection with bias, biased "
              "top-four routing, mixing, output bias/residual and sink attention arithmetic; "
              "the one-token MoE FFN, biased QKV projection, full/sliding sink attention "
              "and YaRN rotation "
              "are NPU-checked, "
              "but no layer recipe composes them. "
              "o_proj bias still has no slot in the existing layer design; the MoE FFN has to "
              "compose with sliding-window layers, which no recipe does today; QKV projection "
              "and YaRN RoPE are checked separately but not yet connected to the sink path; "
              "and the expert "
              "intermediate equals hidden, which "
              "qwen36moe's core layout does not survive -- the stripe assignment, the core "
              "scratch and the expert hidden's element count each refuse it by name "
              "(OPEN-MOE-WIDE-FF). The packer reads the container's attention projections "
              "and head (std_fuse, OPEN-PACK-CHUNK-FUSE) and knows where each expert's "
              "gate, up and down live (OPEN-PACK-EXPERT-ORDER), but has no op that PLACES "
              "them, because where they go waits on that layout. ATTN_SINK landed at "
              "e0511bd7 and is no longer a gap. See .plans/flm-kernel-coverage.md",
}
