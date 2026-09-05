"""The op catalogue: each kernel template, the parameters it takes, and the
points it has been VALIDATED at (a design's make_test.py / compare.py, or a
whole-layer run against the fp64 replica). A recipe asking for a point outside
a template's validated set fails here, at generation time, with the constraint
named -- never at run time with garbage.

A template that is parametric in the source but only tested at one point lists
that one point: `attn` takes HD / NH / KVH as compile-time macros, and its set
grows when a new point has been run and compared, not when the macro exists.

Also here: the physical limits a generated design must respect (checked, not
derived -- see the plan's "design geometry" item).

Traces: OPEN-OP-RANGE (specs/open-engine/spec.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class OpRangeError(ValueError):
    """A recipe asked a template for a parameter outside its validated set."""


@dataclass(frozen=True)
class Param:
    """One parameter of a template: an explicit set of validated values, or a
    predicate with a description of the rule."""
    values: frozenset | None = None
    rule: Callable[[Any], bool] | None = None
    describe: str = ""

    def ok(self, v: Any) -> bool:
        if self.values is not None:
            return v in self.values
        return bool(self.rule(v)) if self.rule else True

    def expected(self) -> str:
        if self.values is not None:
            return "{" + ", ".join(str(x) for x in sorted(self.values)) + "}"
        return self.describe


def values(*vs) -> Param:
    return Param(values=frozenset(vs))


def multiple_of(m: int) -> Param:
    return Param(rule=lambda v: isinstance(v, int) and v > 0 and v % m == 0, describe=f"a positive multiple of {m}")


def any_positive() -> Param:
    return Param(rule=lambda v: v > 0, describe="any positive value (host-side)")


@dataclass(frozen=True)
class Template:
    name: str
    source: str                          # where the kernel lives (open_kernels/designs/...)
    params: dict[str, Param] = field(default_factory=dict)
    note: str = ""

    def require(self, **kw) -> None:
        for k, v in kw.items():
            if k not in self.params:
                raise OpRangeError(f"{self.name}: no parameter {k!r} (has {sorted(self.params)})")
            p = self.params[k]
            if not p.ok(v):
                raise OpRangeError(f"{self.name}: {k}={v!r} is outside the validated set {p.expected()}"
                                   + (f" ({self.note})" if self.note else ""))


CATALOGUE: dict[str, Template] = {t.name: t for t in [
    Template("gemv_q4", "designs/gemv_q4/gemv_q4.h",
             {"K": values(2048, 4096),           # the int16 activation tables that exist (gemv_q4_prep_k{K})
              "rs": values(2, 4),                # band row split: standard layout / expert stripes
              "rows_per_core": multiple_of(64),  # one y element per 64-row band
              "per_call": values(2)},            # chunks per w element (10 KB)
             note="a new K needs gemv_q4/make_test.py + compare.py at that K first"),
    Template("gemv_q4_prep_f32", "designs/gemv_q4/gemv_tab.h",
             {"K": values(512)},                 # the expert hidden h (moe_experts' law)
             note="validated through the whole-layer MoE block only"),
    Template("attn", "designs/attn/attn.h",
             {"head_dim": values(256), "num_heads": values(16), "num_kv_heads": values(2),
              "rotary_dim": values(64), "rope_theta": any_positive(),
              "qk_norm": values(True), "attn_gate": values(True)},
             note="ATTN_HD / ATTN_NH / ATTN_KVH / ATTN_ROT are compile-time macros; only this point has been compared"),
    Template("deltanet", "designs/layer_x/dnx.h",
             {"heads": values(32), "dim": values(128), "key_heads": values(16), "conv_kernel": values(4)},
             note="Qwen3-Next / 3.5 / 3.6 families only"),
    Template("ln", "designs/ln/ln.cc",
             {"width": values(2048)}),
    Template("router", "designs/router/router.h",
             {"experts": values(256), "topk": values(8)}),
    Template("moe", "designs/layer_x/moe_*.cc",
             {"ff": values(512), "experts": values(256), "topk": values(8), "shared_expert": values(True),
              "hidden": values(2048), "n_cores": values(8)}),
    Template("lm_head_q8", "designs/lm_head_q8/lm_head_q8.h",
             {"K": values(2048), "vocab": multiple_of(128)}),
]}


# Physical limits of one whole-layer design on the XDNA2 array (npu2, 8 columns):
# a generated design must CHECK these; they are not tunables.
LIMITS = {
    "max_buffer_args": 8,        # ERT state 6 above that (opcode-3 runs)
    "n_cols": 8,                 # main cores: one per column
    "shim_fills": 13,            # per design (phase 2 whole-layer plan)
    "shim_drains": 11,
    "program_bytes": 16384,      # per core; only the build can measure it
}


def require(op: str, **kw) -> Template:
    t = CATALOGUE.get(op)
    if t is None:
        raise OpRangeError(f"no template {op!r} in the catalogue (has {sorted(CATALOGUE)})")
    t.require(**kw)
    return t


def check_buffer_args(design: str, args: list[str]) -> None:
    if len(args) > LIMITS["max_buffer_args"]:
        raise OpRangeError(f"{design}: {len(args)} buffer arguments {args}, the runtime allows "
                           f"{LIMITS['max_buffer_args']}")
