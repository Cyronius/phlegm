"""Generate the small kernel TUs of the dense layer design (one extern "C" entry per file).

    python gen_kernels.py            # for OPEN_KERNELS_SPEC (a qwen3 spec)

The GEMV band entry (gemv_q4_gy) is generated here with the recipe's chunks-per-element. Also: the up /
gate band into the silu scratch, silu(gate) * up for one 64-row band, and the
activation-table preps that take the ELEMENT index (the core loops over 4 KB
elements; the kernel derives the block range, so no arithmetic on loop indices
in the IRON body).

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes import dense as QR  # noqa: E402


def files(R) -> dict[str, str]:
    G = R.geo
    hdr = f'''#define GEMV_PER_CALL {G.PER_CALL}
#include "gemv_q4.h"
'''
    return {
        "gemv_q4_gy.cc": hdr + '''// A band into its y element: runtime band law (per_band chunks, row split rs).
extern "C" {
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "gemv_q4_gms.cc": hdr + f'''// A 64-row band into the silu scratch at ms + dst (the up band at {G.MS_U}, the gate band at {G.MS_G}).
extern "C" {{
void gemv_q4_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
                 int32_t group, int32_t per_band, int32_t dst) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}}
}}
''',
        "dense_silu.cc": f'''// h band = silu(g) * u for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}) -> one f32 y element.
#include "vecmath.h"

extern "C" {{
void dense_silu(const float *__restrict ms, float *__restrict h) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {G.MS_U};
  const float *__restrict g = ms + {G.MS_G};
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32)
    aie::store_v(h + j, fmul32(vsiluN<32>(aie::load_v<32>(g + j)), aie::load_v<32>(u + j)));
}}
}}
''',
        "dense_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void dense_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "dense_prep_f32.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void dense_prep_f32(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
    }


def generate(R, out: Path = HERE) -> int:
    fs = files(R)
    for name, src in fs.items():
        p = out / name
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
    return len(fs)


if __name__ == "__main__":
    n = generate(QR.recipe(current_spec()))
    print(f"{n} kernel files")
