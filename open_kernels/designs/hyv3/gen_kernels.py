"""Generate hy.py's kernel TUs (one extern "C" entry per file) for OPEN_KERNELS_SPEC (hyv3).

    python gen_kernels.py            # into this directory

The GEMVs are gemv_q4.h's q4_1 band walk (std_perm's 64-row bands, row split 2), as Flint's fl_* TUs:
hy_gy (a band into its y element: the attention), hy_gms (a slot's gate / up band into the scratch at
ms + dst) and hy_gacc (a down band into the scratch at ms + 128, then onto the core's slot accumulator,
fresh for slot 0). hy_act is w * silu(g) * u with the weight of the core's j-th gate | up slot under the
pair law (slot 2 j + half: cores 0-3 are half 0, cores 4-7 half 1; recipes/hyv3.py core_slots).
hy_wgrab lifts the 16-float weight record off the x element that carries xm; hy_copy hands an
accumulated band to the y fifo; hy_prep / hy_prepf build the activation table from bf16 / f32 elements.

Scratch `ms` (floats): u[64] @MS_U | g[64] @MS_G | a down band [64] @128.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # open_kernels/
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes import hyv3 as HR  # noqa: E402


def files(R) -> dict[str, str]:
    G, L = R.geo, R.layout
    q4 = f'''#define GEMV_PER_CALL {G.PER_CALL}
#include "gemv_q4.h"
'''
    w_off = L.AD_W - L.AD_XM - 4096 * (G.XM_ELEMS - 1)       # the weights' byte offset in xm's last element
    assert w_off == G.W_OFF and 0 <= w_off and w_off + 4 * HR.WV <= 4096
    assert G.MS_U == 0 and G.MS_G == 64
    return {
        "hy_gy.cc": q4 + '''// A q4_1 band into its y element (runtime band law: per_band chunks, row split rs).
extern "C" {
void hy_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
           int32_t group, int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
''',
        "hy_gms.cc": q4 + '''// A q4_1 64-row band of a slot's gate or up into the scratch at ms + dst (up at 0, gate at 64).
extern "C" {
void hy_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
''',
        "hy_gacc.cc": q4 + '''#include "vecmath.h"
// A q4_1 down band of slot e into the scratch at ms + 128 and, after the band's last group, onto
// band `band` of the core's slot accumulator (fresh for e == 0).
extern "C" {
void hy_gacc(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
             float *__restrict acc, int32_t group, int32_t per_band, int32_t band, int32_t e,
             int32_t ngroups) {
  float *__restrict y = ms + 128;
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, 2);
  if (group != ngroups - 1)
    return;
  float *__restrict a = acc + band * 64;
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 16) {
    const vfN<16> v = aie::load_v<16>(y + j);
    if (e == 0)
      aie::store_v(a + j, v);
    else
      aie::store_v(a + j, faddN<16>(aie::load_v<16>(a + j), v));
  }
}
}
''',
        "hy_act.cc": f'''// h band = w[2 j + half] * silu(g) * u for one 64-row band (ms: u @{G.MS_U}, g @{G.MS_G}) -> one f32 y
// element: the core's j-th gate | up slot under the pair law (recipes/hyv3.py core_slots). silu(x) =
// x sigmoid(x) as the dense dx computes it (vsiluN), then the slot's routing weight (1 for the shared
// expert and the dense MLP's slices).
#include "vecmath.h"

extern "C" {{
void hy_act(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t j, int32_t half) {{
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + {G.MS_U};
  const float *__restrict g = ms + {G.MS_G};
  const v32f w = aie::broadcast<float, 32>(wv[2 * j + half]);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 64; i += 32) {{
    const v32f a = vsiluN<32>(aie::load_v<32>(g + i));
    aie::store_v(h + i, fmul32(fmul32(a, aie::load_v<32>(u + i)), w));
  }}
}}
}}
''',
        "hy_wgrab.cc": f'''// The routing weights (f32[{HR.WV}], the host's hy_route, slot order) from xm's last x element, byte {w_off}.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {{
void hy_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {{
  aie::store_v(wv, aie::load_v<{HR.WV}>((const float *)((const uint8_t *)xe + {w_off})));
}}
}}
''',
        "hy_copy.cc": '''// One accumulated down band (64 floats of acc, band `band`) into its y element.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void hy_copy(const float *__restrict acc, float *__restrict y, int32_t band) {
  const float *__restrict a = acc + band * 64;
#pragma clang loop unroll(full)
  for (unsigned j = 0; j < 64; j += 16) aie::store_v(y + j, aie::load_v<16>(a + j));
}
}
''',
        "hy_prep.cc": '''// Element i of a bf16 activation of K values (2048 per 4 KB element) into the table: blocks
// [64 i, min(64 i + 64, K/32)).
#include "gemv_q4.h"

extern "C" {
void hy_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
''',
        "hy_prepf.cc": '''// Element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as bf16)
// into the table: blocks [32 i, min(32 i + 32, K/32)).
#include "gemv_q4.h"

extern "C" {
void hy_prepf(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
''',
    }


def generate(R, out: Path = HERE) -> int:
    fs = files(R)
    for name, text in fs.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return len(fs)


if __name__ == "__main__":
    spec = current_spec()
    n = generate(for_spec(spec).recipe(spec))
    print(f"wrote {n} TUs into {HERE}")
