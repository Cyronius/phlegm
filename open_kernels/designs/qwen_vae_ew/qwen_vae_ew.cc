// Qwen Image 2.1 VAE pixelwise operations on convolution-padded NHWC tiles.
#define NOCPP
#include <stdint.h>
#include <aie_api/aie.hpp>

#ifndef QVE_C
#define QVE_C 1152
#endif
#ifndef QVE_EL
#define QVE_EL 2304
#endif
#ifndef QVE_SILU
#define QVE_SILU 1
#endif

using VB = aie::vector<bfloat16, 32>;
using AC = aie::accum<accfloat, 32>;
constexpr int C = QVE_C, EL = QVE_EL;
static_assert(C % 32 == 0 && EL % C == 0);

static inline AC acc_of(VB x) {
  AC a;
  a.from_vector(x);
  return a;
}

using VF = aie::vector<float, 16>;
static inline VF fmul(VF a, VF b) { return aie::mul(a, b).to_vector<float>(); }
static inline VF fpoly(VF z2, float c1, float c2, float c3) {   // 1 + z2 (c1 + z2 (c2 + z2 c3))
  VF p = aie::add(aie::broadcast<float, 16>(c2), fmul(z2, aie::broadcast<float, 16>(c3)));
  p = aie::add(aie::broadcast<float, 16>(c1), fmul(z2, p));
  return aie::add(aie::broadcast<float, 16>(1.0f), fmul(z2, p));
}

// silu(x) = h (1 + tanh h), h = x / 2 (exact in bf16). The bf16 tanh LUT is ~2-5% off
// for |h| in 0.3-1, where the Qwen VAE encoder's small-gamma activations sit, so tanh is
// the [7/6] Pade z(135135 + 17325z^2 + 378z^4 + z^6) / (135135 + 62370z^2 + 3150z^4 +
// 28z^6) in fp32 on z = clamp(h, -5, 5) (absolute error < 1e-4; clamped in bf16, exact),
// its divide a bf16 reciprocal plus two fp32 Newton steps. Rounded to bf16 once.
static inline VF tanh_pade(VF z) {
  const VF z2 = fmul(z, z);
  const VF num = fpoly(z2, 0.128205128f, 0.0027972028f, 7.4000074e-6f);
  const VF den = fpoly(z2, 0.461538462f, 0.0233100233f, 2.0720207e-4f);
  aie::accum<accfloat, 16> da;
  da.from_vector(den);
  aie::accum<accfloat, 16> ra;
  ra.from_vector(aie::inv(da.to_vector<bfloat16>()));
  VF r = ra.to_vector<float>();
  const VF two = aie::broadcast<float, 16>(2.0f);
  r = fmul(r, aie::sub(two, fmul(den, r)));
  r = fmul(r, aie::sub(two, fmul(den, r)));
  return fmul(fmul(z, num), r);
}

static inline VB silu(VB x) {
  const VB h = aie::mul(x, (bfloat16)0.5f).to_vector<bfloat16>();
  const VB z = aie::min(aie::max(h, (bfloat16)-5.0f), (bfloat16)5.0f);
  const auto zf = acc_of(z).to_vector<float>();
  const auto hf = acc_of(h).to_vector<float>();
  const VF h0 = hf.extract<16>(0), h1 = hf.extract<16>(1);
  aie::vector<float, 32> out;
  out.insert(0, aie::add(h0, fmul(h0, tanh_pade(zf.extract<16>(0)))));
  out.insert(1, aie::add(h1, fmul(h1, tanh_pade(zf.extract<16>(1)))));
  AC o;
  o.from_vector(out);
  return o.to_vector<bfloat16>();
}

extern "C" {
void qve_rms(bfloat16 *__restrict x, bfloat16 *__restrict gamma,
             bfloat16 *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int pixel = 0; pixel < EL / C; ++pixel) {
    bfloat16 *__restrict xi = x + pixel * C;
    bfloat16 *__restrict yi = y + pixel * C;
    AC squares = aie::zeros<accfloat, 32>();
    for (int j = 0; j < C; j += 32) {
      VB v = aie::load_v<32>(xi + j);
      squares = aie::mac(squares, v, v);
    }
    const float sumsq = aie::reduce_add(squares.to_vector<float>());
    const float rs = aie::invsqrt(sumsq * (1.0f / C) + 1e-12f);
    const bfloat16 hi = (bfloat16)rs;
    const bfloat16 lo = (bfloat16)(rs - float(hi));
    for (int j = 0; j < C; j += 32) {
      VB v = aie::load_v<32>(xi + j);
      VB g = aie::load_v<32>(gamma + j);
      VB n = aie::mac(aie::mul(v, hi), v, lo).to_vector<bfloat16>();
      VB out = aie::mul(n, g).to_vector<bfloat16>();
#if QVE_SILU
      out = silu(out);
#endif
      aie::store_v(yi + j, out);
    }
  }
}

void qve_add(bfloat16 *__restrict x, bfloat16 *__restrict b,
             bfloat16 *__restrict y) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int j = 0; j < EL; j += 32)
    aie::store_v(y + j,
                 aie::add(acc_of(aie::load_v<32>(x + j)),
                          aie::load_v<32>(b + j)).to_vector<bfloat16>());
}
}
