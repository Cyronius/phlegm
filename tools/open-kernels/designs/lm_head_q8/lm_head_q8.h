#pragma once
//===- lm_head_q8.h ----------------------------------------*- C++ -*-===//
//
// W8A16 GEMV for the lm_head from phlegm's POOL-ORDER q8 chunks:
//   logits[248320] = W[248320, 2048] @ x[2048]
//
// Inner arithmetic ported from vegah/LLMNpuTest designs/lm_head (Apache-2.0,
// ../../LICENSE.LLMNpuTest); the chunk layout is FLM's q8 (q4nx.rs):
//
//   chunk = 32 output rows x 256 K, 8704 B:
//     scales[256] bf16 at [0   : 512]     index kb*32 + r          (r = 0..31)
//     codes [8192] int8 at [512 : 8704]   index (r/16)*4096 + k*16 + (r%16)
//   value = code * scale
//
// so `load_v<32>(s + kb*32)` is the 32 output rows in order, and two 16-lane
// code loads 4096 apart concatenate to those rows at one k: one 32-lane MAC
// per activation element, no gather.
//
// POOL ORDER (pools.rs build_lmhead_pool, "128-row supertile transpose"):
//   pool chunk k <- file chunk (4*(k/32) + (k%4))*8 + ((k%32)/4), i.e.
//   band = 32 consecutive chunks = 128 output rows x 2048 K;
//   chunk c inside its band: row quarter = c % 4, k-tile = c / 4.
// One band is consumed with a 128-float accumulator; the runtime `group`
// argument walks (quarter, kt) off the chunk index.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kRows = 32;        // output rows per chunk
static constexpr unsigned kKBlocks = 8;      // 32-wide K blocks per chunk
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;      // K per chunk
static constexpr unsigned kScaleBytes = 512;
static constexpr unsigned kRowBlockStride = 4096;  // in codes
static constexpr unsigned kTileBytes = 8704;
static constexpr unsigned kRowSplit = 4;     // 32-row quarters per 128-row band

#ifndef LMHEAD_PER_CALL
#define LMHEAD_PER_CALL 2
#endif
static constexpr unsigned kPerCall = LMHEAD_PER_CALL;

// Runtime kt/first, noinline + inline (COMDAT): one body in program memory.
__attribute__((noinline)) inline void gemv_q8_tile(const uint8_t *__restrict tile,
                                                   const bfloat16 *__restrict x,
                                                   unsigned kt, bool first,
                                                   float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict s = (const bfloat16 *)tile;
  const int8_t *__restrict c = (const int8_t *)(tile + kScaleBytes);
  const bfloat16 *__restrict xt = x + kt * kTileK;

  aie::accum<accfloat, kRows> acc;
  if (first)
    acc = aie::zeros<accfloat, kRows>();
  else
    acc.from_vector(aie::load_v<kRows>(y));

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    aie::vector<bfloat16, kRows> sv = aie::load_v<kRows>(s + kb * kRows);

    // int8 codes are exact in bf16, so this partial sum is exact in fp32.
    aie::accum<accfloat, kRows> part = aie::zeros<accfloat, kRows>();
#pragma clang loop unroll(disable)
    for (unsigned kk = 0; kk < kKInBlock; ++kk) {
      const unsigned k = kb * kKInBlock + kk;
      aie::vector<int8_t, 16> c0 = aie::load_v<16>(c + k * 16);
      aie::vector<int8_t, 16> c1 = aie::load_v<16>(c + kRowBlockStride + k * 16);
      part = aie::mac(part, aie::to_float<bfloat16>(aie::concat(c0, c1), 0), xt[k]);
    }

    // acc += part * scale with bf16 x bf16 products only (no fp32 vector mul
    // on AIE2P): split the fp32 partial into hi + lo bf16 halves.
    aie::vector<bfloat16, kRows> hi = part.template to_vector<bfloat16>();
    aie::vector<bfloat16, kRows> lo = aie::sub(part, hi).template to_vector<bfloat16>();
    acc = aie::mac(acc, hi, sv);
    acc = aie::mac(acc, lo, sv);
  }

  aie::store_v(y, acc.template to_vector<float>());
  event1();
}
