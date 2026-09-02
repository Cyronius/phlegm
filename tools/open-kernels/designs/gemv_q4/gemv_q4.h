#pragma once
//===- gemv_q4.h -------------------------------------------*- C++ -*-===//
//
// q4_1 GEMV on the AIE core against phlegm's POOL-ORDER chunks:
//   y[N] = W[N, K] @ x[K],  W in FLM's q4 form, chunks in the layer-pool order
//   the closed kernel consumes (npu-engine/src/pools.rs std_perm), so the same
//   kernel later streams straight out of the resident 512 MB pool.
//
// Inner tile arithmetic ported from vegah/LLMNpuTest designs/granite_gemv
// (Apache-2.0, see ../../LICENSE.LLMNpuTest). The chunk layout is identical:
//
//   chunk = 32 output rows x 256 K, 5120 B:
//     d  [256] bf16 at [0    : 512]    index kb*32 + r        (r = 0..31)
//     m  [256] bf16 at [512  : 1024]   same index
//     nib[4096] B at [1024 : 5120]     nibble (r/16)*4096 + k*16 + (r%16)
//                                      = elem(r, k); even index = low nibble
//   value = nib*d + m
//
// (npu-engine/src/q4nx.rs: nibble[(r/16)*4096 + bc*512 + i*16 + r%16], with
// k = bc*32 + i -- the same thing.)
//
// POOL ORDER (pools.rs std_perm, K = in_dim):
//   band = K/128 consecutive chunks = 64 output rows x K.
//   chunk c inside its band: row half = c % 2 (rows 0..31 / 32..63),
//                            k-tile   = c / 2 (cols 256*(c/2) ..).
// So one band is consumed as `per_band` chunks with a 64-float accumulator,
// and the entry point for chunk group g just walks (half, kt) off the index.
//
// Row index is the FASTEST axis inside a chunk: 16 consecutive nibbles are 16
// different rows at one k, so the 16 scales/mins load once per (rb, kb) and the
// inner loop is mask, to_float with a shifted binary point, zip, mac. No gather.
//
// The Q4_1 minimum factors out of the block sum:
//   y[r] = sum_kb { d[kb][r] * sum_{k in kb} nib[k][r]*x[k]  +  m[kb][r] * sum_{k in kb} x[k] }
//
// TRAPS (LLMNpuTest CLAUDE.md): default rounding is floor -> set conv_even;
// AIE2P has NO fp32 vector multiply (returns zero silently) -> split fp32 into
// two bf16 halves; no scalar float; aie::downshift on uint8 is an error -> mask
// + to_float shift; entry points one per TU; body noinline+inline (COMDAT) so
// the 16 KB program memory holds one copy.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kRowBlocks = 2;
static constexpr unsigned kRowsPerBlock = 16;
static constexpr unsigned kRows = 32;         // output rows per chunk
static constexpr unsigned kBandRows = 64;     // output rows per pool band
static constexpr unsigned kKBlocks = 8;       // 32-wide K blocks per chunk
static constexpr unsigned kKInBlock = 32;
static constexpr unsigned kTileK = 256;       // K per chunk
static constexpr unsigned kDBytes = 512;
static constexpr unsigned kMetaBytes = 1024;
static constexpr unsigned kTileBytes = 5120;

// Chunks per entry-point call (one ObjectFifo element = kPerCall * 5120 B,
// double-buffered in L1), chunks per pool band (= ROWSPLIT * K/256), and the
// band's row split: 2 for the standard layout (std_perm: 64-row bands,
// half = c%2, kt = c/2), 4 for the expert stripes / down experts (128-row
// bands, quarter = c%4, kt = c/4 -- pools.rs stripe_transpose / down_perm).
#ifndef GEMV_PER_CALL
#define GEMV_PER_CALL 4
#endif
#ifndef GEMV_PER_BAND
#define GEMV_PER_BAND 16
#endif
#ifndef GEMV_ROWSPLIT
#define GEMV_ROWSPLIT 2
#endif
static constexpr unsigned kPerCall = GEMV_PER_CALL;
static constexpr unsigned kPerBand = GEMV_PER_BAND;
static constexpr unsigned kRowSplit = GEMV_ROWSPLIT;

// `kt` selects the 256-wide slice of x; `first` starts the accumulator instead
// of adding to y (so the K tiles chain without a zeroing pass). Runtime args +
// noinline + inline: emitted ONCE (COMDAT) regardless of entry-point count.
__attribute__((noinline)) inline void gemv_q4_tile(const uint8_t *__restrict tile,
                                                   const bfloat16 *__restrict x,
                                                   unsigned kt, bool first,
                                                   float *__restrict y) {
  event0();
  aie::set_rounding(aie::rounding_mode::conv_even);

  const bfloat16 *__restrict dp = (const bfloat16 *)tile;
  const bfloat16 *__restrict mp = (const bfloat16 *)(tile + kDBytes);
  const uint8_t *__restrict nib = tile + kMetaBytes;
  const bfloat16 *__restrict xt = x + kt * kTileK;

  aie::accum<accfloat, kRowsPerBlock> acc[kRowBlocks];
  if (first) {
    acc[0] = aie::zeros<accfloat, kRowsPerBlock>();
    acc[1] = aie::zeros<accfloat, kRowsPerBlock>();
  } else {
    acc[0].from_vector(aie::load_v<kRowsPerBlock>(y));
    acc[1].from_vector(aie::load_v<kRowsPerBlock>(y + kRowsPerBlock));
  }

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKBlocks; ++kb) {
    // sum of x over this K block (the whole cost of the q4_1 minimum), fp32.
    aie::accum<accfloat, kKInBlock> xa;
    xa.from_vector(aie::load_v<kKInBlock>(xt + kb * kKInBlock));
    const float xs = aie::reduce_add(xa.template to_vector<float>());

#pragma clang loop unroll(disable)
    for (unsigned rb = 0; rb < kRowBlocks; ++rb) {
      const unsigned g = kb * kRows + rb * kRowsPerBlock;
      aie::vector<bfloat16, kRowsPerBlock> d16 = aie::load_v<kRowsPerBlock>(dp + g);
      aie::vector<bfloat16, kRowsPerBlock> m16 = aie::load_v<kRowsPerBlock>(mp + g);

      const uint8_t *__restrict src = nib + rb * 2048 + kb * 256;

      aie::accum<accfloat, kRowsPerBlock> part = aie::zeros<accfloat, kRowsPerBlock>();

#pragma clang loop unroll(disable)
      for (unsigned kk = 0; kk < kKInBlock; kk += 8) {
        aie::vector<uint8_t, 64> p = aie::load_v<64>(src + kk * 8);
        aie::vector<bfloat16, 64> flo =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0x0F, p), 0);
        aie::vector<bfloat16, 64> fhi =
            aie::to_float<bfloat16>(aie::bit_and((uint8_t)0xF0, p), 4);
        // low nibble = even weight index -> zip at chunk 1 restores k order
        auto [c0, c1] = aie::interleave_zip(flo, fhi, 1);

        const unsigned kbase = kb * kKInBlock + kk;
        part = aie::mac(part, c0.template extract<kRowsPerBlock>(0), xt[kbase + 0]);
        part = aie::mac(part, c0.template extract<kRowsPerBlock>(1), xt[kbase + 1]);
        part = aie::mac(part, c0.template extract<kRowsPerBlock>(2), xt[kbase + 2]);
        part = aie::mac(part, c0.template extract<kRowsPerBlock>(3), xt[kbase + 3]);
        part = aie::mac(part, c1.template extract<kRowsPerBlock>(0), xt[kbase + 4]);
        part = aie::mac(part, c1.template extract<kRowsPerBlock>(1), xt[kbase + 5]);
        part = aie::mac(part, c1.template extract<kRowsPerBlock>(2), xt[kbase + 6]);
        part = aie::mac(part, c1.template extract<kRowsPerBlock>(3), xt[kbase + 7]);
      }

      // acc += d*part + m*xs, every product bf16 x bf16 (no fp32 vector mul on
      // AIE2P): split each fp32 operand into hi + lo bf16 halves.
      aie::vector<bfloat16, kRowsPerBlock> hi = part.template to_vector<bfloat16>();
      aie::vector<bfloat16, kRowsPerBlock> lo =
          aie::sub(part, hi).template to_vector<bfloat16>();
      acc[rb] = aie::mac(acc[rb], hi, d16);
      acc[rb] = aie::mac(acc[rb], lo, d16);

      const bfloat16 xs_hi = (bfloat16)xs;
      const bfloat16 xs_lo = (bfloat16)(xs - (float)xs_hi);
      acc[rb] = aie::mac(acc[rb], m16, xs_hi);
      acc[rb] = aie::mac(acc[rb], m16, xs_lo);
    }
  }

  aie::store_v(y, acc[0].template to_vector<float>());
  aie::store_v(y + kRowsPerBlock, acc[1].template to_vector<float>());
  event1();
}

// One call = kPerCall consecutive POOL-ORDER chunks of one band, group `group`
// (chunks group*kPerCall .. +kPerCall-1 of the band). y is the band's 64-float
// accumulator; it is only complete after the last group of the band.
static inline void gemv_q4_pool_group(const uint8_t *__restrict chunks,
                                      const bfloat16 *__restrict x,
                                      unsigned group, float *__restrict y) {
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kPerCall; ++i) {
    const unsigned c = group * kPerCall + i;   // index within the band
    const unsigned part = c % kRowSplit;       // which 32-row slice of the band
    const unsigned kt = c / kRowSplit;
    gemv_q4_tile(chunks + i * kTileBytes, x, kt, kt == 0, y + part * kRows);
  }
}

#define GEMV_Q4_ENTRY__(P, B, R, N)                                         \
  void gemv_q4_p##P##b##B##r##R##_k##N(const uint8_t *__restrict t,         \
                                       const bfloat16 *__restrict x,        \
                                       float *__restrict y) {               \
    gemv_q4_pool_group(t, x, N, y);                                         \
  }
#define GEMV_Q4_ENTRY_(P, B, R, N) GEMV_Q4_ENTRY__(P, B, R, N)
#define GEMV_Q4_ENTRY(N) GEMV_Q4_ENTRY_(GEMV_PER_CALL, GEMV_PER_BAND, GEMV_ROWSPLIT, N)
