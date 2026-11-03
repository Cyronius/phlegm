// td_npu2: td_npu plus asynchronous runs bound to chosen buffers (tdn_bind/tdn_start/tdn_wait),
// so the host can prepare one batch while the NPU executes the previous one.
// td_npu: a C API over XRT for the tiny planner's host driver (run_planner.py, ctypes).
// One handle per (xclbin, insts.bin); the dispatch is open_kernels/harness's classic path.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_ext.h"

#ifdef _WIN32
#define TDN_EXPORT __declspec(dllexport)
#else
#define TDN_EXPORT __attribute__((visibility("default")))
#endif

namespace {
std::unique_ptr<xrt::device> g_dev;
thread_local std::string last_error;

constexpr int D = 384;        // model width
constexpr int TOKENS = 256;   // token rows a dispatch (two 128-token slots, eight rows a core)

struct Set {
  int run, mask, x, out;
};

struct Host {
  int vocab = 0, max_pos = 0, out_floats = 0;
  float eps = 0;
  std::vector<int8_t> word, pos;
  std::vector<float> word_scale, pos_scale, tok_type, gamma, beta;
  std::vector<Set> sets;
};

// word + position + token-type embedding, then LayerNorm, into row r of an input buffer laid out
// [core][k / 8][token][8] (eight token rows a core; f2_pre). The same arithmetic as the model's
// TritEmbed.embed: fp32 products and sums, then LayerNorm. The moments are fp32 sums over 16 lanes (so they
// vectorize); torch's own LayerNorm sums in yet another order, so outputs agree to about 1e-6, not bitwise.
void embed_row(const Host &m, int64_t id, int64_t p, float *xb, int r) {
  constexpr int L = 16;
  if (id < 0 || id >= m.vocab || p < 0 || p >= m.max_pos) throw std::runtime_error("Token id or position out of range");
  const int8_t *w = m.word.data() + id * D, *q = m.pos.data() + p * D;
  const float ws = m.word_scale[id], ps = m.pos_scale[p];
  const float *tt = m.tok_type.data();
  alignas(64) float v[D];
  alignas(64) float s[L] = {};
  for (int i = 0; i < D; i += L)
    for (int j = 0; j < L; ++j) {
      v[i + j] = (float(w[i + j]) * ws + float(q[i + j]) * ps) + tt[i + j];
      s[j] += v[i + j];
    }
  float sum = 0;
  for (int j = 0; j < L; ++j) sum += s[j];
  const float fm = sum / D;
  alignas(64) float s2[L] = {};
  for (int i = 0; i < D; i += L)
    for (int j = 0; j < L; ++j) {
      const float d = v[i + j] - fm;
      s2[j] += d * d;
    }
  float var = 0;
  for (int j = 0; j < L; ++j) var += s2[j];
  const float rstd = 1.0f / std::sqrt(var / D + m.eps);
  float *base = xb + size_t(r / 8) * D * 8 + (r % 8) * 8;
  for (int g = 0; g < D / 8; ++g)
    for (int j = 0; j < 8; ++j) {
      const int i = g * 8 + j;
      base[g * 64 + j] = (v[i] - fm) * rstd * m.gamma[i] + m.beta[i];
    }
}

struct Td {
  Host *host = nullptr;
  xrt::hw_context ctx;
  xrt::kernel kern;
  xrt::bo instr;
  int nwords = 0;
  std::vector<xrt::bo> bufs;
  std::unique_ptr<xrt::run> run;
  std::vector<std::unique_ptr<xrt::run>> runs;
  std::vector<std::chrono::steady_clock::time_point> started;
};
}  // namespace

extern "C" {

TDN_EXPORT const char *tdn_last_error() { return last_error.c_str(); }

TDN_EXPORT void *tdn_open(const char *xclbin, const char *insts) {
  last_error.clear();
  try {
    if (!g_dev) g_dev = std::make_unique<xrt::device>(0u);
    auto t = std::make_unique<Td>();
    xrt::xclbin x{std::string(xclbin)};
    t->ctx = xrt::hw_context(*g_dev, g_dev->register_xclbin(x));
    t->kern = xrt::kernel(t->ctx, "MLIR_AIE");
    std::ifstream f(insts, std::ios::binary);
    std::vector<char> w((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!f || w.empty() || w.size() % 4) throw std::runtime_error("Invalid instruction file");
    t->nwords = static_cast<int>(w.size() / 4);
    t->instr = xrt::bo(*g_dev, w.size(), xrt::bo::flags::cacheable, t->kern.group_id(1));
    std::memcpy(t->instr.map<void *>(), w.data(), w.size());
    t->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return t.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}

// a zeroed buffer at kernel argument 3 + its index; returns its host mapping
TDN_EXPORT void *tdn_buf(void *h, size_t size) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    size_t pad = (size + 4095) & ~size_t(4095);
    t->bufs.emplace_back(xrt::ext::bo(*g_dev, pad));
    auto *m = t->bufs.back().map<uint8_t *>();
    std::memset(m, 0, pad);
    t->bufs.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return m;

  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}

TDN_EXPORT void tdn_sync(void *h, int i, int to_device, size_t size, size_t off) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    t->bufs.at(i).sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, off);

  } catch (const std::exception &e) {
    last_error = e.what();
    return;
  }
}

// start + wait; wall ms, or -state on failure
TDN_EXPORT double tdn_run(void *h) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    if (!t->run) {
    t->run = std::make_unique<xrt::run>(t->kern);
    t->run->set_arg(0, 3);
    t->run->set_arg(1, t->instr);
    t->run->set_arg(2, t->nwords);
    for (size_t i = 0; i < t->bufs.size(); ++i) t->run->set_arg(static_cast<int>(3 + i), t->bufs[i]);
    }
    auto t0 = std::chrono::steady_clock::now();
    t->run->start();
    auto st = t->run->wait(std::chrono::milliseconds(20000));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return st == ERT_CMD_STATE_COMPLETED ? ms : -static_cast<double>(st);

  } catch (const std::exception &e) {
    last_error = e.what();
    return -1.;
  }
}

// A reusable run whose kernel buffer arguments are bufs[idx[0..n)]; returns its id, or -1.
TDN_EXPORT int tdn_bind(void *h, int n, const int *idx) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    auto r = std::make_unique<xrt::run>(t->kern);
    r->set_arg(0, 3);
    r->set_arg(1, t->instr);
    r->set_arg(2, t->nwords);
    for (int i = 0; i < n; ++i) r->set_arg(3 + i, t->bufs.at(idx[i]));
    t->runs.push_back(std::move(r));
    t->started.emplace_back();
    return static_cast<int>(t->runs.size() - 1);
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}

// Queue a bound run without waiting; 0 on success.
TDN_EXPORT int tdn_start(void *h, int id) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    t->started.at(id) = std::chrono::steady_clock::now();
    t->runs.at(id)->start();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}

// Wait for a started run; ms since its start (including queueing), or -state on failure.
TDN_EXPORT double tdn_wait(void *h, int id) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    auto st = t->runs.at(id)->wait(std::chrono::milliseconds(20000));
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t->started.at(id)).count();
    return st == ERT_CMD_STATE_COMPLETED ? ms : -static_cast<double>(st);
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1.;
  }
}

TDN_EXPORT void tdn_close(void *h) {
  auto *t = static_cast<Td *>(h);
  delete t->host;
  delete t;
}

// ---- trit_embed host loop: embedding, input LayerNorm, NPU runs and mean pooling in one call ----
//
// tdn_host_init copies the model's host tables (int8 word/position embeddings with fp32 row scales,
// the token-type row, the input LayerNorm) and the buffer sets (bound run id, segment-ID, input-state
// and output buffer indexes). tdn_encode then runs a whole layout (trit_npu.TritNpu.layout) with every
// set in flight: per dispatch it embeds only the token rows in use straight into the input buffer in
// the cores' layout, syncs the touched cores, starts the run, and mean-pools the finished run's texts.
// Python only tokenizes, packs, and applies the output projection.

TDN_EXPORT int tdn_host_init(void *h, const int8_t *word, const float *word_scale, int vocab,
                             const int8_t *pos, const float *pos_scale, int max_pos,
                             const float *tok_type, const float *gamma, const float *beta, float eps,
                             int sets, const int *run, const int *mask, const int *x, const int *out,
                             int out_floats) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    auto m = std::make_unique<Host>();
    m->vocab = vocab;
    m->max_pos = max_pos;
    m->eps = eps;
    m->out_floats = out_floats;
    m->word.assign(word, word + size_t(vocab) * D);
    m->word_scale.assign(word_scale, word_scale + vocab);
    m->pos.assign(pos, pos + size_t(max_pos) * D);
    m->pos_scale.assign(pos_scale, pos_scale + max_pos);
    m->tok_type.assign(tok_type, tok_type + D);
    m->gamma.assign(gamma, gamma + D);
    m->beta.assign(beta, beta + D);
    for (int s = 0; s < sets; ++s) {
      if (run[s] < 0 || run[s] >= static_cast<int>(t->runs.size())) throw std::runtime_error("Unknown run");
      m->sets.push_back({run[s], mask[s], x[s], out[s]});
    }
    delete t->host;
    t->host = m.release();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}

// One layout of `runs` dispatches. ids/pos/seg: [runs * TOKENS] (seg 0 = padding). Text i (0 .. n - 1)
// occupies flat token rows first[i] .. first[i] + length[i] - 1 of the layout, inside one dispatch; its
// mean-pooled final states go to pooled[row[i]] ([.., 384] fp32). Returns 0, or -1 with tdn_last_error.
TDN_EXPORT int tdn_encode(void *h, int runs, const int64_t *ids, const int64_t *pos, const int32_t *seg,
                          int n, const int64_t *first, const int64_t *length, const int64_t *row,
                          float *pooled) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    Host *m = t->host;
    if (!m) throw std::runtime_error("tdn_host_init first");
    const int sets = static_cast<int>(m->sets.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return first[a] < first[b]; });
    std::vector<int> bound(runs + 1, n);
    for (int k = 0, j = 0; k <= runs; ++k) {
      while (j < n && first[order[j]] < int64_t(k) * TOKENS) ++j;
      bound[k] = j;
    }
    std::vector<int> used(runs, 0);   // rows each dispatch must read back
    for (int k = 0; k < runs; ++k)
      for (int j = bound[k]; j < bound[k + 1]; ++j) {
        const int i = order[j];
        used[k] = std::max(used[k], static_cast<int>(first[i] - int64_t(k) * TOKENS + length[i]));
      }

    using clk = std::chrono::steady_clock;
    static const bool prof = std::getenv("TDN_PROFILE") != nullptr;
    double tw = 0, tos = 0, tp = 0, te = 0, tis = 0, tst = 0;
    auto lap = [&](clk::time_point &a, double &acc) { auto b = clk::now(); acc += std::chrono::duration<double, std::milli>(b - a).count(); a = b; };
    auto pool = [&](int k) {
      const Set &st = m->sets[k % sets];
      auto a = clk::now();
      if (t->runs.at(st.run)->wait(std::chrono::milliseconds(20000)) != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("Incomplete trit_embed dispatch");
      lap(a, tw);
      const int cores = (used[k] + 7) / 8;
      if (cores == 0) return;
      auto &ob = t->bufs.at(st.out);
      const size_t core_bytes = size_t(m->out_floats) * 4;
      for (int hf = 0; hf < 2; ++hf)   // output halves: tokens 0-3 and 4-7 of each core, all 32 cores each
        ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cores * core_bytes, hf * 32 * core_bytes);
      lap(a, tos);
      const float *o = ob.map<const float *>();
      float acc[D];
      for (int j = bound[k]; j < bound[k + 1]; ++j) {
        const int i = order[j];
        const int start = static_cast<int>(first[i] - int64_t(k) * TOKENS);
        std::fill(acc, acc + D, 0.f);
        for (int r = start; r < start + length[i]; ++r) {
          const int c = r / 8, w = r % 8;
          const float *src = o + (size_t(w / 4) * 32 + c) * m->out_floats + size_t(w % 4) * D;
          for (int ch = 0; ch < D; ++ch) acc[ch] += src[ch];
        }
        float *dst = pooled + size_t(row[i]) * D;
        for (int ch = 0; ch < D; ++ch) dst[ch] = static_cast<float>(double(acc[ch]) / double(length[i]));
      }
      lap(a, tp);
    };

    for (int k = 0; k < runs; ++k) {
      if (k >= sets) pool(k - sets);
      const Set &st = m->sets[k % sets];
      const int64_t *ki = ids + int64_t(k) * TOKENS, *kp = pos + int64_t(k) * TOKENS;
      const int32_t *ks = seg + int64_t(k) * TOKENS;
      auto &mb = t->bufs.at(st.mask);
      auto &xb = t->bufs.at(st.x);
      std::memcpy(mb.map<int32_t *>(), ks, TOKENS * sizeof(int32_t));
      float *x = xb.map<float *>();
      auto a = clk::now();
      int last = -1;
      for (int r = 0; r < TOKENS; ++r)
        if (ks[r]) {
          embed_row(*m, ki[r], kp[r], x, r);
          last = r;
        }
      lap(a, te);
      mb.sync(XCL_BO_SYNC_BO_TO_DEVICE, TOKENS * sizeof(int32_t), 0);
      xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t(last < 0 ? 1 : last / 8 + 1) * 8 * D * 4, 0);
      lap(a, tis);
      t->started.at(st.run) = std::chrono::steady_clock::now();
      t->runs.at(st.run)->start();
      lap(a, tst);
    }
    for (int k = std::max(0, runs - sets); k < runs; ++k) pool(k);
    if (prof && runs)
      std::fprintf(stderr, "tdn_encode %d runs, ms/run: wait %.3f out-sync %.3f pool %.3f embed %.3f in-sync %.3f start %.3f\n",
                   runs, tw / runs, tos / runs, tp / runs, te / runs, tis / runs, tst / runs);
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
}
