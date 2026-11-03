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
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
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
  std::vector<float> out_t;   // output projection, transposed: [in][out]
  std::vector<Set> sets;
};

// Small kernels with restrict pointers, so GCC vectorizes them (it cannot inside the encode loop).
inline void add_row(float *__restrict acc, const float *__restrict src) {
  for (int i = 0; i < D; ++i) acc[i] += src[i];
}

// word + position + token-type embedding, then LayerNorm, into row r of an input buffer laid out
// [core][k / 8][token][8] (eight token rows a core; f2_pre). The same arithmetic as the model's
// TritEmbed.embed: fp32 products and sums, then LayerNorm. The moments are fp32 sums over 16 lanes (so they
// vectorize); torch's own LayerNorm sums in yet another order, so outputs agree to about 1e-6, not bitwise.
inline void embed_kernel(const int8_t *__restrict w, float ws, const int8_t *__restrict q, float ps,
                         const float *__restrict tt, const float *__restrict gamma, const float *__restrict beta,
                         float eps, float *__restrict base) {
  constexpr int L = 16;
  alignas(64) float v[D];
  alignas(64) float s[L] = {}, s2[L] = {};
  for (int i = 0; i < D; i += L)
    for (int j = 0; j < L; ++j) {
      v[i + j] = (float(w[i + j]) * ws + float(q[i + j]) * ps) + tt[i + j];
      s[j] += v[i + j];
    }
  float sum = 0;
  for (int j = 0; j < L; ++j) sum += s[j];
  const float fm = sum / D;
  for (int i = 0; i < D; i += L)
    for (int j = 0; j < L; ++j) {
      const float d = v[i + j] - fm;
      s2[j] += d * d;
    }
  float var = 0;
  for (int j = 0; j < L; ++j) var += s2[j];
  const float rstd = 1.0f / std::sqrt(var / D + eps);
  for (int g = 0; g < D / 8; ++g)
    for (int j = 0; j < 8; ++j) base[g * 64 + j] = (v[g * 8 + j] - fm) * rstd * gamma[g * 8 + j] + beta[g * 8 + j];
}

void embed_row(const Host &m, int64_t id, int64_t p, float *xb, int r) {
  if (id < 0 || id >= m.vocab || p < 0 || p >= m.max_pos) throw std::runtime_error("Token id or position out of range");
  embed_kernel(m.word.data() + id * D, m.word_scale[id], m.pos.data() + p * D, m.pos_scale[p], m.tok_type.data(),
               m.gamma.data(), m.beta.data(), m.eps, xb + size_t(r / 8) * D * 8 + (r % 8) * 8);
}

inline void prefetch_word(const Host &m, int64_t id) {
  if (id >= 0 && id < m.vocab) {
    const char *row = reinterpret_cast<const char *>(m.word.data() + id * D);
    for (int b = 0; b < D; b += 64) __builtin_prefetch(row + b);
  }
}

// G (1-4) pooled means [G][D] -> y [G][D] = mean @ W^T, with W^T [D][D]; each W^T row is read once for all G.
template <int G>
__attribute__((optimize("fp-contract=fast"))) void head_kernel(const float *__restrict mean, const float *__restrict wt, float *__restrict y) {
  alignas(64) float acc[G][D] = {};
  for (int c = 0; c < D; ++c) {
    const float *__restrict wr = wt + size_t(c) * D;
    for (int g = 0; g < G; ++g) {
      const float v = mean[g * D + c];
      for (int q = 0; q < D; ++q) acc[g][q] += v * wr[q];
    }
  }
  for (int g = 0; g < G; ++g)
    for (int q = 0; q < D; ++q) y[g * D + q] = acc[g][q];
}

// L2-normalize y into dst as F.normalize (eps 1e-12); false if a value is not finite.
inline bool normalize(const float *__restrict y, float *__restrict dst) {
  alignas(64) float ss[16] = {};
  for (int q = 0; q < D; q += 16)
    for (int l = 0; l < 16; ++l) ss[l] += y[q + l] * y[q + l];
  float norm = 0;
  for (float v : ss) norm += v;
  norm = std::max(std::sqrt(norm), 1e-12f);
  bool finite = std::isfinite(norm);
  for (int q = 0; q < D; ++q) dst[q] = y[q] / norm;
  return finite;
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

// ---- trit_embed host loop: packing, embedding, NPU runs, pooling and the output head in one call ----
//
// tdn_host_init copies the model's host tables (int8 word/position embeddings with fp32 row scales,
// the token-type row, the input LayerNorm, the 384x384 output projection) and the buffer sets (bound run
// id, segment-ID, input-state and output buffer indexes). tdn_encode then takes tokenized texts and does
// the rest with every set in flight: it packs texts into 128-token slots (trit_npu.pack's best-fit
// decreasing, so the same layout), and per dispatch embeds only the token rows in use straight into the
// input buffer in the cores' layout, syncs the touched cores and starts the run; meanwhile it mean-pools
// each finished run's texts, applies the output projection and L2-normalizes. Python only tokenizes.

TDN_EXPORT int tdn_host_init(void *h, const int8_t *word, const float *word_scale, int vocab,
                             const int8_t *pos, const float *pos_scale, int max_pos,
                             const float *tok_type, const float *gamma, const float *beta, float eps,
                             const float *out_weight, int sets, const int *run, const int *mask,
                             const int *x, const int *out, int out_floats) {
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
    m->out_t.resize(size_t(D) * D);
    for (int o = 0; o < D; ++o)
      for (int i = 0; i < D; ++i) m->out_t[size_t(i) * D + o] = out_weight[size_t(o) * D + i];
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

namespace {
// trit_npu.pack: best-fit decreasing into 128-token slots, ties as Python's stable sort and list.pop() give.
// Returns the slot count; slot[i] and offset[i] place text i.
int pack_slots(const std::vector<int> &len, std::vector<int> &slot, std::vector<int> &offset) {
  constexpr int SLOT = TOKENS / 2;
  const int n = static_cast<int>(len.size());
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return len[a] > len[b]; });
  std::vector<std::vector<int>> by_room(SLOT + 1);
  std::vector<int> room;
  slot.assign(n, 0);
  offset.assign(n, 0);
  for (int i : order) {
    const int need = len[i];
    int s = -1;
    for (int c = need; c <= SLOT; ++c)
      if (!by_room[c].empty()) {
        s = by_room[c].back();
        by_room[c].pop_back();
        break;
      }
    if (s < 0) {
      s = static_cast<int>(room.size());
      room.push_back(SLOT);
    }
    slot[i] = s;
    offset[i] = SLOT - room[s];
    room[s] -= need;
    by_room[room[s]].push_back(s);
  }
  return static_cast<int>(room.size());
}
}  // namespace

// n tokenized texts: text i is ids[offsets[i] .. offsets[i + 1]) ([CLS] .. [SEP], 1 to 128 ids). Its
// segment ID is seg_base + i + 1. Writes unit vectors[n][384] and *dispatches. Returns 0, or -1 with
// tdn_last_error.
TDN_EXPORT int tdn_encode(void *h, int n, const int32_t *ids, const int64_t *offsets, int seg_base,
                          float *vectors, int *dispatches) {
  last_error.clear();
  try {
    auto *t = static_cast<Td *>(h);
    Host *m = t->host;
    if (!m) throw std::runtime_error("tdn_host_init first");
    const int sets = static_cast<int>(m->sets.size());
    std::vector<int> len(n), slot, offset;
    for (int i = 0; i < n; ++i) {
      len[i] = static_cast<int>(offsets[i + 1] - offsets[i]);
      if (len[i] < 1 || len[i] > TOKENS / 2) throw std::runtime_error("A text needs 1 to 128 tokens");
    }
    const int slots = pack_slots(len, slot, offset);
    const int runs = (slots + 1) / 2;
    if (dispatches) *dispatches = runs;
    // texts by dispatch, in token order; first row of each text within its dispatch
    std::vector<int> order(n), row0(n);
    for (int i = 0; i < n; ++i) {
      order[i] = i;
      row0[i] = (slot[i] % 2) * (TOKENS / 2) + offset[i];
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return slot[a] / 2 != slot[b] / 2 ? slot[a] / 2 < slot[b] / 2 : row0[a] < row0[b];
    });
    std::vector<int> bound(runs + 1, n);
    for (int k = 0, j = 0; k <= runs; ++k) {
      while (j < n && slot[order[j]] / 2 < k) ++j;
      bound[k] = j;
    }

    using clk = std::chrono::steady_clock;
    static const bool prof = std::getenv("TDN_PROFILE") != nullptr;
    double tw = 0, tos = 0, tp = 0, te = 0, tis = 0, tst = 0;
    auto lap = [&](clk::time_point &a, double &acc) {
      auto b = clk::now();
      acc += std::chrono::duration<double, std::milli>(b - a).count();
      a = b;
    };

    auto finish = [&](int k) {
      const Set &st = m->sets[k % sets];
      auto a = clk::now();
      if (t->runs.at(st.run)->wait(std::chrono::milliseconds(20000)) != ERT_CMD_STATE_COMPLETED)
        throw std::runtime_error("Incomplete trit_embed dispatch");
      lap(a, tw);
      int used = 0;
      for (int j = bound[k]; j < bound[k + 1]; ++j) used = std::max(used, row0[order[j]] + len[order[j]]);
      const int cores = (used + 7) / 8;
      auto &ob = t->bufs.at(st.out);
      const size_t core_bytes = size_t(m->out_floats) * 4;
      for (int hf = 0; hf < 2; ++hf)   // output halves: tokens 0-3 and 4-7 of each core, all 32 cores each
        ob.sync(XCL_BO_SYNC_BO_FROM_DEVICE, cores * core_bytes, hf * 32 * core_bytes);
      lap(a, tos);
      const float *o = ob.map<const float *>();
      alignas(64) float acc[D], mean[4 * D], y[4 * D];
      for (int j0 = bound[k]; j0 < bound[k + 1]; j0 += 4) {
        const int g = std::min(4, bound[k + 1] - j0);
        for (int gi = 0; gi < g; ++gi) {   // mean-pool each text's final token states
          const int i = order[j0 + gi];
          std::fill(acc, acc + D, 0.f);
          for (int r = row0[i]; r < row0[i] + len[i]; ++r)
            add_row(acc, o + (size_t((r % 8) / 4) * 32 + r / 8) * m->out_floats + size_t(r % 4) * D);
          for (int ch = 0; ch < D; ++ch) mean[gi * D + ch] = static_cast<float>(double(acc[ch]) / double(len[i]));
        }
        switch (g) {   // output projection (no bias)
          case 4: head_kernel<4>(mean, m->out_t.data(), y); break;
          case 3: head_kernel<3>(mean, m->out_t.data(), y); break;
          case 2: head_kernel<2>(mean, m->out_t.data(), y); break;
          default: head_kernel<1>(mean, m->out_t.data(), y);
        }
        for (int gi = 0; gi < g; ++gi)
          if (!normalize(y + gi * D, vectors + size_t(order[j0 + gi]) * D))
            throw std::runtime_error("Nonfinite trit_embed output");
      }
      lap(a, tp);
    };

    // A second thread finishes runs (wait, read back, pool, head) in order while this one embeds and
    // starts the next; a set is refilled only after its previous run is finished.
    std::mutex mu;
    std::condition_variable cv;
    int started = 0, finished = 0;
    bool failed = false;
    std::exception_ptr error;
    std::thread finisher([&] {
      try {
        for (int k = 0; k < runs; ++k) {
          {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l, [&] { return started > k || failed; });
            if (failed) return;
          }
          finish(k);
          std::lock_guard<std::mutex> l(mu);
          finished = k + 1;
          cv.notify_all();
        }
      } catch (...) {
        std::lock_guard<std::mutex> l(mu);
        error = std::current_exception();
        failed = true;
        cv.notify_all();
      }
    });
    auto stop = [&](std::exception_ptr e) {
      {
        std::lock_guard<std::mutex> l(mu);
        if (!error) error = e;
        failed = true;
        cv.notify_all();
      }
      finisher.join();
      // drain whatever is still on the NPU, so the next call may reuse the sets
      for (int k = finished; k < started; ++k) t->runs.at(m->sets[k % sets].run)->wait(std::chrono::milliseconds(20000));
      std::rethrow_exception(error);
    };
    try {
    for (int k = 0; k < runs; ++k) {
      {
        std::unique_lock<std::mutex> l(mu);
        cv.wait(l, [&] { return finished >= k - sets + 1 || failed; });
        if (failed) break;
      }
      const Set &st = m->sets[k % sets];
      auto &mb = t->bufs.at(st.mask);
      auto &xb = t->bufs.at(st.x);
      int32_t *seg = mb.map<int32_t *>();
      std::fill(seg, seg + TOKENS, 0);
      float *x = xb.map<float *>();
      auto a = clk::now();
      int last = 0;
      for (int j = bound[k]; j < bound[k + 1]; ++j) {
        const int i = order[j];
        const int32_t *ti = ids + offsets[i];
        for (int p = 0; p < len[i]; ++p) {
          if (p + 4 < len[i]) prefetch_word(*m, ti[p + 4]);
          else if (j + 1 < bound[k + 1] && p + 4 - len[i] < len[order[j + 1]])
            prefetch_word(*m, ids[offsets[order[j + 1]] + p + 4 - len[i]]);
          seg[row0[i] + p] = seg_base + i + 1;
          embed_row(*m, ti[p], p, x, row0[i] + p);
        }
        last = std::max(last, row0[i] + len[i]);
      }
      lap(a, te);
      mb.sync(XCL_BO_SYNC_BO_TO_DEVICE, TOKENS * sizeof(int32_t), 0);
      xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t((last + 7) / 8) * 8 * D * 4, 0);
      lap(a, tis);
      t->started.at(st.run) = clk::now();
      t->runs.at(st.run)->start();
      lap(a, tst);
      std::lock_guard<std::mutex> l(mu);
      started = k + 1;
      cv.notify_all();
    }
    } catch (...) {
      stop(std::current_exception());
    }
    finisher.join();
    if (error) {
      for (int k = finished; k < started; ++k) t->runs.at(m->sets[k % sets].run)->wait(std::chrono::milliseconds(20000));
      std::rethrow_exception(error);
    }
    if (prof && runs)
      std::fprintf(stderr, "tdn_encode %d texts %d runs, ms/run: wait %.3f out-sync %.3f pool+head %.3f embed %.3f in-sync %.3f start %.3f\n",
                   n, runs, tw / runs, tos / runs, tp / runs, te / runs, tis / runs, tst / runs);
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
}
