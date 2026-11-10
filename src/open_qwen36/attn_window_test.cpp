/// attn_window_test (no XRT, no fixture): the block attention's host half with Gemma 4's options
/// -- the q / k norms, a weightless V norm, the unit score scale, the proportional rotation
/// (rot / 2 pairs at a distance of hd / 2) and a sliding window -- against a direct fp64 loop.
/// The products the NPU runs (S = Q K^T over the bf16 cache, O = P V) are computed here in fp32
/// from the same bf16 operands, in window chunks as Core::attention_npu walks them, so the
/// window's first row, the chunk carry and the softmax mask are what is under test.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "open_qwen36/block_host.hpp"

using namespace open_qwen36;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Case {
    size_t T, t_real, nh, kvh, hd, rot, pair, pos0, window, chunk;
    double scale;
};

// og [t_real, nh*hd] the way attention_npu computes it, chunks of `chunk` rows from the first
// row the window reaches; kv is the cache after attention_prep
std::vector<float> npu_like(const host::AttnGeom& g, const std::vector<float>& Q, const std::vector<uint16_t>& kv,
                            size_t row, size_t chunk) {
    const size_t grp = g.nh / g.kvh, M = grp * g.T, qw = g.nh * g.hd, kvw = g.kvh * g.hd, hd = g.hd;
    const size_t rows = g.pos0 + g.t_real, first = host::attn_first_row(g);
    std::vector<float> og(g.t_real * qw, 0.f);
    std::vector<size_t> pos(M);
    for (size_t r = 0; r < M; ++r) pos[r] = g.pos0 + r % g.T;
    for (size_t gh = 0; gh < g.kvh; ++gh) {
        std::vector<float> m(M, -INFINITY), l(M, 0.f), acc(M * hd, 0.f);
        for (size_t c0 = first; c0 < rows; c0 += chunk) {
            const size_t L = std::min(chunk, rows - c0);
            std::vector<float> s(M * L);
            for (size_t r = 0; r < M; ++r) {
                const float* q = Q.data() + (r % g.T) * qw + (gh * grp + r / g.T) * hd;
                for (size_t j = 0; j < L; ++j) {
                    const uint16_t* k = kv.data() + (c0 + j) * row + gh * hd;
                    float hi = 0, lo = 0;     // Q as a bf16 hi / lo pair, two passes over one K^T
                    for (size_t d = 0; d < hd; ++d) {
                        const uint16_t qh = to_bf16(q[d]);
                        hi += from_bf16(qh) * from_bf16(k[d]);
                        lo += from_bf16(to_bf16(q[d] - from_bf16(qh))) * from_bf16(k[d]);
                    }
                    s[r * L + j] = hi + lo;
                }
            }
            std::vector<uint16_t> p(M * L);
            host::softmax_chunk(M, L, hd, c0, s.data(), pos.data(), m.data(), l.data(), acc.data(), p.data(), g.window);
            for (size_t r = 0; r < M; ++r)
                for (size_t j = 0; j < L; ++j) {
                    const float pj = from_bf16(p[r * L + j]);
                    const uint16_t* v = kv.data() + (c0 + j) * row + kvw + gh * hd;
                    for (size_t d = 0; d < hd; ++d) acc[r * hd + d] += pj * from_bf16(v[d]);
                }
        }
        for (size_t hl = 0; hl < grp; ++hl)
            for (size_t t = 0; t < g.t_real; ++t) {
                const size_t r = hl * g.T + t;
                for (size_t d = 0; d < hd; ++d) og[t * qw + (gh * grp + hl) * hd + d] = acc[r * hd + d] / l[r];
            }
    }
    return og;
}

double maxrel(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::fabs(a[i] - b[i]));
        den = std::max(den, std::fabs(b[i]));
    }
    return num / (den > 0 ? den : 1);
}

void run(const Case& c, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t qw = c.nh * c.hd, kvw = c.kvh * c.hd, row = 2 * kvw, cap = c.pos0 + c.T;
    std::vector<float> q(c.T * qw), k(c.T * kvw), v(c.T * kvw), qn(c.hd), kn(c.hd);
    for (auto* x : {&q, &k, &v}) for (auto& e : *x) e = 3.f * nd(rng);
    for (auto& e : qn) e = from_bf16(to_bf16(1.f + 0.3f * nd(rng)));
    for (auto& e : kn) e = from_bf16(to_bf16(1.f + 0.3f * nd(rng)));
    std::vector<double> inv(c.rot / 2);
    for (size_t i = 0; i < inv.size(); ++i) inv[i] = std::pow(1e4, -2.0 * static_cast<double>(i) / static_cast<double>(2 * c.pair));
    std::vector<uint16_t> kv(cap * row);
    for (auto& e : kv) e = to_bf16(nd(rng));                    // the cached rows before the block
    const std::vector<uint16_t> kv_in = kv;

    host::AttnGeom g;
    g.T = c.T; g.t_real = c.t_real; g.nh = c.nh; g.kvh = c.kvh; g.hd = c.hd; g.rot = c.rot; g.pos0 = c.pos0;
    g.eps = 1e-6; g.scale = c.scale; g.pair = c.pair; g.window = c.window; g.vnorm = true;
    std::vector<float> Q(c.T * qw);
    host::attention_prep(g, q.data(), k.data(), v.data(), qn.data(), kn.data(), inv.data(), kv.data(), row, Q.data());

    // the fp64 reference: norms, the rotation at distance `pair`, the window, an exact softmax
    auto rms = [&](const float* x, const float* w, double* out) {
        double ss = 0;
        for (size_t d = 0; d < c.hd; ++d) ss += static_cast<double>(x[d]) * x[d];
        const double r = 1.0 / std::sqrt(ss / static_cast<double>(c.hd) + 1e-6);
        for (size_t d = 0; d < c.hd; ++d) out[d] = x[d] * r * (w ? w[d] : 1.0);
    };
    auto rotate = [&](double* x, size_t p) {
        for (size_t i = 0; i < c.rot / 2; ++i) {
            const double a = static_cast<double>(p) * inv[i], x1 = x[i], x2 = x[c.pair + i];
            x[i] = x1 * std::cos(a) - x2 * std::sin(a);
            x[c.pair + i] = x2 * std::cos(a) + x1 * std::sin(a);
        }
    };
    std::vector<double> Kr(cap * kvw), Vr(cap * kvw);
    for (size_t r = 0; r < c.pos0; ++r)
        for (size_t j = 0; j < kvw; ++j) {
            Kr[r * kvw + j] = from_bf16(kv_in[r * row + j]);
            Vr[r * kvw + j] = from_bf16(kv_in[r * row + kvw + j]);
        }
    bool rows_ok = true;
    for (size_t t = 0; t < c.t_real; ++t) {
        const size_t p = c.pos0 + t;
        for (size_t h = 0; h < c.kvh; ++h) {
            double* kh = Kr.data() + p * kvw + h * c.hd;
            double* vh = Vr.data() + p * kvw + h * c.hd;
            rms(k.data() + t * kvw + h * c.hd, kn.data(), kh);
            rotate(kh, p);
            rms(v.data() + t * kvw + h * c.hd, nullptr, vh);
            for (size_t d = 0; d < c.hd; ++d) {        // the cache holds them in bf16, to an ulp
                const double kc = from_bf16(kv[p * row + h * c.hd + d]), vc = from_bf16(kv[p * row + kvw + h * c.hd + d]);
                rows_ok = rows_ok && std::fabs(kc - kh[d]) <= std::fabs(kh[d]) / 128 + 1e-6 &&
                          std::fabs(vc - vh[d]) <= std::fabs(vh[d]) / 128 + 1e-6;
                kh[d] = kc;                              // the reference attends over what is cached
                vh[d] = vc;
            }
        }
    }
    bool rest_ok = true;
    for (size_t i = 0; i < c.pos0 * row; ++i) rest_ok = rest_ok && kv[i] == kv_in[i];
    for (size_t i = (c.pos0 + c.t_real) * row; i < kv.size(); ++i) rest_ok = rest_ok && kv[i] == kv_in[i];
    std::vector<double> ref(c.t_real * qw, 0.0);
    const size_t grp = c.nh / c.kvh;
    for (size_t t = 0; t < c.t_real; ++t) {
        const size_t p = c.pos0 + t, lo = c.window && p + 1 > c.window ? p + 1 - c.window : 0;
        for (size_t h = 0; h < c.nh; ++h) {
            std::vector<double> qh(c.hd);
            rms(q.data() + t * qw + h * c.hd, qn.data(), qh.data());
            rotate(qh.data(), p);
            const size_t kh = h / grp;
            std::vector<double> s;
            double mx = -INFINITY;
            for (size_t r = lo; r <= p; ++r) {
                double d = 0;
                for (size_t j = 0; j < c.hd; ++j) d += qh[j] * Kr[r * kvw + kh * c.hd + j];
                s.push_back(d * c.scale);
                mx = std::max(mx, s.back());
            }
            double den = 0;
            for (auto& e : s) den += (e = std::exp(e - mx));
            for (size_t r = lo; r <= p; ++r)
                for (size_t j = 0; j < c.hd; ++j)
                    ref[t * qw + h * c.hd + j] += s[r - lo] / den * Vr[r * kvw + kh * c.hd + j];
        }
    }
    const auto og = npu_like(g, Q, kv, row, c.chunk);
    const std::string at = " (hd " + std::to_string(c.hd) + ", rot " + std::to_string(c.rot) + " at distance " +
                           std::to_string(c.pair) + ", window " + std::to_string(c.window) + ", pos0 " +
                           std::to_string(c.pos0) + ", chunks of " + std::to_string(c.chunk) + ")";
    check(rows_ok, "attention_prep: K normed + rotated, V normed, cached in bf16" + at);
    check(rest_ok, "attention_prep: rows before the block and past t_real are untouched" + at);
    const double e = maxrel(og, ref);
    check(e < 2e-2, "windowed products + softmax vs fp64 (maxrel " + std::to_string(e) + ")" + at);
    // a window at least as long as the cache changes nothing, bit for bit
    host::AttnGeom wide = g;
    wide.window = 0;
    const auto og0 = npu_like(wide, Q, kv, row, c.chunk);
    wide.window = cap + 1;
    const auto og1 = npu_like(wide, Q, kv, row, c.chunk);
    check(og0 == og1, "a window past every row is no window, bit for bit" + at);
    if (c.window && c.pos0 + c.t_real > c.window) check(og0 != og, "the window changes the output" + at);
}
}  // namespace

int main() {
    std::mt19937 rng(20261029);
    // Gemma 4 12B's two geometries, shrunk: local (rotation over the whole head, half-split, a window)
    // and global (a quarter of the head rotated, pairs half a head apart, no window)
    run({8, 6, 4, 2, 64, 64, 32, 7, 5, 4, 1.0}, rng);       // window starts inside the cached rows
    run({8, 8, 4, 2, 64, 64, 32, 2, 5, 3, 1.0}, rng);       // window starts at row 0 for the first tokens
    run({8, 5, 4, 1, 64, 16, 32, 9, 0, 6, 1.0}, rng);       // proportional rotation, no window
    run({8, 6, 4, 2, 64, 64, 32, 7, 5, 64, 0.125}, rng);    // one chunk, and a non-unit scale
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
