/// kvslice_patch_test (no XRT): attnpos on a stream with per-core window fills (OPEN-ATTN-KVSLICE),
/// shaped as dx.py emits them for Gemma 4's local layers -- four fills of one 2- or 4-row element at
/// in-row offsets 0, 1, 2, 3 KB of an 8 KB row, a 1024-row sliding window -- and for its global
/// ones (two fills, no window): each fill's length becomes its own words a row times the window
/// padded to whole elements, each offset its base plus the window's first row.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "stream_patch.hpp"

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// One DDR transfer as the stream carries it: the BD blockwrite (op 1: register, then the length in
// words) and the address patch after it (op 0x81: the same register + 4, the buffer arg, the offset).
void transfer(std::vector<uint32_t>& w, uint32_t reg, uint32_t len_words, uint32_t arg, uint32_t off) {
    std::vector<uint32_t> bd(12, 0), patch(12, 0);
    bd[0] = 1;
    bd[2] = reg;
    bd[4] = len_words;
    patch[0] = 0x81;
    patch[6] = reg + 4;
    patch[8] = arg;
    patch[10] = off;
    w.insert(w.end(), bd.begin(), bd.end());
    w.insert(w.end(), patch.begin(), patch.end());
}

struct Kind {
    uint64_t kv_row, ptab_row, window, kvs_heads, hd;
    std::vector<uint64_t> bases;     // each fill's first kv head x hd x 2 B
};

void run(const char* name, const Kind& k, uint64_t rb) {
    const uint64_t slice = k.kvs_heads * k.hd * 2;            // bytes of one K (or V) slice
    const uint32_t elem_words = static_cast<uint32_t>(rb * 2 * slice / 4);
    std::vector<uint32_t> w(4, 0);
    uint32_t reg = 0x1d000;
    for (uint64_t b : k.bases) transfer(w, reg += 0x20, elem_words, 3, static_cast<uint32_t>(b));
    transfer(w, reg += 0x20, static_cast<uint32_t>(k.kv_row / 4), 3, static_cast<uint32_t>(k.kv_row));   // the new row
    transfer(w, reg += 0x20, static_cast<uint32_t>(k.ptab_row / 4), 5, 0x80000000u | static_cast<uint32_t>(k.ptab_row));
    stream_patch::AttnGeometry g;
    g.kv_row = k.kv_row;
    g.ptab_row = k.ptab_row;
    g.window = k.window;
    bool refused = false;
    try {
        stream_patch::attn_table(w, name, g);
    } catch (const std::exception&) {
        refused = true;
    }
    check(refused, std::string(name) + ": a plain table refuses several window fills");
    g.kvslice = rb;
    const auto t = stream_patch::attn_table(w, name, g);
    check(t.size() == 2 * k.bases.size() + 2, std::string(name) + ": a length and an offset patch a fill");
    for (uint64_t pos : {0ull, 1ull, 2ull, 7ull, 1023ull, 1024ull, 1025ull, 1500ull, 4095ull}) {
        std::vector<uint32_t> iw = w;
        stream_patch::attn_apply(iw.data(), t, pos, g);
        uint64_t start, nf;
        stream_patch::attn_window(pos, g.window, &start, &nf);
        const uint64_t rows = rb * ((nf + rb - 1) / rb);
        bool ok = true;
        for (size_t f = 0; f < k.bases.size(); ++f) {
            const size_t bd = 4 + f * 24;
            ok = ok && iw[bd + 4] == rows * 2 * slice / 4 && iw[bd + 12 + 10] == k.bases[f] + start * k.kv_row;
        }
        const size_t row = 4 + k.bases.size() * 24, rec = row + 24;
        ok = ok && iw[row + 12 + 10] == pos * k.kv_row && iw[rec + 12 + 10] == (0x80000000u | pos * k.ptab_row);
        ok = ok && start + rows <= std::max<uint64_t>(pos, 1) + rb - 1;   // within kvslice - 1 rows of slack
        check(ok, std::string(name) + " rb " + std::to_string(rb) + " at position " + std::to_string(pos) + ": " + std::to_string(rows) +
                      " rows from row " + std::to_string(start));
    }
}
}  // namespace

int main() {
    for (uint64_t rb : {2ull, 4ull}) {
        run("dx_local", {8192, 4096, 1024, 2, 256, {0, 1024, 2048, 3072}}, rb);
        run("dx", {4096, 2048, 0, 1, 512, {0, 1024}}, rb);
    }
    std::printf(failures ? "KVSLICE-PATCH FAIL\n" : "KVSLICE-PATCH OK\n");
    return failures ? 1 : 0;
}
