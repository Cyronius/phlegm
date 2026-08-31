// M0 Step 2b: replay one captured FLM op on the NPU and verify we reproduce
// FLM's exact output. Drives the kernel via FLM's open recipe (opcode=3,
// instr/ninstr=0, data BOs at args 3+) using a captured control ELF.
//
// usage:
//   m0_replay <xclbin> <elf> [--in ARG SIZE FILE] [--out ARG SIZE EXPECTED] [--zero ARG SIZE]
//     --in   : fill from captured bytes, sync H2D, bind
//     --out  : bind zeroed; after run, sync D2H and compare to captured bytes
//     --zero : bind zeroed, don't check (secondary outputs / scratch)
//     --io   : preload INF, after run compare vs EXPF (in-place op)
//     --dump : preload INFILE ("-" = zeros), after run D2H and write OUTFILE
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <exception>

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { printf("cannot open %s\n", p.c_str()); return {}; }
    size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v;
}
static uint64_t fnv1a(const uint8_t* b, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static size_t padup(size_t n) { const size_t a = 1024 * 1024; return (n + a - 1) / a * a; }

struct Spec { int arg; size_t size; std::string file; std::string expect; char role; }; // 'i' 'o' 'z' 'b'

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: m0_replay <xclbin> <elf> [--in A S F][--out A S F][--zero A S][--io A S INF EXPF]\n"); return 2; }
    std::string xclbin_path = argv[1], elf_path = argv[2];
    std::vector<Spec> specs;
    for (int i = 3; i < argc; ) {
        std::string f = argv[i];
        if (f == "--in")   { specs.push_back({atoi(argv[i+1]), (size_t)atoll(argv[i+2]), argv[i+3], "", 'i'}); i += 4; }
        else if (f == "--out") { specs.push_back({atoi(argv[i+1]), (size_t)atoll(argv[i+2]), argv[i+3], "", 'o'}); i += 4; }
        else if (f == "--zero"){ specs.push_back({atoi(argv[i+1]), (size_t)atoll(argv[i+2]), "", "", 'z'}); i += 3; }
        else if (f == "--io")  { specs.push_back({atoi(argv[i+1]), (size_t)atoll(argv[i+2]), argv[i+3], argv[i+4], 'b'}); i += 5; }
        else if (f == "--dump"){ specs.push_back({atoi(argv[i+1]), (size_t)atoll(argv[i+2]), argv[i+3], argv[i+4], 'd'}); i += 5; }
        else { printf("bad flag %s\n", f.c_str()); return 2; }
    }
    try {
        xrt::device device(0);
        printf("device: %s | xclbin: %s\n", device.get_info<xrt::info::device::name>().c_str(), xclbin_path.c_str());
        xrt::xclbin xcl{xclbin_path};
        auto uuid = device.register_xclbin(xcl);
        xrt::hw_context ctx(device, uuid);
        xrt::elf elf{elf_path};
        xrt::module mod{elf};
        xrt::ext::kernel kern{ctx, mod, "MLIR_AIE"};
        xrt::run run{kern};
        run.set_arg(0, 3); run.set_arg(1, 0); run.set_arg(2, 0);

        std::vector<xrt::ext::bo> bos; bos.reserve(specs.size());
        for (auto& s : specs) {
            size_t ps = padup(s.size);
            bos.emplace_back(device, ps);
            auto* host = bos.back().map<uint8_t*>();
            std::memset(host, 0, ps);
            if (s.role == 'i' || s.role == 'b' || (s.role == 'd' && s.file != "-")) {
                auto d = read_file(s.file);
                std::memcpy(host, d.data(), d.size() < s.size ? d.size() : s.size);
                bos.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
            run.set_arg(s.arg, bos.back());
            printf("  arg%d %c size=%zu\n", s.arg, s.role, s.size);
        }
        printf("submitting...\n");
        run.start();
        ert_cmd_state st = run.wait();
        printf("run state = %d (4=completed, 8=timeout)\n", (int)st);

        // dump requested buffers
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].role != 'd') continue;
            bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* oh = bos[i].map<uint8_t*>();
            std::ofstream of(specs[i].expect, std::ios::binary);
            of.write((const char*)oh, specs[i].size);
            size_t nz = 0; for (size_t k = 0; k < specs[i].size; ++k) if (oh[k]) nz++;
            printf("  DUMP arg%d -> %s (nonzero=%zu, hash=%016llx)\n", specs[i].arg,
                   specs[i].expect.c_str(), nz, (unsigned long long)fnv1a(oh, specs[i].size));
        }

        int checked = 0, matched = 0;
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].role != 'o' && specs[i].role != 'b') continue;
            checked++;
            bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto* oh = bos[i].map<uint8_t*>();
            uint64_t got = fnv1a(oh, specs[i].size);
            auto exp = read_file(specs[i].role == 'b' ? specs[i].expect : specs[i].file);
            uint64_t e = fnv1a(exp.data(), exp.size());
            bool ok = (got == e);
            matched += ok;
            size_t nz = 0; for (size_t k = 0; k < specs[i].size; ++k) if (oh[k]) nz++;
            printf("  OUT arg%d got=%016llx exp=%016llx nonzero=%zu  %s\n",
                   specs[i].arg, (unsigned long long)got, (unsigned long long)e, nz,
                   ok ? "*** MATCH ***" : "MISMATCH");
        }
        printf("RESULT state=%d outputs %d/%d matched\n", (int)st, matched, checked);
        return (st == 4 && matched == checked) ? 0 : 1;
    } catch (const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 3;
    }
}
