// M3: execute a SEQUENCE of kernel runs sharing persistent buffers — the
// open-engine harness for FLM's fused decode kernels (layer.xclbin /
// lm_head.xclbin) and their staging/seqlen ctrlcode.
//
// Script file, one directive per line (# comments):
//   buf <name> <size> [initfile]          allocate BO (zeroed; optionally H2D from file)
//   ctx <name> <xclbin>                   register xclbin -> hw context
//   run <ctx> <elf> [argN=bufname ...]    build+start+wait one run (opcode 3)
//   dump <bufname> <outfile> [size]       D2H + write buffer (prefix) to file
// Buffers persist across runs; args are bound at positions 3.. as given.
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
#include <map>
#include <fstream>
#include <sstream>
#include <exception>

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(3); }
    size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v;
}
static uint64_t fnv1a(const uint8_t* b, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static size_t padup(size_t n) { const size_t a = 1024 * 1024; return (n + a - 1) / a * a; }

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: m3_chain <script>\n"); return 2; }
    std::ifstream script(argv[1]);
    if (!script) { printf("cannot open script %s\n", argv[1]); return 2; }
    try {
        xrt::device device(0);
        printf("device: %s\n", device.get_info<xrt::info::device::name>().c_str());
        std::map<std::string, xrt::hw_context> ctxs;
        std::map<std::string, std::pair<xrt::ext::bo, size_t>> bufs;
        std::string line;
        int runidx = 0;
        while (std::getline(script, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string cmd; ss >> cmd;
            if (cmd == "buf") {
                std::string name, initf; size_t size;
                ss >> name >> size; ss >> initf;
                size_t ps = padup(size);
                xrt::ext::bo bo(device, ps);
                auto* host = bo.map<uint8_t*>();
                std::memset(host, 0, ps);
                if (!initf.empty()) {
                    auto d = read_file(initf);
                    std::memcpy(host, d.data(), d.size() < ps ? d.size() : ps);
                }
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                bufs.emplace(name, std::make_pair(std::move(bo), size));
                printf("buf %-10s %zu%s\n", name.c_str(), size, initf.empty() ? "" : (" <= " + initf).c_str());
            } else if (cmd == "ctx") {
                std::string name, xp; ss >> name >> xp;
                xrt::xclbin xcl{xp};
                auto uuid = device.register_xclbin(xcl);
                ctxs.emplace(name, xrt::hw_context(device, uuid));
                printf("ctx %s = %s\n", name.c_str(), xp.c_str());
            } else if (cmd == "run") {
                std::string cn, elfp; ss >> cn >> elfp;
                xrt::elf elf{elfp};
                xrt::module mod{elf};
                xrt::ext::kernel kern{ctxs.at(cn), mod, "MLIR_AIE"};
                xrt::run run{kern};
                run.set_arg(0, 3); run.set_arg(1, 0); run.set_arg(2, 0);
                std::string kv;
                while (ss >> kv) {
                    auto eq = kv.find('=');
                    int arg = atoi(kv.substr(0, eq).c_str());
                    run.set_arg(arg, bufs.at(kv.substr(eq + 1)).first);
                }
                run.start();
                ert_cmd_state st = run.wait();
                printf("run[%d] %s -> state %d\n", runidx++, elfp.c_str(), (int)st);
                if (st != 4) { printf("RUN FAILED\n"); return 1; }
            } else if (cmd == "load") {
                std::string name, initf; ss >> name >> initf;
                auto& b = bufs.at(name);
                auto* host = b.first.map<uint8_t*>();
                auto d = read_file(initf);
                size_t n = d.size() < b.second ? d.size() : b.second;
                std::memcpy(host, d.data(), n);
                b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                printf("load %s <= %s (%zu B)\n", name.c_str(), initf.c_str(), n);
            } else if (cmd == "dump") {
                std::string name, outf; size_t size = 0;
                ss >> name >> outf; ss >> size;
                auto& b = bufs.at(name);
                if (!size) size = b.second;
                b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                auto* h = b.first.map<uint8_t*>();
                std::ofstream of(outf, std::ios::binary);
                of.write((const char*)h, size);
                size_t nz = 0; for (size_t k = 0; k < size; ++k) if (h[k]) nz++;
                printf("dump %s -> %s (nonzero=%zu hash=%016llx)\n", name.c_str(), outf.c_str(), nz,
                       (unsigned long long)fnv1a(h, size));
            } else {
                printf("bad directive: %s\n", line.c_str()); return 2;
            }
        }
        printf("DONE\n");
        return 0;
    } catch (const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 3;
    }
}
