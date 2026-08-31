// M4: barrier-free NPU decode driver experiment.
//
// Copied from decode_driver.cpp. Adds levers to investigate / remove the
// layer.xclbin "3 consecutive submissions then ERT_CMD_STATE_TIMEOUT" cap
// without the wasteful lm_head cross-context barrier:
//
//   1. QoS on a context:  xclbin <name> <path> [k=v,k=v,...]
//      Creates the hw_context with an xrt::hw_context::cfg_param_type (QoS)
//      map instead of the default. Keys: gops/fps/dma_bandwidth/latency/
//      frame_execution_time/priority (see xrt_hw_context.h).
//
//   2. Extra hw_context on an already-registered xclbin:
//         context <newName> <xclbinName> [k=v,...]
//      Lets two hw_contexts drive the SAME layer.xclbin so we can PING-PONG
//      layer runs across them. A submission to a different hw_context is what
//      resets the layer queue; a second layer context is a far cheaper reset
//      than a full lm_head (517 MB) vocab projection.
//
//   3. A generic cheap barrier: any `layer`/`lmhead`/`submit` directive works
//      on whichever context its kernel was created in, so a "barrier" is just
//      a run on the OTHER layer context.
//
// Everything else is identical to decode_driver.cpp (READ-ONLY original).
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(3); }
    size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v;
}
static size_t padup(size_t n) { const size_t a = 1024 * 1024; return (n + a - 1) / a * a; }

// Parse "k=v,k2=v2" into a QoS cfg_param map. Empty string -> empty map.
static xrt::hw_context::cfg_param_type parse_qos(const std::string& s) {
    xrt::hw_context::cfg_param_type qos;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string k = tok.substr(0, eq);
        uint32_t v = (uint32_t)std::stoul(tok.substr(eq + 1));
        qos[k] = v;
    }
    return qos;
}

struct Kernel {
    std::shared_ptr<xrt::elf> elf;
    std::shared_ptr<xrt::module> mod;
    std::shared_ptr<xrt::ext::kernel> kern;
};

static xrt::run make_run(Kernel& K, std::initializer_list<std::pair<int, xrt::ext::bo*>> args) {
    xrt::run run{*K.kern};
    run.set_arg(0, 3); run.set_arg(1, 0); run.set_arg(2, 0);
    for (auto& a : args) run.set_arg(a.first, *a.second);
    return run;
}

static ert_cmd_state submit(Kernel& K,
                            std::initializer_list<std::pair<int, xrt::ext::bo*>> args) {
    xrt::run run = make_run(K, args);
    run.start();
    return run.wait();
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: decode_driver_nobarrier <config>\n"); return 2; }
    std::ifstream cfg(argv[1]);
    if (!cfg) { printf("cannot open config\n"); return 2; }

    std::unique_ptr<xrt::device> dev;
    std::map<std::string, xrt::hw_context> ctxs;
    std::map<std::string, xrt::uuid> uuids;   // xclbinName -> uuid (for extra contexts)
    std::map<std::string, Kernel> kernels;
    std::map<std::string, std::pair<xrt::ext::bo, size_t>> bufs;

    std::unique_ptr<xrt::runlist> rl;
    std::vector<xrt::run> rl_runs;
    std::string rl_ctx;

    std::string line;
    int layeridx = 0;
    while (std::getline(cfg, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string cmd; ss >> cmd;
        try {
            if (cmd == "device") {
                dev = std::make_unique<xrt::device>(0);
                printf("device: %s\n", dev->get_info<xrt::info::device::name>().c_str());
            } else if (cmd == "xclbin") {
                std::string name, path, qosstr; ss >> name >> path; ss >> qosstr;
                xrt::xclbin xcl{path};
                auto uuid = dev->register_xclbin(xcl);
                uuids.emplace(name, uuid);
                auto qos = parse_qos(qosstr);
                if (qos.empty())
                    ctxs.emplace(name, xrt::hw_context(*dev, uuid));
                else
                    ctxs.emplace(name, xrt::hw_context(*dev, uuid, qos));
                printf("xclbin %s%s\n", name.c_str(), qos.empty() ? "" : " (qos)");
            } else if (cmd == "context") {
                // Additional hw_context bound to an already-registered xclbin.
                std::string name, xclbinName, qosstr; ss >> name >> xclbinName; ss >> qosstr;
                auto uuid = uuids.at(xclbinName);
                auto qos = parse_qos(qosstr);
                if (qos.empty())
                    ctxs.emplace(name, xrt::hw_context(*dev, uuid));
                else
                    ctxs.emplace(name, xrt::hw_context(*dev, uuid, qos));
                printf("context %s (on %s)%s\n", name.c_str(), xclbinName.c_str(),
                       qos.empty() ? "" : " (qos)");
            } else if (cmd == "kernel") {
                std::string name, xn, elfp; ss >> name >> xn >> elfp;
                Kernel k;
                k.elf = std::make_shared<xrt::elf>(elfp);
                k.mod = std::make_shared<xrt::module>(*k.elf);
                k.kern = std::make_shared<xrt::ext::kernel>(ctxs.at(xn), *k.mod, "MLIR_AIE");
                kernels.emplace(name, std::move(k));
                printf("kernel %s (%s)\n", name.c_str(), elfp.c_str());
            } else if (cmd == "buf") {
                std::string name, initf; size_t size;
                ss >> name >> size; ss >> initf;
                size_t ps = padup(size);
                xrt::ext::bo bo(*dev, ps);
                auto* host = bo.map<uint8_t*>();
                std::memset(host, 0, ps);
                if (!initf.empty()) {
                    auto d = read_file(initf);
                    std::memcpy(host, d.data(), d.size() < ps ? d.size() : ps);
                }
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                bufs.emplace(name, std::make_pair(std::move(bo), size));
            } else if (cmd == "load") {
                std::string name, initf; ss >> name >> initf;
                auto& b = bufs.at(name);
                auto* host = b.first.map<uint8_t*>();
                auto d = read_file(initf);
                std::memcpy(host, d.data(), d.size() < b.second ? d.size() : b.second);
                b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            } else if (cmd == "runlist") {
                std::string xn; ss >> xn;
                rl = std::make_unique<xrt::runlist>(ctxs.at(xn));
                rl_runs.clear();
                rl_ctx = xn;
            } else if (cmd == "layer") {
                std::string kn, pool, act, pack, side, state;
                ss >> kn >> pool >> act >> pack >> side >> state;
                auto& K = kernels.at(kn);
                if (rl) {
                    xrt::run r = make_run(K, {{3, &bufs.at(pool).first}, {4, &bufs.at(act).first},
                        {5, &bufs.at(pack).first}, {6, &bufs.at(side).first}, {7, &bufs.at(state).first}});
                    rl_runs.push_back(std::move(r)); rl->add(rl_runs.back());
                } else {
                    xrt::run r2 = make_run(K, {{3, &bufs.at(pool).first}, {4, &bufs.at(act).first},
                        {5, &bufs.at(pack).first}, {6, &bufs.at(side).first}, {7, &bufs.at(state).first}});
                    r2.start();
                    ert_cmd_state st = r2.wait();
                    printf("layer[%d] %s -> state %d\n", layeridx++, kn.c_str(), (int)st);
                    if (st != ERT_CMD_STATE_COMPLETED) { printf("LAYER FAILED\n"); return 1; }
                }
            } else if (cmd == "submit") {
                rl->execute();
                try { rl->wait(); }
                catch (const std::exception& e) { printf("RUNLIST FAILED: %s\n", e.what()); return 1; }
                printf("runlist[%zu layers on %s] -> completed\n", rl_runs.size(), rl_ctx.c_str());
                layeridx += (int)rl_runs.size();
                rl.reset(); rl_runs.clear();
            } else if (cmd == "lmhead") {
                std::string kn, logits, lmpool, act;
                ss >> kn >> logits >> lmpool >> act;
                auto& K = kernels.at(kn);
                ert_cmd_state st = submit(K, {{3, &bufs.at(logits).first},
                    {4, &bufs.at(lmpool).first}, {5, &bufs.at(act).first}});
                printf("lmhead %s -> state %d\n", kn.c_str(), (int)st);
                if (st != ERT_CMD_STATE_COMPLETED) return 1;
            } else if (cmd == "dump") {
                std::string name, outf; size_t size = 0;
                ss >> name >> outf; ss >> size;
                auto& b = bufs.at(name);
                if (!size) size = b.second;
                b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                std::ofstream of(outf, std::ios::binary);
                of.write((const char*)b.first.map<uint8_t*>(), size);
            } else if (cmd == "serve") {
                // Persistent decode loop with ping-pong contexts. Pools AND
                // states stay resident across tokens. The layer program is read
                // from config lines until "endserve"; it uses `runlist <ctx>` /
                // `layer` / `submit` (and optional `lmhead`) directives and is
                // expected to alternate two layer contexts in chunks of <=3 so
                // NO wasteful lm_head barrier is needed. Per stdin line
                // "step <act_in> <hidden_out>": load act, run program, dump
                // hidden (first 8192 bytes). "quit" exits.
                std::vector<std::string> prog;
                std::string pl;
                while (std::getline(cfg, pl)) {
                    if (pl == "endserve") break;
                    if (!pl.empty() && pl[0] != '#') prog.push_back(pl);
                }
                printf("SERVE READY\n"); fflush(stdout);
                std::string req;
                while (std::getline(std::cin, req)) {
                    std::istringstream rs(req);
                    std::string op; rs >> op;
                    if (op == "quit") break;
                    if (op != "step") continue;
                    std::string actin, hidout; rs >> actin >> hidout;
                    { auto& b = bufs.at("act");
                      auto* h = b.first.map<uint8_t*>();
                      auto d = read_file(actin);
                      std::memcpy(h, d.data(), d.size() < b.second ? d.size() : b.second);
                      b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
                    bool ok = true;
                    std::unique_ptr<xrt::runlist> prl;
                    std::vector<xrt::run> prl_runs;
                    for (auto& p : prog) {
                        std::istringstream ls(p);
                        std::string c; ls >> c;
                        if (c == "runlist") { std::string xn; ls >> xn; prl = std::make_unique<xrt::runlist>(ctxs.at(xn)); prl_runs.clear(); }
                        else if (c == "layer") {
                            std::string kn, pool, a, pack, side, state; ls >> kn >> pool >> a >> pack >> side >> state;
                            auto& K = kernels.at(kn);
                            prl_runs.push_back(make_run(K, {{3,&bufs.at(pool).first},{4,&bufs.at(a).first},{5,&bufs.at(pack).first},{6,&bufs.at(side).first},{7,&bufs.at(state).first}}));
                            prl->add(prl_runs.back());
                        } else if (c == "submit") {
                            prl->execute();
                            try { prl->wait(); } catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; break; }
                            prl.reset(); prl_runs.clear();
                        } else if (c == "lmhead") {
                            std::string kn, logits, lmpool, a; ls >> kn >> logits >> lmpool >> a;
                            submit(kernels.at(kn), {{3,&bufs.at(logits).first},{4,&bufs.at(lmpool).first},{5,&bufs.at(a).first}});
                        }
                    }
                    if (!ok) { printf("STEP ERR\n"); fflush(stdout); continue; }
                    { auto& b = bufs.at("act");
                      b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                      std::ofstream of(hidout, std::ios::binary);
                      of.write((const char*)b.first.map<uint8_t*>(), 8192); }
                    printf("STEP OK\n"); fflush(stdout);
                }
                printf("SERVE DONE\n");
            } else if (cmd == "loglogits") {
                std::string name; ss >> name;
                auto& b = bufs.at(name);
                b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const float* lg = b.first.map<const float*>();
                size_t n = 124160; bool finite = true; float amax = 0; size_t arg = 0;
                for (size_t i = 0; i < n; ++i) {
                    if (!std::isfinite(lg[i])) finite = false;
                    if (std::isfinite(lg[i]) && std::fabs(lg[i]) > amax) amax = std::fabs(lg[i]);
                    if (std::isfinite(lg[i]) && lg[i] > lg[arg]) arg = i;
                }
                printf("logits: finite=%d absmax=%.3f argmax_vocab=%zu\n", finite, amax, 2 * arg + 1);
            } else {
                printf("bad directive: %s\n", line.c_str()); return 2;
            }
        } catch (const std::exception& e) {
            printf("EXCEPTION on '%s': %s\n", line.c_str(), e.what());
            return 3;
        }
    }
    printf("DONE\n");
    return 0;
}
