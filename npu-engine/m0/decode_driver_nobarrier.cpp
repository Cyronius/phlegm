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
#include <chrono>

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(3); }
    size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v;
}
static size_t padup(size_t n) { const size_t a = 1024 * 1024; return (n + a - 1) / a * a; }

// Elapsed ms since the first call (steady_clock), shared by the `mark`
// directive and the per-step phase timestamps below.
static double mark_ms() {
    static auto t0 = std::chrono::steady_clock::now();
    static bool init = false;
    auto now = std::chrono::steady_clock::now();
    if (!init) { t0 = now; init = true; }
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

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

    // Per-token position poke (FLM's 480B ELF): template bytes + target ctx.
    // Configured by `poketpl <ctx> <template_path>`; armed when non-empty.
    // Per step with a <pos> arg: patch the 4 seqlen u32s at byte offsets
    // 160/184/208/232 (the only payload words that change per token; the other
    // changing words are the ELF build-id, which XRT does not verify), build a
    // fresh elf/module/kernel/run on the layer context, start+wait. FLM does
    // exactly this every token (m0c capture: fresh 480B ELF + kernel per step).
    std::vector<uint8_t> poke_tpl;
    std::vector<std::string> poke_ctxs;   // poked in order; tile memory is per-context
    auto run_poke = [&](uint32_t pos) {
        std::vector<uint8_t> e = poke_tpl;
        for (size_t off : {(size_t)160, (size_t)184, (size_t)208, (size_t)232})
            std::memcpy(e.data() + off, &pos, 4);
        xrt::elf elf(e.data(), e.size());
        xrt::module mod(elf);
        for (auto& cx : poke_ctxs) {
            xrt::ext::kernel k(ctxs.at(cx), mod, "MLIR_AIE");
            xrt::run r(k);
            r.set_arg(0, 3); r.set_arg(1, 0); r.set_arg(2, 0);
            r.start();
            ert_cmd_state st = r.wait();
            if (st != ERT_CMD_STATE_COMPLETED) return st;
        }
        return ERT_CMD_STATE_COMPLETED;
    };

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
                // "step <act_in> <hidden_out> [<logits_buf> <logits_out>]":
                // load act, run program, dump hidden (first 8192 bytes); with
                // the optional pair, also dump the logits buffer (496640 B =
                // full vocab 248320 bf16 -- the lm_head kernel's real output
                // layout) so an `lmhead` directive in the program replaces the
                // CPU vocab projection. "quit" exits.
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
                    std::string actin, hidout, lgbuf, lgout;
                    rs >> actin >> hidout >> lgbuf >> lgout;
                    printf("MARK recv %.3f\n", mark_ms());
                    { auto& b = bufs.at("act");
                      auto* h = b.first.map<uint8_t*>();
                      auto d = read_file(actin);
                      std::memcpy(h, d.data(), d.size() < b.second ? d.size() : b.second);
                      b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
                    printf("MARK h2d %.3f\n", mark_ms());
                    bool ok = true;
                    std::unique_ptr<xrt::runlist> prl;
                    std::vector<xrt::run> prl_runs;
                    int chunk_idx = 0;
                    std::string chunk_ctx;
                    for (auto& p : prog) {
                        std::istringstream ls(p);
                        std::string c; ls >> c;
                        if (c == "runlist") { std::string xn; ls >> xn; prl = std::make_unique<xrt::runlist>(ctxs.at(xn)); prl_runs.clear(); chunk_ctx = xn; }
                        else if (c == "layer") {
                            std::string kn, pool, a, pack, side, state; ls >> kn >> pool >> a >> pack >> side >> state;
                            auto& K = kernels.at(kn);
                            prl_runs.push_back(make_run(K, {{3,&bufs.at(pool).first},{4,&bufs.at(a).first},{5,&bufs.at(pack).first},{6,&bufs.at(side).first},{7,&bufs.at(state).first}}));
                            prl->add(prl_runs.back());
                        } else if (c == "submit") {
                            // Per-chunk timing: isolates whether the 14-ish
                            // ping-ponged submit/wait round trips per token cost
                            // roughly the same each (fixed submission overhead)
                            // or a few dominate (real compute / context-switch).
                            printf("MARK c%d_%s_n%zu_start %.3f\n", chunk_idx, chunk_ctx.c_str(), prl_runs.size(), mark_ms());
                            prl->execute();
                            try { prl->wait(); } catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; break; }
                            printf("MARK c%d_%s_n%zu_end %.3f\n", chunk_idx, chunk_ctx.c_str(), prl_runs.size(), mark_ms());
                            chunk_idx++;
                            prl.reset(); prl_runs.clear();
                        } else if (c == "lmhead") {
                            std::string kn, logits, lmpool, a; ls >> kn >> logits >> lmpool >> a;
                            printf("MARK lmh_start %.3f\n", mark_ms());
                            submit(kernels.at(kn), {{3,&bufs.at(logits).first},{4,&bufs.at(lmpool).first},{5,&bufs.at(a).first}});
                            printf("MARK lmh_end %.3f\n", mark_ms());
                        }
                    }
                    printf("MARK npu %.3f\n", mark_ms());
                    if (!ok) { printf("STEP ERR\n"); fflush(stdout); continue; }
                    { auto& b = bufs.at("act");
                      b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                      std::ofstream of(hidout, std::ios::binary);
                      of.write((const char*)b.first.map<uint8_t*>(), 8192); }
                    if (!lgbuf.empty() && !lgout.empty()) {
                        auto& b = bufs.at(lgbuf);
                        b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        std::ofstream of(lgout, std::ios::binary);
                        of.write((const char*)b.first.map<uint8_t*>(), 496640);
                    }
                    printf("MARK d2h %.3f\n", mark_ms());
                    printf("STEP OK\n"); fflush(stdout);
                }
                printf("SERVE DONE\n");
            } else if (cmd == "poketpl") {
                // poketpl <ctx[,ctx2,...]> <template_path> — comma-separated
                // context list; each gets its own poke run per token (tile
                // memory is per-hw_context).
                std::string xns, path; ss >> xns >> path;
                poke_tpl = read_file(path);
                poke_ctxs.clear();
                std::stringstream cs(xns); std::string one;
                while (std::getline(cs, one, ',')) if (!one.empty()) poke_ctxs.push_back(one);
                printf("poketpl %s (%zu B) on %s\n", path.c_str(), poke_tpl.size(), xns.c_str());
            } else if (cmd == "servep") {
                // Pipelined serve: same program language and step protocol as
                // `serve`, but each chunk's runlist is execute()d BEFORE the
                // previous chunk's wait() — the previous chunk's completion
                // overlaps the next chunk's host-side construction + submit.
                // Ping-pong contexts make the overlap legal w.r.t. the
                // 3-layer-per-context budget (each context still alternates);
                // whether the firmware tolerates overlapped occupancy is
                // exactly what this mode measures.
                std::vector<std::string> prog;
                std::string pl;
                while (std::getline(cfg, pl)) {
                    if (pl == "endserve") break;
                    if (!pl.empty() && pl[0] != '#') prog.push_back(pl);
                }
                printf("SERVE READY\n"); fflush(stdout);
                struct InFlight {
                    std::unique_ptr<xrt::runlist> rl;
                    std::vector<xrt::run> runs;   // keep alive until waited
                    int idx; std::string ctx;
                };
                std::string req;
                while (std::getline(std::cin, req)) {
                    std::istringstream rs(req);
                    std::string op; rs >> op;
                    if (op == "quit") break;
                    if (op != "step") continue;
                    std::string actin, hidout, lgbuf, lgout;
                    long long pos = -1;
                    rs >> actin >> hidout >> lgbuf >> lgout >> pos;
                    // Prefill steps pass "-" for the logits pair: lmhead
                    // directives and the logits D2H are skipped (the state
                    // update is all a prompt token needs).
                    bool skip_lm = (lgbuf == "-");
                    printf("MARK recv %.3f\n", mark_ms());
                    { auto& b = bufs.at("act");
                      auto* h = b.first.map<uint8_t*>();
                      auto d = read_file(actin);
                      std::memcpy(h, d.data(), d.size() < b.second ? d.size() : b.second);
                      b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
                    printf("MARK h2d %.3f\n", mark_ms());
                    bool ok = true;
                    if (pos >= 0 && !poke_tpl.empty()) {
                        printf("MARK poke_start %.3f\n", mark_ms());
                        ert_cmd_state st = run_poke((uint32_t)pos);
                        printf("MARK poke_end %.3f\n", mark_ms());
                        if (st != ERT_CMD_STATE_COMPLETED) { printf("STEP FAILED: poke state %d\n", (int)st); printf("STEP ERR\n"); fflush(stdout); continue; }
                    }
                    std::unique_ptr<xrt::runlist> prl;
                    std::vector<xrt::run> prl_runs;
                    std::string cur_ctx;
                    InFlight pend{};
                    int chunk_idx = 0;
                    for (auto& p : prog) {
                        std::istringstream ls(p);
                        std::string c; ls >> c;
                        if (c == "runlist") { std::string xn; ls >> xn; prl = std::make_unique<xrt::runlist>(ctxs.at(xn)); prl_runs.clear(); cur_ctx = xn; }
                        else if (c == "layer") {
                            std::string kn, pool, a, pack, side, state; ls >> kn >> pool >> a >> pack >> side >> state;
                            auto& K = kernels.at(kn);
                            prl_runs.push_back(make_run(K, {{3,&bufs.at(pool).first},{4,&bufs.at(a).first},{5,&bufs.at(pack).first},{6,&bufs.at(side).first},{7,&bufs.at(state).first}}));
                            prl->add(prl_runs.back());
                        } else if (c == "submit") {
                            printf("MARK c%d_%s_n%zu_start %.3f\n", chunk_idx, cur_ctx.c_str(), prl_runs.size(), mark_ms());
                            prl->execute();          // submit; do NOT wait yet
                            if (pend.rl) {           // wait the PREVIOUS chunk (overlaps current)
                                try { pend.rl->wait(); } catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; break; }
                                printf("MARK c%d_%s_n%zu_end %.3f\n", pend.idx, pend.ctx.c_str(), pend.runs.size(), mark_ms());
                            }
                            pend.rl = std::move(prl); pend.runs = std::move(prl_runs);
                            pend.idx = chunk_idx; pend.ctx = cur_ctx;
                            prl_runs.clear();
                            chunk_idx++;
                        } else if (c == "lmhead") {
                            if (pend.rl) {
                                try { pend.rl->wait(); } catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; break; }
                                printf("MARK c%d_%s_n%zu_end %.3f\n", pend.idx, pend.ctx.c_str(), pend.runs.size(), mark_ms());
                                pend.rl.reset(); pend.runs.clear();
                            }
                            if (skip_lm) continue;
                            std::string kn, logits, lmpool, a; ls >> kn >> logits >> lmpool >> a;
                            printf("MARK lmh_start %.3f\n", mark_ms());
                            submit(kernels.at(kn), {{3,&bufs.at(logits).first},{4,&bufs.at(lmpool).first},{5,&bufs.at(a).first}});
                            printf("MARK lmh_end %.3f\n", mark_ms());
                        }
                    }
                    if (ok && pend.rl) {   // program without trailing lmhead
                        try { pend.rl->wait(); }
                        catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; }
                        if (ok) printf("MARK c%d_%s_n%zu_end %.3f\n", pend.idx, pend.ctx.c_str(), pend.runs.size(), mark_ms());
                        pend.rl.reset(); pend.runs.clear();
                    }
                    printf("MARK npu %.3f\n", mark_ms());
                    if (!ok) { printf("STEP ERR\n"); fflush(stdout); continue; }
                    { auto& b = bufs.at("act");
                      b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                      std::ofstream of(hidout, std::ios::binary);
                      of.write((const char*)b.first.map<uint8_t*>(), 8192); }
                    if (!skip_lm && !lgbuf.empty() && !lgout.empty()) {
                        auto& b = bufs.at(lgbuf);
                        b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        std::ofstream of(lgout, std::ios::binary);
                        of.write((const char*)b.first.map<uint8_t*>(), 496640);
                    }
                    printf("MARK d2h %.3f\n", mark_ms());
                    printf("STEP OK\n"); fflush(stdout);
                }
                printf("SERVE DONE\n");
            } else if (cmd == "serveq") {
                // Prebuilt-serve: same program language and step protocol as
                // `serve`, but ALL runlists, runs, and arg bindings are built
                // ONCE here and re-executed per token (XRT: "the list can be
                // reused by calling execute() again"). Valid because every
                // layer arg is a resident BO whose binding never changes —
                // only buffer CONTENT changes between tokens. Kills the
                // per-token runlist ctor/dtor + run ctor + 5xset_arg churn
                // that `serve` pays 14x/token. lmhead becomes one persistent
                // run start()ed per token — FLM's own pattern (m0c capture).
                struct QChunk {
                    std::unique_ptr<xrt::runlist> rl;   // null => lmhead item
                    std::vector<xrt::run> runs;
                    std::string ctx;
                    std::unique_ptr<xrt::run> lmrun;
                };
                std::vector<QChunk> qprog;
                std::string pl;
                while (std::getline(cfg, pl)) {
                    if (pl == "endserve") break;
                    if (pl.empty() || pl[0] == '#') continue;
                    std::istringstream ls(pl);
                    std::string c; ls >> c;
                    if (c == "runlist") {
                        std::string xn; ls >> xn;
                        qprog.emplace_back();
                        qprog.back().rl = std::make_unique<xrt::runlist>(ctxs.at(xn));
                        qprog.back().ctx = xn;
                    } else if (c == "layer") {
                        std::string kn, pool, a, pack, side, state; ls >> kn >> pool >> a >> pack >> side >> state;
                        auto& K = kernels.at(kn);
                        auto& q = qprog.back();
                        q.runs.push_back(make_run(K, {{3,&bufs.at(pool).first},{4,&bufs.at(a).first},{5,&bufs.at(pack).first},{6,&bufs.at(side).first},{7,&bufs.at(state).first}}));
                        q.rl->add(q.runs.back());
                    } else if (c == "submit") {
                        // chunk boundary only; the runlist is already complete
                    } else if (c == "lmhead") {
                        std::string kn, logits, lmpool, a; ls >> kn >> logits >> lmpool >> a;
                        auto& K = kernels.at(kn);
                        qprog.emplace_back();
                        qprog.back().lmrun = std::make_unique<xrt::run>(
                            make_run(K, {{3,&bufs.at(logits).first},{4,&bufs.at(lmpool).first},{5,&bufs.at(a).first}}));
                    }
                }
                printf("SERVE READY\n"); fflush(stdout);
                std::string req;
                while (std::getline(std::cin, req)) {
                    std::istringstream rs(req);
                    std::string op; rs >> op;
                    if (op == "quit") break;
                    if (op != "step") continue;
                    std::string actin, hidout, lgbuf, lgout;
                    rs >> actin >> hidout >> lgbuf >> lgout;
                    printf("MARK recv %.3f\n", mark_ms());
                    { auto& b = bufs.at("act");
                      auto* h = b.first.map<uint8_t*>();
                      auto d = read_file(actin);
                      std::memcpy(h, d.data(), d.size() < b.second ? d.size() : b.second);
                      b.first.sync(XCL_BO_SYNC_BO_TO_DEVICE); }
                    printf("MARK h2d %.3f\n", mark_ms());
                    bool ok = true;
                    int ci = 0;
                    for (auto& q : qprog) {
                        if (q.rl) {
                            printf("MARK c%d_%s_n%zu_start %.3f\n", ci, q.ctx.c_str(), q.runs.size(), mark_ms());
                            q.rl->execute();
                            try { q.rl->wait(); } catch (const std::exception& e) { printf("STEP FAILED: %s\n", e.what()); ok = false; break; }
                            printf("MARK c%d_%s_n%zu_end %.3f\n", ci, q.ctx.c_str(), q.runs.size(), mark_ms());
                            ci++;
                        } else {
                            printf("MARK lmh_start %.3f\n", mark_ms());
                            q.lmrun->start();
                            ert_cmd_state st = q.lmrun->wait();
                            printf("MARK lmh_end %.3f\n", mark_ms());
                            if (st != ERT_CMD_STATE_COMPLETED) { printf("STEP FAILED: lmhead state %d\n", (int)st); ok = false; break; }
                        }
                    }
                    printf("MARK npu %.3f\n", mark_ms());
                    if (!ok) { printf("STEP ERR\n"); fflush(stdout); continue; }
                    { auto& b = bufs.at("act");
                      b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                      std::ofstream of(hidout, std::ios::binary);
                      of.write((const char*)b.first.map<uint8_t*>(), 8192); }
                    if (!lgbuf.empty() && !lgout.empty()) {
                        auto& b = bufs.at(lgbuf);
                        b.first.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        std::ofstream of(lgout, std::ios::binary);
                        of.write((const char*)b.first.map<uint8_t*>(), 496640);
                    }
                    printf("MARK d2h %.3f\n", mark_ms());
                    printf("STEP OK\n"); fflush(stdout);
                }
                printf("SERVE DONE\n");
            } else if (cmd == "mark") {
                // print elapsed ms since the FIRST mark (for per-token timing
                // within one process, immune to disk-cache startup variance)
                std::string label; ss >> label;
                printf("MARK %s %.3f\n", label.c_str(), mark_ms()); fflush(stdout);
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
