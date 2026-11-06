// Temporary process adapter over the externally supplied OpenFlowLM C++ Core.
// No copy of that engine is vendored here. See README.md for build provenance.
#include "open_qwen36/core.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using open_qwen36::Core;
using open_qwen36::CoreConfig;

static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void reply(json value) {
    value["ok"] = true;
    std::cout << value.dump() << '\n' << std::flush;
}
static int greedy(const Core& core) {
    const auto& logits = core.logits();
    const size_t n = core.real_vocab();
    if (!n || logits.size() < n) throw std::runtime_error("incomplete logits");
    size_t best = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits; trial discarded");
        if (logits[i] > logits[best]) best = i;
    }
    return static_cast<int>(best);
}
static std::vector<int> prompt_ids(const json& request, const Core& core) {
    const auto ids = request.at("prompt").get<std::vector<int>>();
    if (ids.empty() || ids.size() > core.max_ctx()) throw std::runtime_error("prompt must fit the context capacity");
    for (int id : ids) if (id < 0 || static_cast<size_t>(id) >= core.real_vocab())
        throw std::runtime_error("prompt token outside the model vocabulary");
    return ids;
}
static void run_prefill(Core& core, const std::vector<int>& ids, const std::string& route, bool want_logits = true) {
    if (static_cast<size_t>(core.position()) + ids.size() > core.max_ctx())
        throw std::runtime_error("prefill exceeds context capacity");
    if (ids.empty()) return;
    if (route == "layer-major") {
        core.step_gemm_prompt(ids, want_logits);
    } else if (route == "gemm-block") {
        const size_t block_size = core.gemm_block_t();
        if (!block_size) throw std::runtime_error("kernel set has no block prefill route");
        for (size_t at = 0; at < ids.size();) {
            const size_t real = std::min(block_size, ids.size() - at);
            std::vector<int> block(ids.begin() + at, ids.begin() + at + real);
            block.resize(block_size, block.back());
            core.step_gemm_block(block, real, want_logits && at + real == ids.size());
            at += real;
        }
    } else {
        for (size_t i = 0; i < ids.size(); ++i) core.step(ids[i], want_logits && i + 1 == ids.size());
    }
}

// Prompt reuse (SERVER-PROMPT-REUSE). Phlegm keeps the bridge's context between requests and
// prefills only what a new prompt adds. The KV rows sit at absolute positions, so moving the
// position back is enough for attention layers; recurrent layers (DeltaNet, short conv) cannot
// be run backwards, so their state is saved at a mark inside a prefill and restored from it.
struct Mark {
    bool valid = false;
    open_qwen36::Snapshot snap;  // the recurrent states and the position; no KV rows
};
static size_t recurrent_layers(const Core& core) {
    size_t n = 0;
    for (int l = 0; l < core.num_layers(); ++l) n += !core.is_attention_layer(l);
    return n;
}
static void set_mark(Core& core, Mark& mark) {
    mark.snap = core.checkpoint();
    for (auto& rows : mark.snap.kv) std::vector<uint8_t>().swap(rows);  // the rows stay in the cache
    mark.valid = true;
}
// Prefill `ids` from the current position, saving the state at the mark asked for: on the block
// routes the last block boundary at or before it (so the split adds no block), else the position
// itself. Returns the mark's position.
static size_t prefill_marked(Core& core, const std::vector<int>& ids, const std::string& route,
                             size_t want, Mark& mark) {
    const size_t pos0 = static_cast<size_t>(core.position());
    if (want < pos0 || want > pos0 + ids.size()) throw std::runtime_error("mark outside the prefill");
    size_t at = want - pos0;
    if (route != "sequential" && core.gemm_block_t()) at -= at % core.gemm_block_t();
    if (at == ids.size()) {
        run_prefill(core, ids, route);
        set_mark(core, mark);
    } else {
        run_prefill(core, std::vector<int>(ids.begin(), ids.begin() + at), route, false);
        set_mark(core, mark);
        run_prefill(core, std::vector<int>(ids.begin() + at, ids.end()), route);
    }
    return pos0 + at;
}
static int prefill(Core& core, const std::vector<int>& ids, const std::string& route) {
    run_prefill(core, ids, route);
    return greedy(core);
}
// Logits follow their reply line as raw binary32 bytes ("logits_f32_le_bytes" of them, little
// endian: the bridge builds only for x86-64), not as 1.3 MB of text inside the JSON.
static void reply_logits(json value, const Core& core) {
    const auto& logits = core.logits();
    const size_t n = core.real_vocab();
    if (!n || logits.size() < n) throw std::runtime_error("incomplete logits");
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
    static_assert(sizeof(float) == 4, "logit transport requires binary32");
    const unsigned one = 1;
    if (*reinterpret_cast<const unsigned char*>(&one) != 1) throw std::runtime_error("logit transport requires little endian");
    value["ok"] = true;
    value["logits_f32_le_bytes"] = n * sizeof(float);
    std::cout << value.dump() << '\n';
    std::cout.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(n * sizeof(float)));
    std::cout.flush();
}

int main() {
    // Rust owns this process and its device lock. Exit without XRT destruction:
    // after a dispatch failure, unwinding through driver objects can hang/crash.
    // The OS releases all process-owned device resources on exit.
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<Core> core;
    std::string route;
    Mark mark;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);  // the logits are raw bytes
#endif
    reply({{"protocol", 1}, {"engine", "openflowlm-cpp-reference"}});
    try {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.size() > 1'000'000) throw std::runtime_error("request exceeds 1 MB");
            const json request = json::parse(line);
            const std::string command = request.at("command").get<std::string>();
            if (command == "quit") { reply({}); std::_Exit(0); }
            if (command == "init") {
                if (core) throw std::runtime_error("engine already loaded");
                CoreConfig config;
                config.model_dir = request.at("model_dir").get<std::string>();
                config.kernel_dir = request.at("kernel_dir").get<std::string>();
                config.max_ctx = request.at("max_ctx").get<size_t>();
                config.timeout_ms = request.at("dispatch_timeout_ms").get<unsigned>();
                config.prefill_mode = request.at("prefill_mode").get<std::string>();
                config.verbose = false;
                if (config.max_ctx < 2 || config.max_ctx > 131072 || !config.timeout_ms)
                    throw std::runtime_error("invalid context capacity or dispatch timeout");
                if (!std::filesystem::is_regular_file(std::filesystem::path(config.kernel_dir) / "manifest.json"))
                    throw std::runtime_error("missing kernel manifest.json");
                const auto md = std::filesystem::path(config.model_dir);
                if (!std::filesystem::is_regular_file(md / "model.gguf") && !std::filesystem::is_regular_file(md / "model.q4nx"))
                    throw std::runtime_error("reference model directory needs model.gguf or model.q4nx");
                device = std::make_unique<xrt::device>(0u);
                core = std::make_unique<Core>(config, device.get());
                core->set_decode_route("npu");
                core->load_weights();
                const auto wanted = request.at("prefill_route").get<std::string>();
                if (wanted == "sequential") route = "sequential";
                else if (wanted == "auto") route = core->layer_major_ok() ? "layer-major" : (core->gemm_block_t() ? "gemm-block" : "sequential");
                else if (wanted == "block" && core->gemm_block_t()) route = "gemm-block";
                else throw std::runtime_error("unsupported requested prefill route");
                reply({{"device", device->get_info<xrt::info::device::name>()},
                    {"context_capacity", core->max_ctx()}, {"layers", core->num_layers()},
                    {"real_vocab", core->real_vocab()}, {"prefill_route", route},
                    {"weights_file", core->is_gguf() ? "model.gguf" : "model.q4nx"},
                    {"recurrent_layers", recurrent_layers(*core)}});
                continue;
            }
            if (!core) throw std::runtime_error("init is required before inference");
            if (command == "reset") { core->reset(); mark.valid = false; reply({}); }
            else if (command == "rewind") {
                // Back to `pos` <= the position: the mark's own position (its states restored), or
                // anywhere on a model without recurrent layers. Rows past `pos` are rewritten later.
                const size_t pos = request.at("pos").get<size_t>();
                if (pos > static_cast<size_t>(core->position())) throw std::runtime_error("rewind past the position");
                if (mark.valid && pos == static_cast<size_t>(mark.snap.pos)) core->restore(mark.snap);
                else if (recurrent_layers(*core)) throw std::runtime_error("rewind to a position without a mark");
                else if (pos == 0) core->reset();
                else core->seek(static_cast<int>(pos));
                if (mark.valid && pos < static_cast<size_t>(mark.snap.pos)) mark.valid = false;
                reply({{"position", core->position()}});
            } else if (command == "prefill") {
                const auto ids = prompt_ids(request, *core);
                reply({{"token", prefill(*core, ids, route)}});
            } else if (command == "decode") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                reply({{"token", greedy(*core)}});
            } else if (command == "prefill_logits") {
                // Continues from the current position (a fresh request resets first). With `mark`,
                // also saves the recurrent state there (prefill_marked) and replies with its position.
                const auto ids = prompt_ids(request, *core);
                if (mark.valid && static_cast<size_t>(core->position()) < static_cast<size_t>(mark.snap.pos))
                    mark.valid = false;
                if (request.contains("mark")) {
                    const size_t at = prefill_marked(*core, ids, route, request.at("mark").get<size_t>(), mark);
                    reply_logits({{"mark", at}}, *core);
                } else {
                    run_prefill(*core, ids, route);
                    reply_logits(json::object(), *core);
                }
            } else if (command == "decode_logits") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                reply_logits(json::object(), *core);
            } else if (command == "trial" || command == "survey_trial") {
                const auto ids = prompt_ids(request, *core);
                const bool survey = command == "survey_trial";
                const auto forced = survey ? request.at("decode_tokens").get<std::vector<int>>() : std::vector<int>{};
                for (int token : forced)
                    if (token < 0 || static_cast<size_t>(token) >= core->real_vocab())
                        throw std::runtime_error("survey decode token outside the model vocabulary");
                const size_t outputs = survey ? forced.size() + 1 : request.at("output_tokens").get<size_t>();
                if (outputs < 2 || outputs > 4096 || ids.size() + outputs - 1 > core->max_ctx())
                    throw std::runtime_error("invalid output length or insufficient context capacity");
                std::vector<double> latencies;
                latencies.reserve(outputs - 1);
                std::vector<int> greedy_tokens;
                if (survey) greedy_tokens.reserve(outputs);
                core->reset();                 // excluded; a trial never reuses a prompt
                mark.valid = false;
                const auto start = Clock::now();
                int previous = prefill(*core, ids, route);
                const double prefill_ms = elapsed(start);
                if (survey) greedy_tokens.push_back(previous);
                const auto decode_start = Clock::now();
                for (size_t step = 1; step < outputs; ++step) {
                    const auto token_start = Clock::now();
                    core->step(survey ? forced[step - 1] : previous, true); // synchronous dispatch + logit readback
                    previous = greedy(*core);  // EOS never stops a benchmark
                    latencies.push_back(elapsed(token_start));
                    if (survey) greedy_tokens.push_back(previous);
                }
                const double decode_ms = elapsed(decode_start);
                json result = {{"prompt_tokens", ids.size()}, {"output_tokens", outputs}, {"decode_steps", latencies.size()},
                    {"sample", {{"prefill_ms", prefill_ms}, {"decode_ms", decode_ms}, {"decode_token_ms", latencies}}}};
                if (survey) {
                    result["survey_protocol"] = "phlegm-engine-survey-v1";
                    result["greedy_tokens"] = greedy_tokens;
                }
                reply(std::move(result));
            } else throw std::runtime_error("unknown command");
        }
        std::_Exit(0);
    } catch (const std::exception& e) {
        std::cout << json({{"ok", false}, {"error", e.what()}}).dump() << '\n' << std::flush;
        std::_Exit(1);                         // a failed engine is never reused
    }
}
