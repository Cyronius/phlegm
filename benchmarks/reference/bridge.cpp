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
static void run_prefill(Core& core, const std::vector<int>& ids, const std::string& route) {
    if (static_cast<size_t>(core.position()) + ids.size() > core.max_ctx())
        throw std::runtime_error("prefill exceeds context capacity");
    if (route == "layer-major") {
        core.step_gemm_prompt(ids, true);
    } else if (route == "gemm-block") {
        const size_t block_size = core.gemm_block_t();
        if (!block_size) throw std::runtime_error("kernel set has no block prefill route");
        for (size_t at = 0; at < ids.size();) {
            const size_t real = std::min(block_size, ids.size() - at);
            std::vector<int> block(ids.begin() + at, ids.begin() + at + real);
            block.resize(block_size, block.back());
            core.step_gemm_block(block, real, at + real == ids.size());
            at += real;
        }
    } else {
        for (size_t i = 0; i < ids.size(); ++i) core.step(ids[i], i + 1 == ids.size());
    }
}
static int prefill(Core& core, const std::vector<int>& ids, const std::string& route) {
    run_prefill(core, ids, route);
    return greedy(core);
}
static std::string encode_base64(const unsigned char* data, size_t size) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t i = 0; i < size; i += 3) {
        const unsigned a = data[i];
        const unsigned b = i + 1 < size ? data[i + 1] : 0;
        const unsigned c = i + 2 < size ? data[i + 2] : 0;
        out.push_back(alphabet[a >> 2]);
        out.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
        out.push_back(i + 1 < size ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
        out.push_back(i + 2 < size ? alphabet[c & 63] : '=');
    }
    return out;
}
static json logits_reply(const Core& core) {
    const auto& logits = core.logits();
    const size_t n = core.real_vocab();
    if (!n || logits.size() < n) throw std::runtime_error("incomplete logits");
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite logits");
    static_assert(sizeof(float) == 4, "logit transport requires binary32");
    const auto* bytes = reinterpret_cast<const unsigned char*>(logits.data());
    return {{"logits_f32_le_base64", encode_base64(bytes, n * sizeof(float))}};
}

int main() {
    // Rust owns this process and its device lock. Exit without XRT destruction:
    // after a dispatch failure, unwinding through driver objects can hang/crash.
    // The OS releases all process-owned device resources on exit.
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<Core> core;
    std::string route;
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
                    {"weights_file", core->is_gguf() ? "model.gguf" : "model.q4nx"}});
                continue;
            }
            if (!core) throw std::runtime_error("init is required before inference");
            if (command == "reset") { core->reset(); reply({}); }
            else if (command == "prefill") {
                const auto ids = prompt_ids(request, *core);
                reply({{"token", prefill(*core, ids, route)}});
            } else if (command == "decode") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                reply({{"token", greedy(*core)}});
            } else if (command == "prefill_logits") {
                const auto ids = prompt_ids(request, *core);
                run_prefill(*core, ids, route);
                reply(logits_reply(*core));
            } else if (command == "decode_logits") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                reply(logits_reply(*core));
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
                core->reset();                 // excluded; no prefix caching in Core
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
