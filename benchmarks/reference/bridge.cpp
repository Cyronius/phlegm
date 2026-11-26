// Temporary process adapter over the externally supplied OpenFlowLM C++ Core.
// No copy of that engine is vendored here. See README.md for build provenance.
#include "open_qwen36/core.hpp"
#include "open_qwen36/vision/prompt.hpp"
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
    } else if (route == "gemm-block" && !core.mrope_active()) {   // the block-major route has no M-RoPE positions
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

// Image input (SERVER-IMAGE-INPUT, OPEN-VISION-BLOCK): phlegm resizes each image and expands its placeholder into
// the image's token run; the request lists the images and their RGB8 bytes follow the JSON line on stdin. The
// vision tower runs on the host and the prompt's rows go to the block route at their M-RoPE positions.
namespace vis = open_qwen36::vision;
struct Vision {
    bool on = false;
    vis::VitConfig cfg;
    std::string weights;                       // the tower's file, loaded at the first image
    std::unique_ptr<vis::VitWeights> w;
    long long min_pixels = 65536, max_pixels = 1003520;
};
static Vision vision;
static json vision_info(const Core& core, const std::string& model_dir) {
    vision = Vision{};
    if (!core.has_mrope() || core.image_token_id() < 0) return nullptr;
    const std::string file = vis::vision_weights_path(model_dir);
    if (!std::filesystem::is_regular_file(std::filesystem::path(model_dir) / file)) return nullptr;
    vision.cfg = vis::VitConfig::for_model_dir(model_dir);
    vision.weights = model_dir + "/" + file;
    vision.on = true;
    return {{"image_token_id", core.image_token_id()}, {"patch", vision.cfg.patch}, {"merge", vision.cfg.merge},
            {"min_pixels", vision.min_pixels}, {"max_pixels", vision.max_pixels},
            {"mean", {0.5, 0.5, 0.5}}, {"std", {0.5, 0.5, 0.5}},
            {"placeholder", "<|vision_start|><|image_pad|><|vision_end|>"}};
}
// The request's images: their RGB8 bytes follow the JSON line ("pixels_bytes" of them).
static std::vector<vis::PromptImage> read_images(const json& request) {
    std::vector<vis::PromptImage> out;
    if (!request.contains("images")) return out;
    if (!vision.on) throw std::runtime_error("this model does not accept images");
    const auto& list = request.at("images");
    const size_t total = request.at("pixels_bytes").get<size_t>();
    size_t want = 0;
    for (const auto& im : list) want += im.at("width").get<size_t>() * im.at("height").get<size_t>() * 3;
    if (want != total || list.empty() || list.size() > 64) throw std::runtime_error("images and pixels_bytes disagree");
    std::vector<uint8_t> bytes(total);
    std::cin.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(total));
    if (static_cast<size_t>(std::cin.gcount()) != total) throw std::runtime_error("short image payload");
    const float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    size_t off = 0;
    for (const auto& im : list) {
        vis::RgbImage rgb;
        rgb.width = im.at("width").get<int>();
        rgb.height = im.at("height").get<int>();
        const long long px = static_cast<long long>(rgb.width) * rgb.height;
        if (px > 4 * vision.max_pixels) throw std::runtime_error("image larger than the bridge's pixel bound");
        rgb.rgb.assign(bytes.begin() + off, bytes.begin() + off + static_cast<size_t>(px) * 3);
        off += static_cast<size_t>(px) * 3;
        out.push_back(vis::patchify(vision.cfg, rgb, mean, stdv));
    }
    return out;
}
// The image runs of ids: [start, end) of each run of image_token_id, in order.
static std::vector<std::pair<size_t, size_t>> image_runs(const Core& core, const std::vector<int>& ids) {
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t i = 0; i < ids.size();) {
        if (ids[i] != core.image_token_id()) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == core.image_token_id()) ++j;
        runs.emplace_back(i, j);
        i = j;
    }
    return runs;
}
static void prefill_with_images(Core& core, const std::vector<int>& ids, const std::vector<vis::PromptImage>& images,
                                const std::string& route, bool want_logits) {
    if (ids.empty()) return;
    if (!vision.w) {
        const auto t = Clock::now();
        vision.w = std::make_unique<vis::VitWeights>(vis::load_vit(vision.weights, vision.cfg));
        std::cerr << "bridge: vision tower resident in " << elapsed(t) << " ms\n";
    }
    const auto st = vis::prefill_images(core, vision.cfg, *vision.w, ids, images, route == "layer-major", want_logits);
    std::cerr << "bridge: " << images.size() << " image(s), " << st.image_tokens << " image tokens of " << ids.size()
              << ": vision tower " << st.vit_ms << " ms, prefill " << st.prefill_ms << " ms\n";
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
// prefill_marked with images: the mark never splits an image's run (it moves to the run's start), and each part
// takes the images whose runs it holds.
static size_t prefill_marked_images(Core& core, const std::vector<int>& ids, const std::vector<vis::PromptImage>& images,
                                    const std::string& route, size_t want, Mark& mark) {
    const size_t pos0 = static_cast<size_t>(core.position());
    if (want < pos0 || want > pos0 + ids.size()) throw std::runtime_error("mark outside the prefill");
    const size_t want_at = want - pos0;
    size_t at = want_at;
    if (route != "sequential" && core.gemm_block_t()) at -= at % core.gemm_block_t();
    const auto runs = image_runs(core, ids);
    if (runs.size() != images.size()) throw std::runtime_error("the prompt's image runs and the images disagree");
    size_t before = 0;
    for (const auto& [a, b] : runs) {
        if (a < at && at < b) at = b <= want_at ? b : a;   // past the image when the mark allows, else before it
        if (b <= at) ++before;
    }
    const std::vector<vis::PromptImage> head(images.begin(), images.begin() + before), tail(images.begin() + before, images.end());
    if (at == ids.size()) {
        prefill_with_images(core, ids, images, route, true);
        set_mark(core, mark);
    } else {
        prefill_with_images(core, std::vector<int>(ids.begin(), ids.begin() + at), head, route, false);
        set_mark(core, mark);
        prefill_with_images(core, std::vector<int>(ids.begin() + at, ids.end()), tail, route, true);
    }
    return pos0 + at;
}
static int prefill(Core& core, const std::vector<int>& ids, const std::string& route) {
    run_prefill(core, ids, route);
    return greedy(core);
}
// Protocol 2 (SERVER-SPECULATIVE): rows run as decode steps, so they are decode's logits; seeking back discards the rest.
constexpr size_t kVerifyWidth = 16;
struct Pending {
    bool active = false;
    size_t start = 0;
    size_t rows = 0;
    bool verify = false;   // the rows came from Core::step_verify: accepting commits a state slot
    std::vector<int> inputs;
};
// OPEN-SPEC-MTP-QWEN36MOE: the engine's MTP layer drafts from h, the main model's output for the row that produced
// the next input; each accepted row writes its MTP cache row, so the cache stays current.
struct Mtp {
    bool have_h = false;
    std::vector<float> h, row0, row1;
};
static Mtp mtp;
// QWEN36_BRIDGE_STATS=1: one stderr line per request with the speculation cycles and their costs
struct SpecStats {
    int passes = 0, accepted = 0, plain = 0;
    double draft_ms = 0, pass_ms = 0, commit_ms = 0, plain_ms = 0;
    void flush() {
        static const bool on = [] { const char* e = std::getenv("QWEN36_BRIDGE_STATS"); return e && std::string(e) == "1"; }();
        if (on && (passes || plain))
            std::cerr << "bridge stats: passes " << passes << " (both rows " << accepted << "), plain " << plain
                      << "; ms draft " << draft_ms << ", pass " << pass_ms << ", commit " << commit_ms
                      << ", plain " << plain_ms << '\n';
        *this = {};
    }
};
static SpecStats stats;
static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
static void mtp_after_step(Core& core) {
    if (!core.mtp_ready()) return;
    mtp.h = core.hidden_row(false, 0);
    mtp.have_h = true;
}
// A recurrent model verifies through the engine's verify pass (OPEN-VERIFY-ROWS-QWEN36MOE), whose width is fixed.
static size_t verify_width(const Core& core) { return recurrent_layers(core) ? core.verify_rows() : kVerifyWidth; }
static void accept_rows(Core& core, Pending& pending, size_t n) {
    if (!pending.active) throw std::runtime_error("accept without a pending verify pass");
    if (n > pending.rows) throw std::runtime_error("accept past the pending rows");
    pending.active = false;
    if (pending.verify) {
        const auto t = std::chrono::steady_clock::now();
        if (n) core.verify_commit(n);              // n == 0: the committed state was never touched
        if (core.mtp_ready() && n) {
            // row n - 1 gave the next input; with two rows, row 0's MTP cache row is written now
            if (n == 2) core.mtp_step(pending.inputs[1], mtp.row0.data(), static_cast<int>(pending.start), false);
            mtp.h = n == 2 ? mtp.row1 : mtp.row0;
            mtp.have_h = true;
        }
        stats.accepted += n == 2;
        stats.commit_ms += ms_since(t);
        return;
    }
    const size_t pos = pending.start + n;
    if (pos != static_cast<size_t>(core.position()) && recurrent_layers(core))
        throw std::runtime_error("a recurrent model's plain step cannot be taken back");
    if (pos != static_cast<size_t>(core.position())) core.seek(static_cast<int>(pos));
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
static void verify(Core& core, Pending& pending, const json& request) {
    auto tokens = request.at("tokens").get<std::vector<int>>();
    const int ndraft = request.value("draft", 0);
    if (ndraft) {
        if (!core.mtp_ready() || ndraft != 1 || tokens.size() != 1 || verify_width(core) != 2)
            throw std::runtime_error("this bridge drafts one MTP token after one input (OFLM_OPEN_MTP=1)");
        if (!mtp.have_h) throw std::runtime_error("an MTP draft needs a prefill or decode first");
        if (tokens[0] < 0 || static_cast<size_t>(tokens[0]) >= core.real_vocab())
            throw std::runtime_error("verify token outside the model vocabulary");
        const auto t = std::chrono::steady_clock::now();
        tokens.push_back(core.mtp_step(tokens[0], mtp.h.data(), core.position() - 1, true));
        stats.draft_ms += ms_since(t);
    }
    const size_t width = verify_width(core);
    if (!width) throw std::runtime_error("verify needs a model without recurrent layers, or the engine's verify pass (OFLM_OPEN_VERIFY=1)");
    if (tokens.empty() || tokens.size() > width)
        throw std::runtime_error("verify takes 1 to " + std::to_string(width) + " tokens");
    for (int id : tokens) if (id < 0 || static_cast<size_t>(id) >= core.real_vocab())
        throw std::runtime_error("verify token outside the model vocabulary");
    const size_t start = static_cast<size_t>(core.position()), n = core.real_vocab();
    if (start + tokens.size() > core.max_ctx()) throw std::runtime_error("verify exceeds context capacity");
    std::vector<float> rows(tokens.size() * n);
    // one input on a recurrent model is a plain step: the verify pass would cost two rows for it
    const bool recurrent = recurrent_layers(core) > 0, pass = recurrent && tokens.size() > 1;
    const auto t0 = std::chrono::steady_clock::now();
    if (recurrent && !pass && core.mtp_ready() && mtp.have_h)
        core.mtp_step(tokens[0], mtp.h.data(), static_cast<int>(start) - 1, false);   // keeps the MTP cache current
    if (pass) {
        // a short request fills the pass's rows with its last token; those rows are never accepted
        std::vector<int> ids(width, tokens.back()), am(width);
        std::copy(tokens.begin(), tokens.end(), ids.begin());
        core.step_verify(ids.data(), am.data(), true);
        if (core.mtp_ready()) {
            mtp.row0 = core.hidden_row(true, 0);
            mtp.row1 = core.hidden_row(true, 1);
        }
    }
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!pass) {
            core.step(tokens[i], true);
            if (recurrent) mtp_after_step(core);
        }
        const float* lp = pass ? core.verify_logits(i) : core.logits().data();
        const std::vector<float> logits(lp, lp + n);
        if (logits.size() < n) throw std::runtime_error("incomplete logits");
        for (size_t v = 0; v < n; ++v)
            if (!std::isfinite(logits[v])) throw std::runtime_error("nonfinite logits");
        std::copy(logits.begin(), logits.begin() + n, rows.begin() + i * n);
    }
    (pass ? stats.pass_ms : stats.plain_ms) += ms_since(t0);
    ++(pass ? stats.passes : stats.plain);
    pending = {true, start, tokens.size(), pass, tokens};
    json value = {{"ok", true}, {"rows", tokens.size()}, {"tokens", tokens}, {"position", start},
                  {"logits_f32_le_bytes", rows.size() * sizeof(float)}};
    std::cout << value.dump() << '\n';
    std::cout.write(reinterpret_cast<const char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof(float)));
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
    Pending pending;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);  // the logits are raw bytes
    _setmode(_fileno(stdin), _O_BINARY);   // so are the images' pixels
#endif
    reply({{"protocol", 2}, {"engine", "openflowlm-cpp-reference"}});
    try {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.size() > 1'000'000) throw std::runtime_error("request exceeds 1 MB");
            const json request = json::parse(line);
            const std::string command = request.at("command").get<std::string>();
            static const bool trace = [] { const char* e = std::getenv("QWEN36_BRIDGE_TRACE"); return e && *e; }();
            if (trace) {
                json t = request;
                if (t.contains("prompt")) t["prompt"] = t["prompt"].size();
                std::cerr << "bridge <- " << t.dump() << " at " << (core ? core->position() : -1) << std::endl;
            }
            if (command == "quit") { stats.flush(); reply({}); std::_Exit(0); }
            if (command == "reset" || command == "rewind" || command == "prefill_logits") stats.flush();
            if (command == "init") {
                if (core) throw std::runtime_error("engine already loaded");
                CoreConfig config;
                config.model_dir = request.at("model_dir").get<std::string>();
                config.kernel_dir = request.at("kernel_dir").get<std::string>();
                config.max_ctx = request.at("max_ctx").get<size_t>();
                config.timeout_ms = request.at("dispatch_timeout_ms").get<unsigned>();
                config.prefill_mode = request.at("prefill_mode").get<std::string>();
                config.verbose = false;
                // an older phlegm sends no mode: load whatever the kernels and GGUF allow
                const std::string spec = request.value("speculative", std::string("auto"));
                if (spec != "off" && spec != "lookup" && spec != "mtp" && spec != "auto")
                    throw std::runtime_error("unknown speculative mode " + spec);
                config.verify = spec != "off";
                config.mtp = spec == "mtp" || spec == "auto";
                // no upper cap: models reach 1M tokens, and the engine's allocations fail on a context too large
                if (config.max_ctx < 2 || !config.timeout_ms)
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
                    {"recurrent_layers", recurrent_layers(*core)},
                    {"verify_width", verify_width(*core)}, {"verify_ref", true},
                    {"draft", core->mtp_ready() && verify_width(*core) == 2 ? json{{"method", "mtp"}, {"max", 1}} : json(nullptr)},
                    {"vision", vision_info(*core, config.model_dir)}});
                continue;
            }
            if (!core) throw std::runtime_error("init is required before inference");
            if (command == "verify_logits") {
                if (pending.active) accept_rows(*core, pending, request.at("accept").get<size_t>());
                else if (request.at("accept").get<size_t>() != 0) throw std::runtime_error("accept without a pending verify pass");
                verify(*core, pending, request);
                continue;
            }
            if (command == "accept") {
                accept_rows(*core, pending, request.at("n").get<size_t>());
                reply({{"position", core->position()}});
                continue;
            }
            if (pending.active) throw std::runtime_error("a verify pass is pending; accept it first");
            if (command == "reset") { core->reset(); mark.valid = false; mtp.have_h = false; reply({}); }
            else if (command == "rewind") {
                mtp.have_h = false;
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
                const int tok = prefill(*core, ids, route);
                mtp_after_step(*core);
                reply({{"token", tok}});
            } else if (command == "decode") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                mtp_after_step(*core);
                reply({{"token", greedy(*core)}});
            } else if (command == "prefill_logits") {
                // Continues from the current position (a fresh request resets first). With `mark`,
                // also saves the recurrent state there (prefill_marked) and replies with its position.
                const auto ids = prompt_ids(request, *core);
                const auto images = read_images(request);   // the payload is read even if the prefill fails below
                if (mark.valid && static_cast<size_t>(core->position()) < static_cast<size_t>(mark.snap.pos))
                    mark.valid = false;
                if (!images.empty() || image_runs(*core, ids).size()) {
                    if (images.size() != image_runs(*core, ids).size())
                        throw std::runtime_error("the prompt's image runs and the request's images disagree");
                    if (request.contains("mark")) {
                        const size_t at = prefill_marked_images(*core, ids, images, route, request.at("mark").get<size_t>(), mark);
                        mtp_after_step(*core);
                        reply_logits({{"mark", at}}, *core);
                    } else {
                        prefill_with_images(*core, ids, images, route, true);
                        mtp_after_step(*core);
                        reply_logits(json::object(), *core);
                    }
                } else if (request.contains("mark")) {
                    const size_t at = prefill_marked(*core, ids, route, request.at("mark").get<size_t>(), mark);
                    mtp_after_step(*core);
                    reply_logits({{"mark", at}}, *core);
                } else {
                    run_prefill(*core, ids, route);
                    mtp_after_step(*core);
                    reply_logits(json::object(), *core);
                }
            } else if (command == "decode_logits") {
                const int token = request.at("token").get<int>();
                if (token < 0 || static_cast<size_t>(token) >= core->real_vocab()) throw std::runtime_error("invalid decode token");
                core->step(token, true);
                mtp_after_step(*core);
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
