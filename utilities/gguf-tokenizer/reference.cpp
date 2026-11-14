// Vocab-only reference: link to an independently built llama.cpp libllama.
// Reads JSON lines {text, add_special}; outputs one token-id array per line.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) { std::cerr << "vocab_only load failed\n"; return 1; }
    const auto * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        const auto input = nlohmann::json::parse(line);
        if (input.contains("stop_ids")) {
            std::vector<llama_token> stops;
            for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
                if (llama_vocab_is_eog(vocab, i)) stops.push_back(i);
            }
            std::cout << nlohmann::json({{"stop_ids", stops}}).dump() << '\n';
            continue;
        }
        const auto text = input.at("text").get<std::string>();
        const bool add_special = input.at("add_special").get<bool>();
        int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, true);
        std::vector<llama_token> ids(n < 0 ? -n : n);
        n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), add_special, true);
        if (n < 0) return 3;
        ids.resize(n);
        std::cout << nlohmann::json(ids).dump() << '\n';
    }
    llama_model_free(model);
}
