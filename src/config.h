#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Config {
    // Network architecture: first = n_features (784), last = n_classes (10)
    std::vector<int> layer_sizes = {784, 256, 128, 10};

    float lr          = 0.01f;
    int   batch_size  = 256;   // per rank
    int   epochs      = 10;

    // "mpi_builtin" | "ring"  (tree sum-reduction: see tree_reduce_test / comm/tree_reduce)
    std::string comm_algo = "mpi_builtin";

    std::string data_dir = "./data/fashion-mnist";
    bool verbose = false;
};

// Minimal CLI parser: --key value or --key=value
inline Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        auto eq = std::string(argv[i]);
        std::string key, val;

        auto eqpos = eq.find('=');
        if (eqpos != std::string::npos) {
            key = eq.substr(0, eqpos);
            val = eq.substr(eqpos + 1);
        } else if (i + 1 < argc) {
            key = eq;
            val = argv[++i];
        } else {
            continue;
        }

        if (key == "--lr")         cfg.lr         = std::stof(val);
        else if (key == "--batch") cfg.batch_size = std::stoi(val);
        else if (key == "--epochs")cfg.epochs     = std::stoi(val);
        else if (key == "--algo")  cfg.comm_algo  = val;
        else if (key == "--data")  cfg.data_dir   = val;
        else if (key == "--verbose") cfg.verbose  = true;
        else if (key == "--layers") {
            cfg.layer_sizes.clear();
            char buf[1024];
            strncpy(buf, val.c_str(), sizeof(buf) - 1);
            char* tok = strtok(buf, ",");
            while (tok) {
                cfg.layer_sizes.push_back(std::stoi(tok));
                tok = strtok(nullptr, ",");
            }
        }
    }
    return cfg;
}

inline void print_config(const Config& c, int rank) {
    if (rank != 0) return;
    printf("[Config] layers=");
    for (size_t i = 0; i < c.layer_sizes.size(); i++)
        printf("%d%s", c.layer_sizes[i], i+1 < c.layer_sizes.size() ? "," : "");
    printf("  lr=%.4f  batch=%d  epochs=%d  algo=%s\n",
           c.lr, c.batch_size, c.epochs, c.comm_algo.c_str());
}
