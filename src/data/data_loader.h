#pragma once
#include <vector>
#include <string>

struct Dataset {
    // Layout: images[sample * n_features + feature]
    // This is simultaneously (n_samples × n_features) row-major in C
    // and (n_features × n_samples) column-major in cuBLAS —
    // so a contiguous batch slice is already the right shape for GEMM.
    std::vector<float> images;
    std::vector<int>   labels;
    int n_samples  = 0;
    int n_features = 784;
    int n_classes  = 10;
};

// Load full dataset from IDX binary files (called by rank 0).
Dataset load_fashion_mnist(const std::string& data_dir, bool is_train);

// Rank 0 has full dataset; all ranks receive a local shard via MPI_Scatter.
// n_total must be divisible by world_size.
Dataset scatter_dataset(const Dataset& full, int rank, int world_size);
