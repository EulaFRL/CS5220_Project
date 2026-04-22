#include "data_loader.h"
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <mpi.h>

// Fashion-MNIST uses big-endian 32-bit integers in its headers.
static uint32_t read_be32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) <<  8) |  uint32_t(b[3]);
}

Dataset load_fashion_mnist(const std::string& dir, bool is_train) {
    const std::string prefix = is_train ? "train" : "t10k";
    const std::string img_path = dir + "/" + prefix + "-images-idx3-ubyte";
    const std::string lbl_path = dir + "/" + prefix + "-labels-idx1-ubyte";

    std::ifstream img_f(img_path, std::ios::binary);
    std::ifstream lbl_f(lbl_path, std::ios::binary);
    if (!img_f)
        throw std::runtime_error("Cannot open: " + img_path);
    if (!lbl_f)
        throw std::runtime_error("Cannot open: " + lbl_path);

    // Image file header
    uint32_t magic = read_be32(img_f);
    if (magic != 0x803)
        throw std::runtime_error("Bad image file magic: " + img_path);
    int n      = (int)read_be32(img_f);
    int rows   = (int)read_be32(img_f);
    int cols   = (int)read_be32(img_f);
    int n_feat = rows * cols;  // 784

    // Label file header
    read_be32(lbl_f);  // magic (0x801)
    read_be32(lbl_f);  // n (same as image n)

    Dataset ds;
    ds.n_samples  = n;
    ds.n_features = n_feat;
    ds.n_classes  = 10;
    ds.images.resize((size_t)n * n_feat);
    ds.labels.resize(n);

    // Read raw uint8 images and normalize to [0, 1]
    std::vector<uint8_t> raw(n * n_feat);
    img_f.read(reinterpret_cast<char*>(raw.data()), n * n_feat);
    for (int i = 0; i < n * n_feat; i++)
        ds.images[i] = raw[i] / 255.0f;

    // Read labels
    std::vector<uint8_t> raw_lbl(n);
    lbl_f.read(reinterpret_cast<char*>(raw_lbl.data()), n);
    for (int i = 0; i < n; i++)
        ds.labels[i] = raw_lbl[i];

    return ds;
}

Dataset scatter_dataset(const Dataset& full, int rank, int world_size) {
    // Broadcast metadata from rank 0
    int meta[3] = {0, 0, 0};  // {n_total, n_features, n_classes}
    if (rank == 0) {
        meta[0] = full.n_samples;
        meta[1] = full.n_features;
        meta[2] = full.n_classes;
    }
    MPI_Bcast(meta, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int n_total    = meta[0];
    int n_features = meta[1];
    int n_classes  = meta[2];

    if (n_total % world_size != 0) {
        // Truncate to the largest multiple of world_size
        // (lose at most world_size-1 samples, negligible for 60000)
        n_total = (n_total / world_size) * world_size;
    }
    int local_n = n_total / world_size;

    Dataset local;
    local.n_samples  = local_n;
    local.n_features = n_features;
    local.n_classes  = n_classes;
    local.images.resize((size_t)local_n * n_features);
    local.labels.resize(local_n);

    MPI_Scatter(
        rank == 0 ? full.images.data() : nullptr,
        local_n * n_features, MPI_FLOAT,
        local.images.data(),
        local_n * n_features, MPI_FLOAT,
        0, MPI_COMM_WORLD);

    MPI_Scatter(
        rank == 0 ? full.labels.data() : nullptr,
        local_n, MPI_INT,
        local.labels.data(),
        local_n, MPI_INT,
        0, MPI_COMM_WORLD);

    return local;
}
