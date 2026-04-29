#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "config.h"
#include "timer.h"
#include "data/data_loader.h"
#include "mlp/mlp.h"
#include "comm/ring_allreduce.h"
#include "comm/tree_reduce.h"

// ---------------------------------------------------------------------------
// Convenience macro for CUDA in main()
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                \
        }                                                                \
    } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Parse CLI config
    Config cfg = parse_args(argc, argv);
    print_config(cfg, rank);

    if (cfg.comm_algo != "mpi_builtin" &&
        cfg.comm_algo != "ring" &&
        cfg.comm_algo != "tree") {
        if (rank == 0)
            fprintf(stderr, "[Error] Unknown --algo %s (use mpi_builtin|ring|tree).\n",
                    cfg.comm_algo.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Assign one GPU per MPI rank (round-robin within node)
    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    int local_rank = rank % n_gpus;
    CUDA_CHECK(cudaSetDevice(local_rank));

    // Print rank→GPU→hostname mapping from every rank so we can verify
    // cross-node assignment is correct (ranks packed consecutively per node).
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    // Serialize output: rank 0 prints first, then signals next rank
    for (int r = 0; r < world; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r)
            printf("[Init] rank %d → GPU %d @ %s\n", rank, local_rank, hostname);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("[Init] %d ranks total, %d GPUs per node\n", world, n_gpus);

    // -----------------------------------------------------------------------
    // Load & scatter training data
    // -----------------------------------------------------------------------
    Dataset train_full, test_full;
    if (rank == 0) {
        train_full = load_fashion_mnist(cfg.data_dir, /*is_train=*/true);
        test_full  = load_fashion_mnist(cfg.data_dir, /*is_train=*/false);
        printf("[Data] train=%d  test=%d  features=%d\n",
               train_full.n_samples, test_full.n_samples, train_full.n_features);
    }

    Dataset train_local = scatter_dataset(train_full, rank, world);

    int n_features = train_local.n_features;
    int n_classes  = train_local.n_classes;
    int local_n    = train_local.n_samples;

    if (rank == 0)
        printf("[Scatter] %d samples per rank\n", local_n);

    // Copy local training data to device (stays there for all epochs)
    float* d_train_images;
    int*   d_train_labels;
    CUDA_CHECK(cudaMalloc(&d_train_images,
                          (size_t)local_n * n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_train_labels,
                          local_n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_train_images, train_local.images.data(),
                          (size_t)local_n * n_features * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_labels, train_local.labels.data(),
                          local_n * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Copy test data to device on rank 0 only
    float* d_test_images = nullptr;
    int*   d_test_labels = nullptr;
    int    n_test        = 0;
    if (rank == 0) {
        n_test = test_full.n_samples;
        CUDA_CHECK(cudaMalloc(&d_test_images,
                              (size_t)n_test * n_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_test_labels, n_test * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_test_images, test_full.images.data(),
                              (size_t)n_test * n_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_test_labels, test_full.labels.data(),
                              n_test * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // Build MLP
    // -----------------------------------------------------------------------
    if (cfg.layer_sizes.front() != n_features) {
        if (rank == 0)
            fprintf(stderr, "[Error] layer_sizes[0]=%d but n_features=%d\n",
                    cfg.layer_sizes.front(), n_features);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (cfg.layer_sizes.back() != n_classes) {
        if (rank == 0)
            fprintf(stderr, "[Error] layer_sizes.back()=%d but n_classes=%d\n",
                    cfg.layer_sizes.back(), n_classes);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MLP model(cfg.layer_sizes, cfg.batch_size);

    // Pinned host buffer for gradient allreduce (allocated once)
    int    grad_n   = model.grad_total_size();
    float* h_grads  = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_grads, grad_n * sizeof(float)));

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    // Index array for shuffling batches each epoch
    std::vector<int> indices(local_n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42 + rank);  // each rank uses a different seed

    Timer timer;
    double t_epoch_start;

    for (int epoch = 0; epoch < cfg.epochs; epoch++) {
        timer.reset();
        t_epoch_start = MPI_Wtime();

        // Shuffle sample order for this epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        int   n_batches  = 0;

        for (int start = 0; start + cfg.batch_size <= local_n;
             start += cfg.batch_size) {

            // ------- Gather batch pointers -------
            // Because we shuffled indices (not the actual data), we need to
            // copy a shuffled mini-batch into a temporary device buffer.
            // For Phase 1 we keep this simple: copy on host, then to device.
            // (In Phase 2 we can optimise with a GPU gather kernel.)
            int b = cfg.batch_size;
            static std::vector<float> h_batch_img;
            static std::vector<int>   h_batch_lbl;
            h_batch_img.resize((size_t)b * n_features);
            h_batch_lbl.resize(b);

            for (int i = 0; i < b; i++) {
                int idx = indices[start + i];
                memcpy(h_batch_img.data() + (size_t)i * n_features,
                       train_local.images.data() + (size_t)idx * n_features,
                       n_features * sizeof(float));
                h_batch_lbl[i] = train_local.labels[idx];
            }

            // Allocate/reuse a per-batch device buffer (static: allocated once)
            static float* d_batch_img = nullptr;
            static int*   d_batch_lbl = nullptr;
            static int    alloc_batch = 0;
            if (alloc_batch < b) {
                cudaFree(d_batch_img);
                cudaFree(d_batch_lbl);
                CUDA_CHECK(cudaMalloc(&d_batch_img,
                                      (size_t)b * n_features * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_batch_lbl, b * sizeof(int)));
                alloc_batch = b;
            }

            CUDA_CHECK(cudaMemcpy(d_batch_img, h_batch_img.data(),
                                  (size_t)b * n_features * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_batch_lbl, h_batch_lbl.data(),
                                  b * sizeof(int), cudaMemcpyHostToDevice));

            // ------- Forward + Backward -------
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.start();

            model.zero_grad();
            float loss = model.forward(d_batch_img, d_batch_lbl, b);
            model.backward();

            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop_compute();

            // ------- AllReduce gradients (3-phase timing) -------
            timer.start();
            model.pack_gradients(h_grads);   // D→H
            timer.stop_d2h();

            timer.start();
            if (cfg.comm_algo == "ring") {
                ring_allreduce_sum_inplace(h_grads, grad_n, MPI_COMM_WORLD);
            } else if (cfg.comm_algo == "tree") {
                tree_allreduce_sum_inplace(h_grads, grad_n, MPI_COMM_WORLD);
            } else {
                MPI_Allreduce(MPI_IN_PLACE, h_grads, grad_n,
                              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            timer.stop_mpi();

            // Scale by 1/P then copy back to device (both counted as h2d overhead)
            float inv_P = 1.0f / world;
            for (int i = 0; i < grad_n; i++) h_grads[i] *= inv_P;
            timer.start();
            model.unpack_gradients(h_grads); // H→D
            timer.stop_h2d();

            // ------- SGD update -------
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.start();
            model.update(cfg.lr);
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop_compute();

            epoch_loss += loss;
            n_batches++;
        }

        // ------- Epoch summary -------
        double epoch_time = MPI_Wtime() - t_epoch_start;
        float  avg_loss   = n_batches > 0 ? epoch_loss / n_batches : 0.0f;

        // Reduce avg_loss across ranks to print a global value from rank 0
        float global_loss = avg_loss;
        MPI_Reduce(&avg_loss, &global_loss, 1, MPI_FLOAT, MPI_SUM,
                   0, MPI_COMM_WORLD);
        if (rank == 0) global_loss /= world;

        // Validation accuracy (rank 0 only, on full test set)
        float test_acc = 0.0f;
        if (rank == 0 && d_test_images) {
            test_acc = model.accuracy(d_test_images, d_test_labels, n_test);
        }

        if (rank == 0) {
            printf("[Epoch %2d/%d] loss=%.4f  test_acc=%.4f  time=%.3fs\n",
                   epoch + 1, cfg.epochs, global_loss, test_acc, epoch_time);
        }
        timer.report_epoch(rank, epoch + 1);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaFreeHost(h_grads);
    cudaFree(d_train_images);
    cudaFree(d_train_labels);
    if (rank == 0) {
        cudaFree(d_test_images);
        cudaFree(d_test_labels);
    }

    MPI_Finalize();
    return 0;
}
