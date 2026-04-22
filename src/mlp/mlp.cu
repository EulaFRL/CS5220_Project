#include "mlp.h"
#include "activations.cuh"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d  status=%d\n",                \
                    __FILE__, __LINE__, (int)_s);                              \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static float* dev_alloc_zeros(size_t n) {
    float* p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
    return p;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
MLP::MLP(const std::vector<int>& sizes, int max_batch)
    : sizes_(sizes), max_batch_(max_batch) {

    CUBLAS_CHECK(cublasCreate(&cublas_));

    int n_layers = (int)sizes.size() - 1;
    layers_.resize(n_layers);

    grad_buf_n_ = 0;
    for (int l = 0; l < n_layers; l++) {
        Layer& la = layers_[l];
        la.fin  = sizes[l];
        la.fout = sizes[l + 1];

        la.W  = dev_alloc_zeros((size_t)la.fout * la.fin);
        la.b  = dev_alloc_zeros(la.fout);
        la.dW = dev_alloc_zeros((size_t)la.fout * la.fin);
        la.db = dev_alloc_zeros(la.fout);

        la.Z  = dev_alloc_zeros((size_t)la.fout * max_batch);
        la.A  = dev_alloc_zeros((size_t)la.fout * max_batch);
        la.dZ = dev_alloc_zeros((size_t)la.fout * max_batch);
        la.dA = dev_alloc_zeros((size_t)la.fout * max_batch);

        grad_buf_n_ += la.fout * la.fin + la.fout;  // W + b
    }

    CUDA_CHECK(cudaMalloc(&d_loss_arr_, max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels_,   max_batch * sizeof(int)));

    // Pinned host buffer (used by pack/unpack; Phase 2 may use d_grad_buf_ for CUDA-aware MPI)
    CUDA_CHECK(cudaMallocHost(&h_grad_buf_, grad_buf_n_ * sizeof(float)));

    init_weights();
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
MLP::~MLP() {
    for (auto& la : layers_) {
        cudaFree(la.W);  cudaFree(la.b);
        cudaFree(la.dW); cudaFree(la.db);
        cudaFree(la.Z);  cudaFree(la.A);
        cudaFree(la.dZ); cudaFree(la.dA);
    }
    cudaFree(d_loss_arr_);
    cudaFree(d_labels_);
    cudaFreeHost(h_grad_buf_);
    cublasDestroy(cublas_);
}

// ---------------------------------------------------------------------------
// He initialisation: W ~ N(0, sqrt(2 / fan_in))
// ---------------------------------------------------------------------------
void MLP::init_weights() {
    std::mt19937 rng(42);
    for (auto& la : layers_) {
        float std = sqrtf(2.0f / la.fin);
        std::normal_distribution<float> dist(0.0f, std);
        int n = la.fout * la.fin;
        std::vector<float> h_w(n);
        for (auto& v : h_w) v = dist(rng);
        CUDA_CHECK(cudaMemcpy(la.W, h_w.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));
        // biases stay zero
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// X: device pointer (fin × batch) column-major
// labels: device int pointer (batch,)
// Returns: average cross-entropy loss
// ---------------------------------------------------------------------------
float MLP::forward(const float* X, const int* labels, int batch) {
    cur_X_     = X;
    cur_batch_ = batch;

    // Copy labels to internal buffer (needed by softmax kernel)
    CUDA_CHECK(cudaMemcpy(d_labels_, labels, batch * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    const float one  = 1.0f;
    const float zero = 0.0f;

    const float* A_prev = X;
    int          fan_in_prev = sizes_[0];

    int n_layers = (int)layers_.size();
    for (int l = 0; l < n_layers; l++) {
        Layer& la = layers_[l];

        // Z = W * A_prev    (fout × batch) = (fout × fin) * (fin × batch)
        // cuBLAS: C(m×n) = A(m×k) * B(k×n)  all column-major
        CUBLAS_CHECK(cublasSgemm(
            cublas_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            la.fout, batch, la.fin,          // m, n, k
            &one,
            la.W,   la.fout,                 // A: fout×fin,  lda=fout
            A_prev, la.fin,                  // B: fin×batch, ldb=fin
            &zero,
            la.Z,   la.fout));               // C: fout×batch, ldc=fout

        // Z += b  (broadcast bias along batch dimension)
        launch_bias_add(la.Z, la.b, la.fout, batch);

        bool is_output = (l == n_layers - 1);
        if (is_output) {
            // Softmax + CE: fills la.A (probabilities), d_loss_arr_, la.dZ
            CUDA_CHECK(cudaMemset(d_loss_arr_, 0, batch * sizeof(float)));
            launch_softmax_ce(la.Z, d_labels_, la.A, d_loss_arr_, la.dZ,
                              la.fout, batch);
        } else {
            // ReLU activation
            launch_relu_fwd(la.Z, la.A, la.fout * batch);
        }

        A_prev       = la.A;
        fan_in_prev  = la.fout;
        (void)fan_in_prev;
    }

    // Compute mean loss on host (batch is small; avoid extra kernel)
    std::vector<float> h_loss(batch);
    CUDA_CHECK(cudaMemcpy(h_loss.data(), d_loss_arr_, batch * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float mean_loss = 0.0f;
    for (float v : h_loss) mean_loss += v;
    return mean_loss / batch;
}

// ---------------------------------------------------------------------------
// Backward pass — computes dW and db for every layer.
// dZ for the output layer was already set in forward().
// ---------------------------------------------------------------------------
void MLP::backward() {
    const float one  = 1.0f;
    const float zero = 0.0f;

    int n_layers = (int)layers_.size();
    int batch    = cur_batch_;

    for (int l = n_layers - 1; l >= 0; l--) {
        Layer& la = layers_[l];

        // For hidden layers, compute dZ from dA via ReLU backward.
        // For the output layer, dZ was already set in forward().
        if (l < n_layers - 1) {
            launch_relu_bwd(la.dA, la.Z, la.dZ, la.fout * batch);
        }

        // dW += dZ * A_prev^T    (fout × fin) = (fout × batch) * (batch × fin)
        const float* A_prev = (l == 0) ? cur_X_ : layers_[l - 1].A;
        int          fin    = la.fin;

        CUBLAS_CHECK(cublasSgemm(
            cublas_,
            CUBLAS_OP_N, CUBLAS_OP_T,
            la.fout, fin, batch,             // m, n, k
            &one,
            la.dZ,  la.fout,                 // A: fout×batch
            A_prev, fin,                     // B (transposed): fin×batch → batch×fin
            &one,                            // beta=1: accumulate into dW
            la.dW,  la.fout));

        // db += rowsum(dZ)
        launch_bias_grad(la.dZ, la.db, la.fout, batch);

        // Propagate gradient to previous layer: dA_{l-1} = W^T * dZ
        if (l > 0) {
            Layer& prev = layers_[l - 1];
            CUBLAS_CHECK(cublasSgemm(
                cublas_,
                CUBLAS_OP_T, CUBLAS_OP_N,
                fin, batch, la.fout,         // m, n, k
                &one,
                la.W,   la.fout,             // A (transposed): fout×fin → fin×fout
                la.dZ,  la.fout,             // B: fout×batch
                &zero,
                prev.dA, fin));              // C: fin×batch  (overwrites)
        }
    }
}

// ---------------------------------------------------------------------------
// Zero gradients
// ---------------------------------------------------------------------------
void MLP::zero_grad() {
    for (auto& la : layers_) {
        CUDA_CHECK(cudaMemset(la.dW, 0, (size_t)la.fout * la.fin * sizeof(float)));
        CUDA_CHECK(cudaMemset(la.db, 0,  la.fout                 * sizeof(float)));
    }
}

// ---------------------------------------------------------------------------
// Pack all gradients (dW, db per layer) into a flat host buffer.
// Called before MPI_Allreduce in main().
// ---------------------------------------------------------------------------
void MLP::pack_gradients(float* h_buf) const {
    CUDA_CHECK(cudaDeviceSynchronize());
    int offset = 0;
    for (const auto& la : layers_) {
        int nW = la.fout * la.fin;
        int nb = la.fout;
        CUDA_CHECK(cudaMemcpy(h_buf + offset, la.dW,
                              nW * sizeof(float), cudaMemcpyDeviceToHost));
        offset += nW;
        CUDA_CHECK(cudaMemcpy(h_buf + offset, la.db,
                              nb * sizeof(float), cudaMemcpyDeviceToHost));
        offset += nb;
    }
}

// ---------------------------------------------------------------------------
// Unpack gradients from flat host buffer back to device after MPI_Allreduce.
// ---------------------------------------------------------------------------
void MLP::unpack_gradients(const float* h_buf) {
    int offset = 0;
    for (auto& la : layers_) {
        int nW = la.fout * la.fin;
        int nb = la.fout;
        CUDA_CHECK(cudaMemcpy(la.dW, h_buf + offset,
                              nW * sizeof(float), cudaMemcpyHostToDevice));
        offset += nW;
        CUDA_CHECK(cudaMemcpy(la.db, h_buf + offset,
                              nb * sizeof(float), cudaMemcpyHostToDevice));
        offset += nb;
    }
}

// ---------------------------------------------------------------------------
// SGD update: param -= lr * grad  (after allreduce)
// ---------------------------------------------------------------------------
void MLP::update(float lr) {
    for (auto& la : layers_) {
        int nW = la.fout * la.fin;
        launch_sgd_update(la.W, la.dW, lr, nW);
        launch_sgd_update(la.b, la.db, lr, la.fout);
    }
}

// ---------------------------------------------------------------------------
// Accuracy: run forward pass in chunks (no gradient needed), compare argmax.
// ---------------------------------------------------------------------------
float MLP::accuracy(const float* X, const int* labels, int n) {
    // We reuse the existing layer buffers; max chunk = max_batch_
    int correct_total = 0;
    int chunk = max_batch_;

    int* d_correct;
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));

    for (int start = 0; start < n; start += chunk) {
        int b = (start + chunk <= n) ? chunk : (n - start);

        // Forward pass (no labels needed for logits; pass dummy nullptr-safe labels)
        // We pass labels pointer shifted — they're not used beyond computing dZ which
        // we ignore here. Use a dedicated inference forward that skips loss computation.
        // For simplicity, reuse forward() and just ignore the returned loss.
        (void)forward(X + (size_t)start * sizes_[0],
                      labels + start, b);

        CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        // Output logits are in layers_.back().Z  (or .A for softmax probs — use .Z)
        launch_count_correct(layers_.back().A, d_labels_, d_correct,
                             sizes_.back(), b);

        int h_correct;
        CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int),
                              cudaMemcpyDeviceToHost));
        correct_total += h_correct;
    }
    cudaFree(d_correct);
    return (float)correct_total / n;
}
