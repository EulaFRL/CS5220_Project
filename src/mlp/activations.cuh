#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// All matrices use column-major layout (cuBLAS convention).
// For a matrix of shape (rows × cols):  element (r, c) lives at index r + c*rows
// ---------------------------------------------------------------------------

// ReLU forward: A = max(0, Z),  element-wise over all n elements
__global__ void relu_fwd(const float* __restrict__ Z,
                         float* __restrict__ A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = Z[i] > 0.0f ? Z[i] : 0.0f;
}

// ReLU backward: dZ = dA * (Z > 0),  element-wise
__global__ void relu_bwd(const float* __restrict__ dA,
                         const float* __restrict__ Z,
                         float* __restrict__ dZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dZ[i] = Z[i] > 0.0f ? dA[i] : 0.0f;
}

// Bias add: Z[:, b] += bias  for each batch column b
// Z shape: (fan_out × batch), column-major  →  element (r, c) at r + c*fan_out
__global__ void bias_add(float* __restrict__ Z, const float* __restrict__ bias,
                         int fan_out, int batch) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // row (neuron)
    int c = blockIdx.x * blockDim.x + threadIdx.x;  // col (sample)
    if (r < fan_out && c < batch)
        Z[r + c * fan_out] += bias[r];
}

// Bias gradient: db[r] = sum_c dZ[r, c]
// dZ shape: (fan_out × batch), column-major
__global__ void bias_grad(const float* __restrict__ dZ,
                          float* __restrict__ db,
                          int fan_out, int batch) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < fan_out) {
        float s = 0.0f;
        for (int c = 0; c < batch; c++)
            s += dZ[r + c * fan_out];
        // Accumulate (caller zeroes db before backward pass)
        db[r] += s;
    }
}

// Softmax + cross-entropy (output layer only).
// One thread per sample in the batch.
//
// Inputs:
//   Z      : (n_classes × batch) logits, column-major
//   labels : (batch) integer class indices
// Outputs:
//   A      : (n_classes × batch) softmax probabilities
//   loss_per_sample : (batch) per-sample cross-entropy loss
//   dZ     : (n_classes × batch) gradient dL/dZ = (A - one_hot(y)) / batch
//            (normalised here so the caller can directly MPI-allreduce and update)
__global__ void softmax_ce_fwd(const float* __restrict__ Z,
                               const int*   __restrict__ labels,
                               float* __restrict__ A,
                               float* __restrict__ loss_per_sample,
                               float* __restrict__ dZ,
                               int n_classes, int batch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= batch) return;

    const float* z = Z + col * n_classes;   // pointer to this sample's logits
    float*       a = A + col * n_classes;
    float*       dz = dZ + col * n_classes;

    // Numerically stable softmax: subtract max
    float mx = -FLT_MAX;
    for (int c = 0; c < n_classes; c++) mx = fmaxf(mx, z[c]);

    float sum = 0.0f;
    for (int c = 0; c < n_classes; c++) {
        a[c] = expf(z[c] - mx);
        sum += a[c];
    }
    float inv = 1.0f / sum;
    for (int c = 0; c < n_classes; c++) a[c] *= inv;

    int lbl = labels[col];
    loss_per_sample[col] = -logf(fmaxf(a[lbl], 1e-10f));

    // dL/dZ_c = (a_c - 1{c==y}) / batch
    float inv_batch = 1.0f / (float)batch;
    for (int c = 0; c < n_classes; c++)
        dz[c] = (a[c] - (c == lbl ? 1.0f : 0.0f)) * inv_batch;
}

// SGD weight update: param -= lr * grad
__global__ void sgd_update(float* __restrict__ param,
                           const float* __restrict__ grad,
                           float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) param[i] -= lr * grad[i];
}

// Accuracy: count how many argmax(logits) == label
// logits: (n_classes × batch), column-major
// correct: device int[1], initialised to 0 by caller
__global__ void count_correct(const float* __restrict__ logits,
                              const int*   __restrict__ labels,
                              int* __restrict__ correct,
                              int n_classes, int batch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= batch) return;
    const float* lg = logits + col * n_classes;
    int pred = 0;
    for (int c = 1; c < n_classes; c++)
        if (lg[c] > lg[pred]) pred = c;
    if (pred == labels[col]) atomicAdd(correct, 1);
}

// ---------------------------------------------------------------------------
// Launcher helpers (keep kernel launch boilerplate out of mlp.cu)
// ---------------------------------------------------------------------------
inline void launch_relu_fwd(const float* Z, float* A, int n) {
    int threads = 256;
    relu_fwd<<<(n + threads - 1) / threads, threads>>>(Z, A, n);
}

inline void launch_relu_bwd(const float* dA, const float* Z, float* dZ, int n) {
    int threads = 256;
    relu_bwd<<<(n + threads - 1) / threads, threads>>>(dA, Z, dZ, n);
}

inline void launch_bias_add(float* Z, const float* b, int fan_out, int batch) {
    dim3 threads(16, 16);
    dim3 blocks((batch + 15) / 16, (fan_out + 15) / 16);
    bias_add<<<blocks, threads>>>(Z, b, fan_out, batch);
}

inline void launch_bias_grad(const float* dZ, float* db, int fan_out, int batch) {
    int threads = 256;
    bias_grad<<<(fan_out + threads - 1) / threads, threads>>>(dZ, db, fan_out, batch);
}

inline void launch_softmax_ce(const float* Z, const int* labels,
                               float* A, float* loss_arr, float* dZ,
                               int n_classes, int batch) {
    int threads = 256;
    softmax_ce_fwd<<<(batch + threads - 1) / threads, threads>>>(
        Z, labels, A, loss_arr, dZ, n_classes, batch);
}

inline void launch_sgd_update(float* p, const float* g, float lr, int n) {
    int threads = 256;
    sgd_update<<<(n + threads - 1) / threads, threads>>>(p, g, lr, n);
}

inline void launch_count_correct(const float* logits, const int* labels,
                                  int* correct, int n_classes, int batch) {
    int threads = 256;
    count_correct<<<(batch + threads - 1) / threads, threads>>>(
        logits, labels, correct, n_classes, batch);
}
