#pragma once
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Per-layer storage (all device memory, column-major).
// W shape: (fout × fin),  b shape: (fout,)
// Cache Z, A: (fout × max_batch)
// dZ, dA:    (fout × max_batch)
// ---------------------------------------------------------------------------
struct Layer {
    int fin = 0, fout = 0;

    float* W  = nullptr;   // weights
    float* b  = nullptr;   // biases
    float* dW = nullptr;   // weight gradient (accumulated over batch)
    float* db = nullptr;   // bias gradient

    float* Z  = nullptr;   // pre-activation output
    float* A  = nullptr;   // post-activation output (also input to next layer)
    float* dZ = nullptr;   // gradient w.r.t. Z
    float* dA = nullptr;   // gradient w.r.t. A (written by next layer's backward)
};

// ---------------------------------------------------------------------------
// MLP with configurable depth/width.
// Activations: ReLU for all hidden layers, Softmax+CE at the output layer.
// Optimizer: SGD (momentum can be added in Phase 3 if desired).
// ---------------------------------------------------------------------------
class MLP {
public:
    // sizes: {n_input, hidden1, hidden2, ..., n_classes}
    MLP(const std::vector<int>& sizes, int max_batch);
    ~MLP();

    // Forward pass.  X is a device pointer of shape (fin × batch), column-major.
    // labels is a device int pointer of shape (batch,).
    // Returns average cross-entropy loss over the batch (host scalar).
    float forward(const float* X, const int* labels, int batch);

    // Backward pass — must be called after forward().
    // Computes dW[l] and db[l] for every layer.
    // Gradients are NOT normalised by world_size here; caller does that via
    // allreduce_gradients() before update().
    void backward();

    // Zero dW and db for all layers (call before each backward pass).
    void zero_grad();

    // Copy all gradients into a caller-supplied host buffer (flat, dW then db per layer).
    // Buffer must have size >= grad_total_size().
    void pack_gradients(float* h_buf) const;

    // Copy back from a caller-supplied host buffer (after allreduce in main).
    void unpack_gradients(const float* h_buf);

    // Total number of gradient floats across all layers.
    int grad_total_size() const { return grad_buf_n_; }

    // SGD update: param -= lr * grad  (after allreduce_gradients).
    void update(float lr);

    // Evaluate accuracy on a device array (X: fin×n, labels: n).
    // Returns fraction of correct predictions.
    float accuracy(const float* X, const int* labels, int n);

    int n_layers() const { return (int)layers_.size(); }
    Layer& layer(int l) { return layers_[l]; }

private:
    void init_weights();   // He initialisation for ReLU

    std::vector<Layer>   layers_;
    std::vector<int>     sizes_;
    int                  max_batch_;
    int                  cur_batch_ = 0;

    const float*         cur_X_ = nullptr;  // input pointer for current forward pass

    float*               d_loss_arr_ = nullptr;   // (max_batch,) per-sample loss
    int*                 d_labels_   = nullptr;   // (max_batch,) current labels copy

    cublasHandle_t       cublas_;

    // Pinned host buffer for pack_gradients/unpack_gradients
    float*               h_grad_buf_ = nullptr;
    int                  grad_buf_n_ = 0;
};
