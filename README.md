# Parallel Neural Network Training with MPI

CS 5220 Project

Data-parallel SGD for a configurable MLP, implemented from scratch in C++/CUDA with MPI. The project compares **Tree Reduction** and **Ring AllReduce** gradient aggregation strategies across small and large network sizes on Perlmutter (A100 GPUs).

---

## Repository Structure

```
cs5220-project/
├── CMakeLists.txt
├── src/
│   ├── config.h              # CLI argument parsing, Config struct
│   ├── timer.h               # Split timer: compute / d2h / mpi / h2d
│   ├── data/
│   │   ├── data_loader.h
│   │   └── data_loader.cpp   # Fashion-MNIST IDX reader + MPI scatter
│   ├── mlp/
│   │   ├── activations.cuh   # CUDA kernels: ReLU, softmax+CE, bias, SGD
│   │   ├── mlp.h
│   │   └── mlp.cu            # MLP forward, backward, update, pack/unpack grads
│   ├── comm/                 # Phase 2: custom collectives
│   │   ├── ring_allreduce.h
│   │   └── ring_allreduce.cpp
│   ├── tools/
│   │   └── ring_allreduce_test.cpp  # MPI-only correctness check vs MPI_Allreduce
│   └── main.cpp              # Training loop with 3-phase timed allreduce
├── scripts/
│   ├── download_data.sh      # Fetch Fashion-MNIST from GitHub mirror
│   ├── run_local.sh          # Local build + smoke test
│   ├── smoke_test.sbatch     # Single-GPU SLURM test job
│   └── submit_perlmutter.sbatch  # Full experiment job (2 nodes, 8 GPUs)
└── data/
    └── fashion-mnist/        # Downloaded by download_data.sh
```

---

## Environment (Perlmutter)

The following modules are already loaded in the standard Perlmutter GPU environment:

```bash
module load cudatoolkit/12.9
module load cray-mpich/9.0.1
```

**Important:** Always set `MPICH_GPU_SUPPORT_ENABLED=0` before running. Our gradient allreduce stages through pinned host memory (explicit D→H / H→D copies), so we do not use CUDA-aware MPI. Setting it to 1 causes a GTL linker error at runtime.

---

## Build

```bash
cd cs5220-project
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -Wno-dev
cmake --build build -j$(nproc)
# Produces: build/mlp_train
```

---

## Get the Data

```bash
bash scripts/download_data.sh
# Downloads and unpacks 4 IDX binary files into data/fashion-mnist/
```

---

## Running

### Interactive session (recommended for testing)

On Perlmutter **login nodes**, plain `srun --ntasks=4 ./app` is rejected (`Job request does not match any supported policy`) because no Slurm allocation exists yet. Either:

1. **Get an interactive allocation first**, then run `srun` inside that shell (inherits the job step):

```bash
# Replace --account if needed (course project example: m4341_g).
salloc --account=m4341_g --constraint="gpu&hbm40g" --nodes=1 --ntasks=4 --gpus-per-node=4 \
       --qos=interactive --time=0:30:00

cd /path/to/CS5220_Project/build
export MPICH_GPU_SUPPORT_ENABLED=0
srun ./ring_allreduce_test 10007
srun ./mlp_train --algo ring --epochs 1 --data ../data/fashion-mnist
```

2. **Or submit a batch script** (works from login; Slurm allocates the node for you):

```bash
cd /path/to/CS5220_Project
mkdir -p logs
# Uses m4341_g by default (edit scripts/quick_ring_test.sbatch if your account differs), then:
sbatch scripts/quick_ring_test.sbatch
```

After `salloc` returns a shell (or any interactive GPU allocation), examples:

```bash
cd build
export MPICH_GPU_SUPPORT_ENABLED=0

# Single rank
srun ./mlp_train \
  --layers 784,256,128,10 \
  --batch 256 --epochs 5 --lr 0.01 \
  --data ../data/fashion-mnist

# Multi-rank (4 GPUs, 1 node) — omit extra srun flags; the allocation already defines task count
srun ./mlp_train \
  --layers 784,256,128,10 \
  --batch 256 --epochs 5 --lr 0.01 \
  --data ../data/fashion-mnist
```

For four MPI ranks on one node, request `--ntasks=4 --gpus-per-node=4` in `salloc`.

### SLURM batch job (2 nodes, 8 GPUs)

```bash
sbatch scripts/submit_perlmutter.sbatch
```

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--layers A,B,...,Z` | `784,256,128,10` | Network architecture. First must be 784, last must be 10. |
| `--batch N` | `256` | Mini-batch size **per rank** |
| `--epochs N` | `10` | Number of training epochs |
| `--lr F` | `0.01` | SGD learning rate |
| `--algo NAME` | `mpi_builtin` | Allreduce backend: `mpi_builtin` \| `tree` \| `ring` |
| `--data PATH` | `./data/fashion-mnist` | Path to Fashion-MNIST IDX files |

### Example: reproduce small-network (latency-bound) vs large-network (bandwidth-bound)

```bash
# Small: expect tree to win in Phase 2
MPICH_GPU_SUPPORT_ENABLED=0 srun --ntasks=8 build/mlp_train \
  --layers 784,64,10 --batch 256 --epochs 5 --algo tree

# Large: expect ring to win in Phase 2
MPICH_GPU_SUPPORT_ENABLED=0 srun --ntasks=8 build/mlp_train \
  --layers 784,2048,2048,2048,10 --batch 256 --epochs 5 --algo ring
```

---

## Output Format

Each epoch prints two lines:

```
[Epoch  2/5] loss=0.7503  test_acc=0.7625  time=0.122s
[Epoch  2] compute=0.0477s  d2h=0.0176s  mpi=0.0000s  h2d=0.0186s  comm_total=0.0362s  total=0.0838s  mpi%=0.0%
```

| Field | Meaning |
|---|---|
| `compute` | GPU time: forward pass + backward pass + SGD update |
| `d2h` | Device→host memcpy of all gradients (PCIe, baseline cost) |
| `mpi` | **Pure MPI algorithm time** (Allreduce / Tree / Ring) — the key metric for Phase 3 comparison |
| `h2d` | CPU gradient scaling (÷P) + host→device memcpy |
| `mpi%` | `mpi / total` — isolates network communication from PCIe overhead |

**Why separate d2h/mpi/h2d?** With 1 rank, `d2h ≈ h2d ≈ 4.5ms` per epoch and `mpi ≈ 0ms`. This PCIe cost is unavoidable and shared by all algorithms. Phase 3 α-β analysis uses only the `mpi` column to compare Tree vs Ring.

---

## Architecture: MLP

**File:** `src/mlp/mlp.h` + `src/mlp/mlp.cu`

All matrices are stored **column-major** (cuBLAS convention). For a matrix of shape `(rows × cols)`, element `(r, c)` lives at index `r + c * rows`.

```
Input X: (n_features × batch)   ← matches how Fashion-MNIST is laid out in memory
Layer l: Z = W*X + b             ← cublasSgemm
         A = ReLU(Z)             ← hidden layers
         A = Softmax(Z)          ← output layer (also computes CE loss and dZ)
```

### Key methods

```cpp
MLP model(layer_sizes, max_batch);

// Forward pass. X and labels are device pointers.
// Returns average cross-entropy loss over the batch.
float loss = model.forward(d_X, d_labels, batch_size);

// Backward pass. Populates dW and db for each layer.
// Call zero_grad() first to clear accumulators.
model.zero_grad();
model.backward();

// Gradient communication interface (Phase 2 hook):
model.pack_gradients(h_buf);    // device → pinned host, size = grad_total_size()
// ... run your allreduce on h_buf here ...
model.unpack_gradients(h_buf);  // pinned host → device

// How many floats in the gradient buffer (dW+db for all layers):
int n = model.grad_total_size();

// SGD weight update (after allreduce):
model.update(lr);

// Evaluation (runs forward on device array, returns fraction correct):
float acc = model.accuracy(d_X, d_labels, n_samples);
```

### Layer access (for Phase 2 layer-wise pipelining)

```cpp
for (int l = 0; l < model.n_layers(); l++) {
    Layer& la = model.layer(l);
    // la.W, la.b    — weight/bias device pointers
    // la.dW, la.db  — gradient device pointers
    // la.fin, la.fout — dimensions
    // gradient buffer size: la.fout * la.fin + la.fout floats
}
```

---

## Architecture: Data Loading

**File:** `src/data/data_loader.h` + `data_loader.cpp`

```cpp
// Rank 0 only: load full dataset from IDX binary files
Dataset train = load_fashion_mnist("data/fashion-mnist", /*is_train=*/true);
Dataset test  = load_fashion_mnist("data/fashion-mnist", /*is_train=*/false);

// All ranks: rank 0 scatters shards via MPI_Scatter
// Each rank receives local_n = n_total / world_size samples
Dataset local = scatter_dataset(train, rank, world_size);

// local.images: float[n_samples * 784], layout (n_features × n_samples) column-major
// local.labels: int[n_samples]
```

Memory layout note: `images[s * 784 + f]` for sample `s`, feature `f`. A contiguous slice of `batch_size` samples starting at index `s` is directly usable as the input matrix `(784 × batch_size)` in column-major order — no transposition needed before passing to `model.forward()`.

---

## Architecture: Timer

**File:** `src/timer.h`

```cpp
Timer timer;
timer.reset();                          // zero all accumulators

cudaDeviceSynchronize();
timer.start();
// ... GPU compute work ...
cudaDeviceSynchronize();
timer.stop_compute();

timer.start();
model.pack_gradients(h_buf);            // includes internal cudaDeviceSynchronize
timer.stop_d2h();

timer.start();
MPI_Allreduce(...);                     // your algorithm here
timer.stop_mpi();

timer.start();
model.unpack_gradients(h_buf);
timer.stop_h2d();

timer.report_epoch(rank, epoch);        // prints only on rank 0
```

---

## Phase 2: Communication primitives

**Task A — Tree reduction (`MPI_Reduce` semantics):** binomial tree sum to one root in `src/comm/tree_reduce.{h,cpp}`. This is **not** an allreduce: only the root holds the global sum. Use it for latency experiments on **small** gradient vectors (shallow/narrow MLP). Build and test:

```bash
cmake --build build -j
MPICH_GPU_SUPPORT_ENABLED=0 srun --ntasks=4 ./build/tree_reduce_test 64 2000
# argv: [n] [bench_iters]  — bench_iters 0 skips timing; >0 prints tree vs MPI_Reduce latency on rank 0
sbatch scripts/quick_tree_test.sbatch
# GPU-node allocation (matches many course policies); binary is still CPU-only.
# Pure-CPU Slurm (if your account allows): `sbatch scripts/quick_tree_test_cpu.sbatch`
# No batch script: inside `salloc`, `source scripts/env_cpu_mpi.sh` (sets `MPICH_GPU_SUPPORT_ENABLED=0` on GPU nodes), then `bash scripts/run_mpi_comm_local.sh` or `srun .../tree_reduce_test` (Cray has no mpirun; script uses `srun` when SLURM_JOB_ID is set)
```

**Task B — Ring allreduce:** `ring_allreduce_sum_inplace` in `src/comm/ring_allreduce.cpp`; training uses `--algo ring`. Correctness:

```bash
MPICH_GPU_SUPPORT_ENABLED=0 srun --ntasks=4 ./build/ring_allreduce_test 10007
```

`mlp_train` supports `--algo mpi_builtin|ring` (full gradient sync requires an allreduce, not a reduce-only primitive).

### Tree reduce cost model (latency-oriented small `n`)

Roughly `ceil(log2 P)` steps per child/parent message of size `n`: `T ≈ ceil(log2 P) * (α + β·n)` for the custom tree; compare to `MPI_Reduce` in `tree_reduce_test` when `bench_iters > 0`.

### Verified baselines

**1 rank, `{784,256,128,10}`, batch=256, epoch 2–5:**
```
compute=0.0477s  d2h=0.0176s  mpi=0.0000s  h2d=0.0186s  total=0.0838s  mpi%=0.0%
test_acc after 5 epochs: 80.8%
```

**4 ranks (1 node), `{784,256,128,10}`, batch=256, epoch 2–5:**
```
compute=0.0121s  d2h=0.0044s  mpi=0.0260s  h2d=0.0048s  total=0.0472s  mpi%=55.1%
test_acc after 5 epochs: 73.0%
```
The lower accuracy at P=4 is expected: each rank sees only 15000/60000 samples and makes 58 gradient updates per epoch vs 234 at P=1. Accuracy converges to the same level with more epochs.

With P>1, `mpi` will increase further. For **small** packed gradient size `N`, a tree-shaped reduce has fewer steps than ring allreduce’s `2(P-1)` steps; for **large** `N`, ring’s ring bandwidth pattern usually wins for full allreduce.

---

## Known Issues / Notes

- **P must divide 60000 evenly** for the data scatter. P=1,2,4,8 all work (60000/8=7500). If using a non-divisor, the loader truncates to the nearest multiple.
- **Tree reduce** in this repo uses a binomial tree and works for **any** `P >= 1` (not only powers of two).
- **Accuracy baseline:** `{784,256,128,10}`, 5 epochs, lr=0.01 → ~80.8% test accuracy. Expect ~85%+ with 10+ epochs or a larger network.
- The `accuracy()` method runs a forward pass internally and reuses the model's layer buffers — do not call it concurrently with training.
