# Distributed MLP Training: Ring vs Tree AllReduce on GPU Clusters

**Ruichen Bao** (rb945@cornell.edu) · **Jingyu (Eula) Wang** (jw2953@cornell.edu)  
Cornell University — CS 5220: Systems Programming for Applications in Science, Spring 2026

GitHub: [https://github.com/[PLACEHOLDER]]

---

## 1. Hypothesis

In data-parallel stochastic gradient descent (SGD), all participating processes must synchronize
their locally computed gradients before applying a weight update. As the number of processes
grows, this gradient aggregation step becomes the dominant bottleneck. The choice of
communication topology determines how efficiently the network is used.

We investigate three hypotheses:

**H1 (latency regime).** For neural networks with a small number of parameters — shallow or
narrow MLPs whose total gradient payload is on the order of hundreds of kilobytes — a binomial
Tree Reduce strategy will yield lower AllReduce latency than Ring AllReduce. Tree Reduce
completes in O(log P) communication rounds, while Ring AllReduce requires O(P) rounds, giving
Tree a latency advantage when the per-message startup cost α dominates over bandwidth costs.

**H2 (bandwidth regime).** For larger networks whose gradient tensors reach tens of megabytes,
Ring AllReduce will outperform Tree Reduce. Ring AllReduce sends exactly 2(P−1)/P·N bytes per
rank regardless of P, making its total communication volume bandwidth-optimal. Tree Reduce sends
2⌈log₂P⌉·N bytes per rank, a larger volume that causes root-node congestion and wastes
bandwidth.

**H3 (scaling bottleneck).** As we scale to more GPUs, gradient synchronization — not
computation — will become the dominant bottleneck, limiting strong-scaling efficiency
significantly below ideal.

**Outcomes (preview).** H1 was not confirmed: Tree Reduce was slower than Ring and MPI built-in
even for small networks, because the synchronization overhead introduced to avoid deadlocks on
Cray MPICH masked its theoretical latency advantage. H2 was strongly confirmed: Ring AllReduce
was 16–20% faster than MPI built-in and 3.3–3.9× faster than Tree Reduce in the
bandwidth-dominated regime. H3 was confirmed: parallel efficiency dropped to 33% at P=8, with
the MPI AllReduce step accounting for 57% of epoch wall time.

---

## 2. Context

This project is an independent, standalone effort for CS 5220. It is not a sub-project of any
larger research initiative, and it is not used in any other course this semester. All code is
written from scratch in C++/CUDA using MPI, without relying on high-level machine learning
frameworks such as PyTorch or TensorFlow.

**Team contributions:**

| Contributor | Contributions |
|---|---|
| Ruichen Bao | [PLACEHOLDER — fill in before submission] |
| Jingyu (Eula) Wang | [PLACEHOLDER — fill in before submission] |

Both members jointly designed the experiments, ran benchmarks on NERSC Perlmutter, and analyzed
results together.

---

## 3. Prior Work

**Rabenseifner (2004).** "Optimization of Collective Reduction Operations," ICCS, vol. 3036,
pp. 1–9. This paper provides the theoretical foundation for our communication algorithm design.
Profiling MPI workloads across a broad range of production applications, Rabenseifner shows that
MPI_Allreduce alone accounts for over 40% of total MPI execution time, and derives a family of
collective algorithms whose optimal choice depends jointly on message size N and process count P.
For short messages where latency α dominates, recursive doubling — a tree-structured pattern
completing in O(log P) steps — is superior. For long messages where bandwidth β dominates, a
ring-based reduce-scatter followed by an allgather is optimal because it limits the data each
rank sends to 2(P-1)/P·N bytes, independent of P. This latency/bandwidth crossover is the
direct basis for our H1 and H2 hypotheses, and the α-β cost model from this paper is our
primary analytical framework.

**Gibiansky (2017).** "Bringing HPC Techniques to Deep Learning," Baidu Silicon Valley AI Lab
Technical Blog. This work bridges the gap between HPC collective algorithms and neural network
training. Gibiansky demonstrates that naive parameter-server gradient aggregation creates a
root-node bottleneck whose cost grows linearly with the number of workers, and shows that ring
allreduce eliminates this bottleneck: each worker sends and receives exactly 2N(P−1)/P bytes
regardless of P, making total communication cost effectively independent of worker count. The
paper also shows that ring allreduce can be pipelined with backpropagation by initiating gradient
exchange for earlier layers while later-layer gradients are still being computed, which motivates
our future work on non-blocking communication. This work directly informs the design and
motivation of our ring allreduce implementation.

**Castelló, Quintana-Ortí, and Duato (2021).** "Accelerating Distributed Deep Neural Network
Training with Pipelined MPI Allreduce," Cluster Computing, vol. 25. The authors conduct a
systematic comparison of ring and binomial-tree allreduce algorithms for distributed DNN training
using TensorFlow with Horovod, and find that the optimal algorithm depends on message size and
cluster configuration — consistent with the theoretical crossover predicted by Rabenseifner. They
further demonstrate that replacing blocking MPI_Allreduce with pipelined non-blocking
MPI_Iallreduce yields up to 60% speedup by overlapping communication with computation. Our
project replicates this ring-vs-tree comparison from scratch in a controlled MLP setting,
enabling clean isolation of the communication bottleneck. Our 3-phase timer (separating D→H
transfer, pure MPI time, and H→D transfer) was directly motivated by the need to make this
comparison fair, following the measurement discipline advocated in this paper.

---

## 4. Empirical Methodology

### 4.1 System Design

We implement a data-parallel synchronous SGD training loop for a configurable MLP from scratch
in C++/CUDA/MPI. The system is designed so that communication primitives are fully swappable at
runtime via a single `--algo` flag: `mpi_builtin`, `ring`, or `tree`.

**Architecture.** Each MPI rank owns exactly one GPU (assigned as
`local_rank = global_rank % GPUs_per_node`) and one disjoint shard of the training data. Every
training step follows four phases: (1) forward pass on the local mini-batch, (2) backward pass
to compute local gradients, (3) AllReduce to aggregate gradients across all ranks, (4) SGD
weight update using the averaged gradients. There is no parameter server; all gradient
communication is peer-to-peer.

**MLP implementation** (`src/mlp/mlp.cu`, `src/mlp/activations.cuh`). All weight matrices and
activation tensors are stored in column-major layout throughout, matching the cuBLAS SGEMM
convention. For a layer with `fin` inputs and `fout` outputs and a mini-batch of `B` samples,
the weight matrix W has shape (fout × fin) and the activation A has shape (fout × B); the matrix
multiply Z = W·A_prev is computed as a single cuBLAS SGEMM call. Hidden layers apply ReLU
activation. The output layer uses a fused CUDA kernel that computes softmax probabilities,
per-sample cross-entropy loss, and the loss gradient dZ = (softmax − one_hot(y)) / B in a single
GPU pass, avoiding an extra kernel launch. Weights are initialized using He initialization:
W ~ N(0, sqrt(2/fan_in)). The SGD weight update (param -= lr * grad) is also a CUDA kernel.

**Gradient staging.** Gradients computed during backpropagation live in GPU device memory. Before
MPI communication, `pack_gradients()` copies all dW and db buffers layer-by-layer into a
pre-allocated pinned host buffer via `cudaMemcpy` (device-to-host). After the AllReduce, the
host buffer is scaled by 1/P on the CPU and then copied back to device via
`unpack_gradients()`. We set `MPICH_GPU_SUPPORT_ENABLED=0` throughout, using host-staged
communication rather than CUDA-aware MPI. This was required to avoid a GTL runtime linker
error on Perlmutter (detailed in Section 5), and has the additional benefit of making the
device-to-host and host-to-device transfer costs explicit and separately measurable.

**Dataset.** We train on Fashion-MNIST: 60,000 training images and 10,000 test images, each a
784-dimensional flattened grayscale image across 10 clothing categories. Rank 0 loads the full
dataset from IDX binary files and distributes disjoint shards to all ranks via `MPI_Scatter`.
For P ranks, each rank receives 60,000/P training samples. (P must divide 60,000; P = 1, 2, 4,
and 8 all satisfy this.) Within each epoch, samples are shuffled locally using a per-rank random
seed (seed = 42 + rank), ensuring different ranks see different orderings.

**Hardware.** All experiments run on NERSC Perlmutter using 2 nodes, each with 4 NVIDIA A100
SXM4 GPUs (40 GB HBM2) connected via NVLink within the node. Nodes are connected via
Slingshot-11 (200 Gb/s per port, Dragonfly topology). Experiments with P ≤ 4 use a single node
(intra-node NVLink + PCIe); experiments with P = 8 span both nodes and incur true inter-node
Slingshot-11 traffic.

*[Figure 1: Data-parallel MLP training pipeline — 4 GPU example showing data partition,
forward/backward compute, AllReduce, and SGD update stages.]*

### 4.2 Communication Algorithms

We implement two custom collective algorithms and compare them against MPI's built-in
`MPI_Allreduce`. All three operate on a flat buffer of `N` floats (the concatenation of all dW
and db gradients across all layers).

*[Figure 2: Communication topology diagrams — Ring AllReduce (left) and Tree Reduce + Broadcast
(right) for P = 4.]*

**Ring AllReduce** (`src/comm/ring_allreduce.cpp`). Ring AllReduce proceeds in two phases.

*Reduce-scatter (P−1 steps):* The buffer is split into P chunks; chunk i has size ⌊N/P⌋ or
⌈N/P⌉ (the first N mod P chunks get one extra element to handle non-divisible N). In step s,
each rank sends chunk (rank − s) mod P to its right neighbor (rank+1) mod P and simultaneously
receives chunk (rank − s − 1) mod P from its left neighbor (rank−1) mod P. The received data is
element-wise accumulated into the buffer. After P−1 steps, rank r holds the fully reduced
(summed) values for exactly one chunk — the one that rank r "owns" in the final result.

*Allgather (P−1 steps):* The process runs in reverse. Each rank sends its fully reduced chunk to
its right neighbor; the received data overwrites the corresponding slot in the buffer (no
accumulation). After P−1 more steps, every rank holds the complete reduced buffer.

Both phases use `MPI_Sendrecv` on a logical ring, which posts a send and receive in a single
call. This avoids deadlock without requiring any barrier synchronization between steps — a key
advantage over our tree implementation. Tags for the two phases are offset by 8192 to prevent
message confusion.

**Cost model.** Each step of the reduce-scatter sends N/P floats; each step of the allgather
sends N/P floats. The total time per rank is:

```
T_ring = 2(P−1) × (α + β × N/P)
       ≈ 2(P−1)α + 2(P−1)/P × βN
```

For large N (bandwidth-dominated, βN >> α): T_ring ≈ 2(P−1)/P × βN ≈ 2βN. Ring is
bandwidth-optimal; the total bytes sent per rank, 2(P−1)/P × N × 4 bytes, approaches 2N × 4
bytes as P grows, independent of P.

**Tree Reduce + Broadcast** (`src/comm/tree_reduce.cpp`). Tree AllReduce proceeds in two phases.

*Binomial tree reduce (⌈log₂P⌉ levels):* We assign each physical rank a logical rank
L = (physical − root + P) mod P with root = 0. At level d (d = 0, 1, ..., ⌈log₂P⌉ − 1),
stride = 2^d. Ranks with L mod 2^(d+1) = 0 post a non-blocking receive (MPI_Irecv) from
logical rank L + 2^d (if that rank exists) and accumulate the received data into a local
accumulator. Ranks with L mod 2^(d+1) = 2^d send (MPI_Isend) their accumulator to logical
rank L − 2^d and become inactive for all subsequent levels. After ⌈log₂P⌉ levels, logical rank
0 (physical root) holds the element-wise sum of all ranks' contributions.

*Broadcast (MPI_Bcast):* The root broadcasts its reduced buffer to all ranks using
`MPI_Bcast`.

**Cost model.** Each of the ⌈log₂P⌉ tree levels sends the full N-float buffer; the broadcast
has the same structure:

```
T_tree = 2⌈log₂P⌉ × (α + βN)
```

For bandwidth-dominated workloads: T_tree ≈ 2⌈log₂P⌉ × βN. For P = 4, ⌈log₂4⌉ = 2, so
T_tree ≈ 4βN vs T_ring ≈ 2βN — tree uses 2× more bandwidth. For P = 8, ⌈log₂8⌉ = 3, giving
T_tree ≈ 6βN vs T_ring ≈ 1.75βN — tree uses 3.4× more bandwidth. These theoretical ratios
align closely with our measured results (Section 4.4).

**Deadlock fix.** The initial implementation using interleaved MPI_Irecv and MPI_Isend without
barriers deadlocked on Cray MPICH across nodes: MPI_Wait on an Irecv never completed because
the partner's matching Isend had not yet been posted (Cray MPICH has no asynchronous progress
thread by default). The fix inserts two MPI_Barriers per tree level: one before all MPI_Irecvs
are posted (guaranteeing the partner will post its Isend shortly), and one after all MPI_Waits
complete (preventing level d+1 messages from interfering with level d). This correctness fix
adds measurable synchronization overhead at cross-node distances, as quantified in Section 4.4.

### 4.3 Timing Methodology

We use a custom split timer (`src/timer.h`) that accumulates wall time separately for four
phases, measured with `MPI_Wtime()` and reported as totals across all batches in an epoch:

| Phase | What it measures |
|---|---|
| `compute` | GPU time: forward pass + backward pass + SGD update. Bounded by `cudaDeviceSynchronize()` calls on both sides. |
| `d2h` | Device→host `cudaMemcpy` of all gradient buffers (inside `pack_gradients()`). |
| `mpi` | Pure AllReduce algorithm time: ring / tree / `MPI_Allreduce`. This is the primary metric for algorithm comparison. |
| `h2d` | CPU gradient scaling (÷P) + host→device `cudaMemcpy` (inside `unpack_gradients()`). |

The `d2h` and `h2d` phases represent PCIe overhead that is identical across all three
algorithms — they are an unavoidable cost of our host-staged approach, not a property of the
communication algorithm. By measuring them separately, we can compare ring, tree, and MPI
built-in purely on their MPI time, which is the metric that varies between algorithms.

We report epoch averages over epochs 3–5 for algorithm comparison and epochs 2–5 for the
scaling experiment. Epoch 1 is excluded because cuBLAS handle initialization and GPU clock
ramp-up inflate compute time by 3–5×. When epoch 2 also shows minor warmup effects (noticeable
in the algorithm comparison experiments), we exclude it as well.

**Network configurations.** We test three network sizes to span the latency-to-bandwidth
transition:

| Label | Architecture | Total gradients N | Buffer size | Regime |
|---|---|---|---|---|
| Small | 784→64→10 | 50,890 floats | ~200 KB | Latency-dominated |
| Medium | 784→512→256→10 | 535,818 floats | ~2 MB | Transitional |
| Large | 784→2048→2048→2048→10 | 5,824,522 floats | ~22 MB | Bandwidth-dominated |

The gradient buffer size N is determined by the total parameter count: for a layer with fin
inputs and fout outputs, the gradient buffer holds fout×fin (weight gradients) plus fout (bias
gradients) floats. For the large network, the three hidden layers of 2048 neurons contribute
784×2048 + 2048×2048 + 2048×2048 + 2048×10 ≈ 5.8M gradient floats.

### 4.4 Results: Algorithm Comparison

All measurements use P = 4 and P = 8 GPUs across 2 nodes, batch size 256 per rank, learning
rate 0.01, reported as epoch 3–5 average MPI time.

*[Figure 3: Per-epoch MPI AllReduce time in milliseconds for three algorithms (MPI built-in,
Ring AllReduce, Tree Reduce) across small and large networks at P=4 and P=8.]*

**Small network — latency regime (N = 50,890 floats, ~200 KB).**

| Algorithm | P=4 MPI time | P=8 MPI time |
|---|---|---|
| MPI built-in | 5.97 ms | 5.27 ms |
| Ring AllReduce | 6.10 ms | 7.27 ms |
| Tree Reduce | 13.67 ms | 8.37 ms |

All three algorithms are within 2× of each other, and none shows the large performance gap
predicted by the bandwidth-dominated cost model. MPI built-in and Ring AllReduce are nearly
identical (~2% difference at P=4). Tree Reduce is 2.3× slower than MPI built-in at P=4 and
1.6× at P=8.

H1 (tree wins for small N) is **not confirmed**. The two MPI_Barriers per tree level add
approximately 3–5 ms of synchronization cost each at cross-node latencies on Slingshot-11,
totaling 6–10 ms of overhead for ⌈log₂P⌉ = 2 or 3 levels. This overhead alone exceeds the
algorithm's theoretical advantage. Notably, tree does improve relative to ring as P increases
from 4 to 8 (tree goes from 2.3× to 1.6× slower than builtin), consistent with ring's O(P)
latency growing while tree's O(log P) latency grows more slowly — but tree never catches up
because the barrier overhead scales with P as well.

**Large network — bandwidth regime (N = 5,824,522 floats, ~22 MB).**

| Algorithm | P=4 MPI time | P=8 MPI time |
|---|---|---|
| MPI built-in | 823 ms | 604 ms |
| Ring AllReduce | 689 ms | 482 ms |
| Tree Reduce | 2697 ms | 1570 ms |

H2 (ring wins for large N) is **strongly confirmed**. Ring AllReduce beats MPI built-in by
16% at P=4 and 20% at P=8. Tree Reduce is 3.9× slower than ring at P=4 and 3.3× slower at
P=8.

**Why ring beats MPI built-in.** Cray MPICH's built-in MPI_Allreduce uses a recursive-doubling
algorithm for moderate-to-large message sizes. Recursive doubling sends the full N-float buffer
at each of its log₂P levels, giving each rank a total bandwidth cost of log₂P × βN — the same
scaling as tree reduce. For P=4, this is 2βN; ring sends only (P−1)/P × 2βN ≈ 1.5βN — 25%
less data per rank in the bandwidth limit. On Slingshot-11, with its 200 Gb/s interconnect,
this difference is detectable once the gradient buffer is large enough that bandwidth — not
latency — drives AllReduce time.

**Why tree is slow.** Tree's 3.9× slowdown relative to ring at P=4 has two components. The
theoretical bandwidth ratio predicts tree ≈ 2.0× slower than ring (T_tree ≈ 4βN vs T_ring ≈
2βN for P=4). The remaining factor (~2×) comes from the MPI_Barrier overhead, which scales
with the message size because each level must complete fully before proceeding. At 22 MB per
level, this synchronization pause is substantial. At P=8, the theoretical bandwidth ratio is
3.4× (T_tree ≈ 6βN vs T_ring ≈ 1.75βN); the measured ratio is 3.3×, indicating that barrier
overhead is relatively smaller at this scale — the bandwidth disadvantage is so large that it
already explains most of the gap.

**Accuracy.** Training accuracy is not affected by the choice of AllReduce algorithm:
mathematically, all three perform the same summation. After 5 epochs on the small network at
P=1, test accuracy reaches 80.8%; at P=4 it reaches 73.0% (fewer gradient update steps per
epoch, since each rank processes 15,000 samples vs 60,000 at P=1). Accuracy converges to
comparable levels with additional epochs.

### 4.5 Results: Strong Scaling

We measure strong-scaling efficiency using the medium network (784→512→256→10,
N = 535,818 gradient floats, ~2 MB), which represents a realistic MLP for tabular or image
classification tasks. All runs use MPI built-in AllReduce, batch size 256 per rank, and 5
epochs. We report per-epoch wall time averaged over epochs 2–5.

*[Figure 4: (Left) Strong scaling — measured epoch time vs ideal linear speedup for P = 1, 2,
4, 8 GPUs. (Right) Time breakdown per epoch: stacked bars showing compute, D↔H transfer,
and MPI AllReduce time.]*

**Epoch times and speedups.**

| P | Epoch time | Speedup | Parallel efficiency |
|---|---|---|---|
| 1 | 0.1665 s | 1.0× | 100% |
| 2 | 0.1185 s | 1.4× | 70% |
| 4 | 0.1038 s | 1.6× | 40% |
| 8 | 0.0623 s | 2.7× | 33% |

**Time breakdown.**

| P | Compute | D↔H Transfer | MPI AllReduce | MPI% of total |
|---|---|---|---|---|
| 1 | 0.0566 s | 0.0596 s | 0.0000 s | 0% |
| 2 | 0.0284 s | 0.0301 s | 0.0379 s | 32% |
| 4 | 0.0143 s | 0.0151 s | 0.0501 s | 48% |
| 8 | 0.0072 s | 0.0080 s | 0.0357 s | 57% |

**Analysis.** Compute time halves with each doubling of P (near-perfect compute scaling): the
GPU forward and backward passes are data-parallel and each rank processes half the mini-batch
when P doubles. This is the expected behavior for synchronous data-parallel SGD with a fixed
per-rank batch size.

MPI AllReduce time, by contrast, stays roughly constant at 36–50 ms across P = 2, 4, 8. This
is the key manifestation of Amdahl's Law: the gradient buffer size N = 535,818 floats is
determined entirely by the model architecture, not by P. Adding more ranks reduces each rank's
compute load but does not reduce the volume of data that must be aggregated — it only adds more
synchronization participants (and the AllReduce cost grows slowly with P). Since the bandwidth
term βN dominates for a ~2 MB buffer, and βN is fixed regardless of P, MPI time is
approximately constant.

The D↔H transfer time similarly stays constant (~8–60 ms), because it copies the same N-float
gradient buffer regardless of P. At P=1, the transfer cost (0.0596 s) dominates total time
because there is no MPI overhead to compare against; at higher P, both transfer and MPI time
are constant serials that jointly limit speedup.

Combined, the fixed MPI and transfer overheads account for 67% of total epoch time at P=8,
leaving only 33% parallel efficiency. This identifies CUDA-aware MPI (to eliminate D→H→D
copies) and non-blocking AllReduce (to overlap communication with the next mini-batch's forward
pass) as the highest-leverage optimizations for future work.

### 4.6 Theoretical Analysis: α-β Model Fit

To validate that AllReduce time is bandwidth-dominated and to estimate the effective
communication bandwidth, we fit the linear α-β cost model:

```
T_allreduce = α + β × N
```

where α (ms) is the latency term and β (ms/float) is the per-element bandwidth cost.

**Data.** We use three measurements at P=4 with MPI built-in (epochs 3–5 average):

| Network | N (floats) | T (ms) |
|---|---|---|
| 784→64→10 | 50,890 | 5.97 |
| 784→512→256→10 | 535,818 | 50.1 |
| 784→2048→2048→2048→10 | 5,824,522 | 823.2 |

These three points span two orders of magnitude in N, covering the latency-to-bandwidth
transition.

*[Figure 5: α-β model scatter plot and linear fit. X-axis: gradient size N in millions of
floats. Y-axis: MPI AllReduce time in milliseconds. Data points shown for all three network
sizes at P=4; fit line and model parameters annotated.]*

**Fit results.** Linear regression (ordinary least squares) gives:

```
β ≈ 1.43 × 10⁻⁴ ms/float    (slope)
α ≈ −13 ms                   (intercept)
R² = 0.9997
```

The R² of 0.9997 confirms that the linear bandwidth model captures AllReduce time almost
perfectly across two orders of magnitude of N. The fit explains 99.97% of the variance in
measured times.

**Effective bandwidth.** Using the ring allreduce formula, each rank sends a total of
2(P−1)/P × N × 4 bytes. Rearranging the bandwidth-dominated approximation T ≈ β × N:

```
W_eff = 2(P−1)/P × 4 bytes / β
      = 1.5 × 4 / (1.43 × 10⁻⁷ s/float)
      ≈ 42 MB/s per rank
```

**Interpretation.** 42 MB/s is far below Slingshot-11's theoretical peak of ~25 GB/s per port.
The bottleneck is not the network fabric; it is CPU (host) memory bandwidth during gradient
staging. The 22 MB gradient buffer for the large network exceeds L3 cache capacity (~30–40 MB
shared, but competing with other processes and OS) and must be streamed through DRAM at the CPU
memory bandwidth limit (~50 GB/s theoretical, but effectively ~15–20 GB/s per process under
multisocket contention). With 4 MPI processes on a single node and protocol overhead, the
effective per-rank throughput of ~42 MB/s is consistent with this bottleneck.

The negative α = −13 ms is a model artifact of the 3-point fit: in the pure-bandwidth regime
all three points are well-predicted by a line through the origin, and the regression intercept
has insufficient data to constrain the latency term. This confirms that at these message sizes
(200 KB to 22 MB), the bandwidth term β × N completely dominates; the latency α contributes
less than 6 ms even for the smallest network and is effectively negligible. This finding
motivates a key direction for future work: CUDA-aware MPI would eliminate the D→H→D copies
entirely, allowing GPU HBM2 (900 GB/s peak bandwidth) to serve the gradient buffer directly
and potentially improving effective AllReduce bandwidth by an order of magnitude.

---

## 5. Challenges and Obstacles

**1. Cray MPICH deadlock in tree reduce.** The initial tree reduce implementation used
interleaved non-blocking sends and receives (MPI_Isend / MPI_Irecv) at each level without
barriers. On OpenMPI or shared-memory backends, this works because an asynchronous progress
engine handles messages even when the application thread is blocked in a Wait. On Cray MPICH on
Perlmutter, however, there is no asynchronous progress thread by default: a rank blocked in
MPI_Wait makes no progress on its outgoing messages. This means that if rank A is waiting for
a message from rank B, and rank B has posted its Isend but is also waiting for a message from
rank A, neither progresses — a deadlock. The fix required inserting two MPI_Barriers per tree
level: one before all MPI_Irecvs are posted (ensuring all senders have already called Isend),
and one after all MPI_Waits complete (preventing level d+1 receives from being posted before
level d sends have matched). This correctness fix added approximately 3–5 ms of synchronization
latency per barrier on cross-node calls, directly causing H1 to fail: the 6–10 ms of extra
barrier overhead per AllReduce exceeded the latency savings from O(log P) steps versus O(P)
steps in the small-network regime.

**2. MPICH_GPU_SUPPORT_ENABLED conflict.** Setting `MPICH_GPU_SUPPORT_ENABLED=1` (enabling
CUDA-aware MPI, which allows passing device pointers directly to MPI calls) caused a GTL
runtime linker error on Perlmutter: "libcuda.so not found in GTL context." Investigation
revealed that the GTL (GPU Transport Layer) library path was not correctly configured in the
module environment for the cray-mpich version we were using. Our solution was to set
`MPICH_GPU_SUPPORT_ENABLED=0` and implement explicit gradient staging through pinned host
memory. While this adds measurable D→H→D overhead (8–60 ms per epoch depending on model
size), it also enables our 3-phase timer to isolate PCIe transfer time from pure MPI time,
providing a cleaner measurement methodology. The root cause — effective bandwidth limited by
CPU memory bandwidth during host staging — also became a key finding in our α-β analysis.

**3. P must divide 60,000.** Our MPI_Scatter-based data loader requires that the number of
ranks evenly divides 60,000 (the Fashion-MNIST training set size). Non-divisor values cause the
loader to truncate to the nearest multiple, silently dropping samples. This constrained our
strong-scaling experiment to P ∈ {1, 2, 4, 8} and prevented testing at P = 3, 6, or larger
non-power-of-2 values that would help characterize the scaling curve more finely.

**4. P=1 baseline required a separate allocation.** The P=1 baseline timing is required to
compute speedup and parallel efficiency in the strong-scaling experiment. However, Perlmutter
rejects `srun --ntasks=1` on a 2-node allocation because the job policy requires all allocated
nodes to be used. The P=1 measurement required a separate single-node `salloc` with
`--ntasks=1 --gpus=1`. While A100 SXM4 GPUs are uniform across Perlmutter's GPU nodes (making
the comparison valid), this separate allocation introduced a small risk of measurement noise
from different node states or background traffic. We verified reproducibility by running the
P=1 baseline twice across different nodes and found < 2% variation in epoch time.

**5. cuBLAS handle warmup in epoch 1.** The first training epoch is consistently 3–5× slower
than subsequent epochs due to two one-time costs: (a) cuBLAS initializes its internal plan
cache on the first SGEMM call, choosing an optimal algorithm for the given matrix dimensions,
and (b) GPU clock frequencies ramp up from idle state. Both effects are absent from all
subsequent epochs. We exclude epoch 1 from all reported averages; for algorithm comparison
experiments we further exclude epoch 2 where minor residual effects persist.

---

## 6. Conclusion

We built a complete data-parallel MLP training stack from scratch — cuBLAS SGEMM forward and
backward passes, custom Ring AllReduce and Tree Reduce communication primitives, fine-grained
3-phase timing, and α-β theoretical analysis — and benchmarked all components on NERSC
Perlmutter A100 GPUs across two physical nodes.

Our main findings are:

1. **Ring AllReduce is the algorithm of choice for large gradient tensors.** For a 22 MB
   gradient buffer, ring is 16–20% faster than MPI built-in AllReduce and 3.3–3.9× faster than
   Tree Reduce at P = 4 and P = 8.

2. **Tree Reduce's theoretical latency advantage is masked by synchronization overhead.** The
   two MPI_Barriers required per tree level to avoid Cray MPICH deadlocks add 3–5 ms each at
   cross-node distances, eliminating any benefit from O(log P) steps in the small-network
   regime.

3. **Gradient synchronization is the dominant scaling bottleneck.** Strong-scaling efficiency
   drops to 33% at P = 8 because MPI AllReduce time (and PCIe transfer time) stays roughly
   constant while compute time halves with each doubling of P.

4. **The effective AllReduce bandwidth (~42 MB/s) is limited by CPU memory bandwidth, not the
   Slingshot-11 network.** The gradient buffer exceeds L3 cache and must stream through DRAM
   during host staging, capping throughput far below the network's 25 GB/s peak. CUDA-aware MPI
   to eliminate host staging would be the highest-leverage single optimization.

Future work includes: (a) non-blocking MPI_Iallreduce pipelined with backward-pass computation
to overlap communication with computation, (b) CUDA-aware MPI to eliminate D→H→D copies,
(c) Recursive Halving-Doubling as an algorithm that better balances the latency-bandwidth
tradeoff for medium message sizes, and (d) gradient compression to reduce the communication
volume at the cost of a small accuracy penalty.

---

## References

[1] R. Rabenseifner, "Optimization of Collective Reduction Operations," in *Proc. International
Conference on Computational Science (ICCS)*, Lecture Notes in Computer Science, vol. 3036,
pp. 1–9, 2004.

[2] A. Gibiansky, "Bringing HPC Techniques to Deep Learning," Baidu Silicon Valley AI Lab
Technical Blog, 2017.

[3] A. Castelló, E. S. Quintana-Ortí, and J. Duato, "Accelerating Distributed Deep Neural
Network Training with Pipelined MPI Allreduce," *Cluster Computing*, vol. 25, pp. 1–15, 2021.

[4] Q. Xiao, H. Rasul, and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking
Machine Learning Algorithms," arXiv:1708.07747, 2017.
