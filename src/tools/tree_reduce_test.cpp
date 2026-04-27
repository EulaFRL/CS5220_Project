// Standalone MPI: tree_reduce vs MPI_Reduce + optional latency (small n = shallow MLP payload).
#include "comm/tree_reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <mpi.h>

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const int root = 0;
    int n = 64;
    if (argc >= 2)
        n = std::atoi(argv[1]);
    int bench_iters = 0;
    if (argc >= 3)
        bench_iters = std::atoi(argv[2]);

    std::mt19937 rng(12345u + (unsigned)rank);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> local((size_t)n), mpi_ref((size_t)n, 0.0f), tree_out((size_t)n, 0.0f);
    for (int i = 0; i < n; i++)
        local[i] = dist(rng);

    tree_reduce_sum(local.data(), tree_out.data(), n, root, MPI_COMM_WORLD);
    MPI_Reduce(local.data(), mpi_ref.data(), n, MPI_FLOAT, MPI_SUM, root, MPI_COMM_WORLD);

    float err = 0.0f;
    if (rank == root)
        err = max_abs_diff(tree_out, mpi_ref);

    const float tol = 1e-3f * std::max(1, world);
    const int pass = (rank != root || err <= tol) ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&pass, &all_ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    if (rank == root) {
        if (all_ok)
            printf("tree_reduce_test PASSED  n=%d  P=%d  root=%d  max|diff|=%g\n",
                   n, world, root, err);
        else
            printf("tree_reduce_test FAILED  n=%d  P=%d  root=%d  max|diff|=%g\n",
                   n, world, root, err);
        fflush(stdout);
    }

    // Optional: latency for small n (shallow/narrow MLP gradients): tree vs MPI_Reduce.
    if (bench_iters > 0 && all_ok) {
        const int warmup = std::min(50, std::max(5, bench_iters / 10));
        for (int i = 0; i < warmup; i++) {
            tree_reduce_sum(local.data(), tree_out.data(), n, root, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Reduce(local.data(), tree_out.data(), n, MPI_FLOAT, MPI_SUM, root,
                       MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_tree0 = MPI_Wtime();
        for (int i = 0; i < bench_iters; i++) {
            tree_reduce_sum(local.data(), tree_out.data(), n, root, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double t_tree1 = MPI_Wtime();

        double t_mpi0 = MPI_Wtime();
        for (int i = 0; i < bench_iters; i++) {
            MPI_Reduce(local.data(), tree_out.data(), n, MPI_FLOAT, MPI_SUM, root,
                       MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double t_mpi1 = MPI_Wtime();

        if (rank == root) {
            const double sec_tree = (t_tree1 - t_tree0) / bench_iters;
            const double sec_mpi  = (t_mpi1 - t_mpi0) / bench_iters;
            printf("bench  n=%d  P=%d  iters=%d\n", n, world, bench_iters);
            printf("  tree_reduce:  avg %.6e s  (~%.2f us/call)\n", sec_tree, sec_tree * 1e6);
            printf("  MPI_Reduce:   avg %.6e s  (~%.2f us/call)\n", sec_mpi, sec_mpi * 1e6);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
