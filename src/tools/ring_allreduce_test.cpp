// Standalone MPI test: ring_allreduce vs MPI_Allreduce (no CUDA).
#include "comm/ring_allreduce.h"

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

    int n = 10007;
    if (argc >= 2)
        n = std::atoi(argv[1]);
    int bench_iters = 0;
    if (argc >= 3)
        bench_iters = std::atoi(argv[2]);

    std::mt19937 rng(12345u + (unsigned)rank);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> local_init((size_t)n);
    for (int i = 0; i < n; i++)
        local_init[i] = dist(rng);

    std::vector<float> x = local_init, y = local_init;

    ring_allreduce_sum_inplace(x.data(), n, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, y.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    const float err = max_abs_diff(x, y);
    const float tol = 1e-3f * std::max(1, world);
    // Use pass (1/0) with MPI_PROD: PROD of fail flags would be 0 when everyone passes.
    const int pass = (err <= tol) ? 1 : 0;

    int all_ok = 0;
    MPI_Allreduce(&pass, &all_ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    if (rank == 0) {
        if (all_ok)
            printf("ring_allreduce_test PASSED  n=%d  P=%d  max|diff|=%g\n", n, world, err);
        else
            printf("ring_allreduce_test FAILED  n=%d  P=%d  max|diff|=%g\n", n, world, err);
        fflush(stdout);
    }

    // Optional: bandwidth/latency — ring vs MPI_Allreduce (same n as packed MLP grads).
    if (bench_iters > 0 && all_ok) {
        std::vector<float> buf((size_t)n);
        const int warmup = std::min(50, std::max(5, bench_iters / 10));
        for (int i = 0; i < warmup; i++) {
            std::copy(local_init.begin(), local_init.end(), buf.begin());
            ring_allreduce_sum_inplace(buf.data(), n, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            std::copy(local_init.begin(), local_init.end(), buf.begin());
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_ring0 = MPI_Wtime();
        for (int i = 0; i < bench_iters; i++) {
            std::copy(local_init.begin(), local_init.end(), buf.begin());
            ring_allreduce_sum_inplace(buf.data(), n, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double t_ring1 = MPI_Wtime();

        double t_mpi0 = MPI_Wtime();
        for (int i = 0; i < bench_iters; i++) {
            std::copy(local_init.begin(), local_init.end(), buf.begin());
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        double t_mpi1 = MPI_Wtime();

        if (rank == 0) {
            const double sec_ring = (t_ring1 - t_ring0) / bench_iters;
            const double sec_mpi  = (t_mpi1 - t_mpi0) / bench_iters;
            printf("bench  n=%d  P=%d  iters=%d\n", n, world, bench_iters);
            printf("  ring_allreduce: avg %.6e s  (~%.2f us/call)\n", sec_ring, sec_ring * 1e6);
            printf("  MPI_Allreduce:  avg %.6e s  (~%.2f us/call)\n", sec_mpi, sec_mpi * 1e6);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
