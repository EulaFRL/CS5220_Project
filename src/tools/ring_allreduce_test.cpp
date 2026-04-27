// Standalone MPI test: ring_allreduce vs MPI_Allreduce (no CUDA).
#include "comm/ring_allreduce.h"

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

    std::mt19937 rng(12345u + (unsigned)rank);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x((size_t)n), y((size_t)n);
    for (int i = 0; i < n; i++) {
        float v = dist(rng);
        x[i] = v;
        y[i] = v;
    }

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
    }

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
