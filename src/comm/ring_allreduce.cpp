#include "ring_allreduce.h"

#include <algorithm>
#include <cstring>
#include <vector>

// Chunk layout: P chunks, sizes differ by at most 1 (first `rem` chunks get +1 element).
static void chunk_layout(int n, int P, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(P);
    displs.resize(P);
    int base = n / P;
    int rem  = n % P;
    int off  = 0;
    for (int i = 0; i < P; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = off;
        off += counts[i];
    }
}

void ring_allreduce_sum_inplace(float* buf, int n, MPI_Comm comm) {
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (P == 1 || n == 0)
        return;

    std::vector<int> counts, displs;
    chunk_layout(n, P, counts, displs);

    int max_chunk = 0;
    for (int c : counts)
        max_chunk = std::max(max_chunk, c);
    std::vector<float> tmp(std::max(1, max_chunk));

    const int dest = (rank + 1) % P;
    const int src  = (rank - 1 + P) % P;

    // ---- Reduce-scatter: after P-1 steps, rank r holds fully reduced data for one chunk slot ----
    for (int step = 0; step < P - 1; step++) {
        const int send_idx = (rank - step + P) % P;
        const int recv_idx = (rank - step - 1 + P) % P;
        const int sc = counts[send_idx];
        const int rc = counts[recv_idx];

        MPI_Sendrecv(buf + displs[send_idx], sc, MPI_FLOAT, dest, step,
                     tmp.data(), rc, MPI_FLOAT, src, step,
                     comm, MPI_STATUS_IGNORE);

        float* acc = buf + displs[recv_idx];
        for (int j = 0; j < rc; j++)
            acc[j] += tmp[j];
    }

    // ---- Allgather: propagate fully reduced chunks ----
    for (int step = 0; step < P - 1; step++) {
        const int send_idx = (rank - step + 1 + P) % P;
        const int recv_idx = (rank - step + P) % P;
        const int sc = counts[send_idx];
        const int rc = counts[recv_idx];

        MPI_Sendrecv(buf + displs[send_idx], sc, MPI_FLOAT, dest, step + 8192,
                     tmp.data(), rc, MPI_FLOAT, src, step + 8192,
                     comm, MPI_STATUS_IGNORE);

        std::memcpy(buf + displs[recv_idx], tmp.data(), (size_t)rc * sizeof(float));
    }
}
