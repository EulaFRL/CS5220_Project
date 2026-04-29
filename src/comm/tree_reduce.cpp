#include "tree_reduce.h"

#include <cstring>
#include <vector>
#include <mpi.h>

namespace {

constexpr int tag_reduce_base = 6000;

static int logical_rank(int phys, int root, int P) {
    return (phys - root + P) % P;
}

static int physical_rank(int logical, int root, int P) {
    return (logical + root) % P;
}

} // namespace

void tree_reduce_sum(const float* sendbuf, float* recvbuf, int n, int root, MPI_Comm comm) {
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (n == 0)
        return;

    if (P == 1) {
        if (rank == root) {
            if (recvbuf != sendbuf)
                std::memcpy(recvbuf, sendbuf, (size_t)n * sizeof(float));
        }
        return;
    }

    std::vector<float> acc((size_t)n);
    std::memcpy(acc.data(), sendbuf, (size_t)n * sizeof(float));

    std::vector<float> tmp((size_t)n);
    int max_level = 0;
    while ((1 << max_level) < P)
        max_level++;

    const int L = logical_rank(rank, root, P);

    for (int d = 0; d < max_level; d++) {
        const int stride = 1 << d;
        const int tag = tag_reduce_base + d;

        MPI_Request req_recv = MPI_REQUEST_NULL;
        MPI_Request req_send = MPI_REQUEST_NULL;

        // Two-phase + barriers: every receiver posts Irecv before any sender starts Isend.
        // Interleaved Irecv/Isend across ranks can hang on Cray MPICH (no progress until Wait).
        if (L + stride < P && L % (2 * stride) == 0) {
            const int peer = physical_rank(L + stride, root, P);
            MPI_Irecv(tmp.data(), n, MPI_FLOAT, peer, tag, comm, &req_recv);
        }
        MPI_Barrier(comm);

        if (L % (2 * stride) == stride) {
            const int peer = physical_rank(L - stride, root, P);
            MPI_Isend(acc.data(), n, MPI_FLOAT, peer, tag, comm, &req_send);
        }

        if (req_recv != MPI_REQUEST_NULL)
            MPI_Wait(&req_recv, MPI_STATUS_IGNORE);
        if (req_send != MPI_REQUEST_NULL)
            MPI_Wait(&req_send, MPI_STATUS_IGNORE);

        if (L + stride < P && L % (2 * stride) == 0) {
            for (int i = 0; i < n; i++)
                acc[i] += tmp[i];
        }

        MPI_Barrier(comm);
    }

    if (rank == root)
        std::memcpy(recvbuf, acc.data(), (size_t)n * sizeof(float));
}

void tree_allreduce_sum_inplace(float* buf, int n, MPI_Comm comm) {
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (P == 1 || n == 0)
        return;

    std::vector<float> tmp((size_t)n);
    tree_reduce_sum(buf, tmp.data(), n, /*root=*/0, comm);
    if (rank == 0)
        std::memcpy(buf, tmp.data(), (size_t)n * sizeof(float));
    MPI_Bcast(buf, n, MPI_FLOAT, 0, comm);
}