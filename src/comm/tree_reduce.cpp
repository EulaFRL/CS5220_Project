#include "tree_reduce.h"

#include <cstring>
#include <vector>

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
        if (L + stride >= P)
            continue;

        if (L % (2 * stride) == 0) {
            const int peer = physical_rank(L + stride, root, P);
            MPI_Recv(tmp.data(), n, MPI_FLOAT, peer, tag_reduce_base + d, comm,
                     MPI_STATUS_IGNORE);
            for (int i = 0; i < n; i++)
                acc[i] += tmp[i];
        } else if (L % (2 * stride) == stride) {
            const int peer = physical_rank(L - stride, root, P);
            MPI_Send(acc.data(), n, MPI_FLOAT, peer, tag_reduce_base + d, comm);
        }
    }

    if (rank == root)
        std::memcpy(recvbuf, acc.data(), (size_t)n * sizeof(float));
}
