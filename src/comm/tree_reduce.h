#pragma once

#include <mpi.h>

// Binomial tree sum-reduction to one rank (MPI_Reduce semantics, not Allreduce).
// On exit, only rank `root` holds the element-wise sum of all ranks' sendbuf[].
// Other ranks' recvbuf is untouched; their sendbuf may be treated as consumed after the call.
// P == 1: copies sendbuf to recvbuf on the lone rank (must be root). n == 0: no-op.
void tree_reduce_sum(const float* sendbuf, float* recvbuf, int n, int root, MPI_Comm comm);
