#pragma once

#include <mpi.h>

// Binomial tree sum-reduction to one rank (MPI_Reduce semantics, not Allreduce).
// On exit, only rank `root` holds the element-wise sum of all ranks' sendbuf[].
// Other ranks' recvbuf is untouched; their sendbuf may be treated as consumed after the call.
// P == 1: copies sendbuf to recvbuf on the lone rank (must be root). n == 0: no-op.
void tree_reduce_sum(const float* sendbuf, float* recvbuf, int n, int root, MPI_Comm comm);

// AllReduce wrapper: tree_reduce_sum to root=0, then MPI_Bcast back to all ranks.
// On exit, every rank's buf[] holds the element-wise sum.
// P == 1 or n == 0: no-op.
void tree_allreduce_sum_inplace(float* buf, int n, MPI_Comm comm);
