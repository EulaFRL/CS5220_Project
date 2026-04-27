#pragma once

#include <mpi.h>

// In-place sum-reduction across ranks: on exit, buf[i] = sum over ranks of buf[i] on entry.
// Uses a ring reduce-scatter + ring allgather (no root bottleneck).
// P == 1: no-op. n == 0: no-op.
void ring_allreduce_sum_inplace(float* buf, int n, MPI_Comm comm);
