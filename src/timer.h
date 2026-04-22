#pragma once
#include <mpi.h>
#include <cstdio>

// Tracks compute and three comm sub-categories separately.
// Caller must issue cudaDeviceSynchronize() before start() when timing GPU work.
struct Timer {
    double t_compute = 0.0;  // forward + backward + SGD update
    double t_d2h     = 0.0;  // pack_gradients: device → host memcpy
    double t_mpi     = 0.0;  // MPI algorithm (Allreduce / Tree / Ring)
    double t_h2d     = 0.0;  // unpack_gradients: host → device memcpy
    double t0        = 0.0;

    void start()        { t0 = MPI_Wtime(); }
    void stop_compute() { t_compute += MPI_Wtime() - t0; }
    void stop_d2h()     { t_d2h     += MPI_Wtime() - t0; }
    void stop_mpi()     { t_mpi     += MPI_Wtime() - t0; }
    void stop_h2d()     { t_h2d     += MPI_Wtime() - t0; }
    void reset()        { t_compute = t_d2h = t_mpi = t_h2d = 0.0; }

    double t_comm() const { return t_d2h + t_mpi + t_h2d; }

    void report_epoch(int rank, int epoch) const {
        if (rank != 0) return;
        double total = t_compute + t_comm();
        printf("[Epoch %2d] compute=%.4fs  d2h=%.4fs  mpi=%.4fs  h2d=%.4fs  "
               "comm_total=%.4fs  total=%.4fs  mpi%%=%.1f%%\n",
               epoch, t_compute, t_d2h, t_mpi, t_h2d,
               t_comm(), total,
               total > 0 ? 100.0 * t_mpi / total : 0.0);
    }
};
