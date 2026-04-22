#pragma once
#include <mpi.h>
#include <cstdio>

// Tracks compute vs. communication time separately.
// Caller must issue cudaDeviceSynchronize() before start/stop when timing GPU work.
struct Timer {
    double t_compute = 0.0;
    double t_comm    = 0.0;
    double t0        = 0.0;

    void start()        { t0 = MPI_Wtime(); }
    void stop_compute() { t_compute += MPI_Wtime() - t0; }
    void stop_comm()    { t_comm    += MPI_Wtime() - t0; }
    void reset()        { t_compute = t_comm = 0.0; }

    void report_epoch(int rank, int epoch) const {
        if (rank != 0) return;
        double total = t_compute + t_comm;
        printf("[Epoch %2d] compute=%.4fs  comm=%.4fs  total=%.4fs  comm%%=%.1f%%\n",
               epoch, t_compute, t_comm, total,
               total > 0 ? 100.0 * t_comm / total : 0.0);
    }
};
