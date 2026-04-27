# Source before manual srun on Perlmutter GPU nodes for CPU-only MPI binaries:
#   source scripts/env_cpu_mpi.sh
#   srun -n 4 ./build/tree_reduce_test 10007 0
export MPICH_GPU_SUPPORT_ENABLED=0
