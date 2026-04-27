#!/bin/bash
# Same benchmarks as bench_cpu_tree_vs_ring, but on GPU *nodes* using GPU repo (m4341_g).
# Binaries are still CPU-only MPI; set MPICH_GPU_SUPPORT_ENABLED=0 (no CUDA-aware MPI).
# Use this when your project has no Perlmutter CPU allocation (policy rejects -C cpu).
#
#   mkdir -p logs/cpu_bench
#   sbatch scripts/bench_tree_vs_ring_gpu_nodes.sbatch
#
# Override account / node count:
#   sbatch -A m4341_g --nodes=4 scripts/bench_tree_vs_ring_gpu_nodes.sbatch

#SBATCH --job-name=tree_ring_mpi
#SBATCH --account=m4341_g
#SBATCH --constraint="gpu&hbm40g"
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_bench/slurm-gpu-%j.out
#SBATCH --error=logs/cpu_bench/slurm-gpu-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
mkdir -p logs/cpu_bench

module load cray-mpich/9.0.1 2>/dev/null || module load cray-mpich 2>/dev/null || true
export MPICH_GPU_SUPPORT_ENABLED=0

export BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
export OUT_DIR="${OUT_DIR:-${ROOT}/logs/cpu_bench}"
export BENCH_ITERS="${BENCH_ITERS:-200}"
export NODE_COUNTS="${NODE_COUNTS:-2 4 8 16}"
export LAYER_SPECS="${LAYER_SPECS:-784,256,10 784,512,256,10 784,1024,512,256,10 784,2048,2048,2048,10}"

bash "${ROOT}/scripts/bench_cpu_tree_vs_ring.sh"
