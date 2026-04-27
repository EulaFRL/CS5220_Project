#!/bin/bash
# Task A: tree *reduction* — pure MPI (no CUDA binary). Same Slurm *resource* lines as
# quick_ring_test.sbatch so course accounts match supported policy (GPU-node alloc, no GPU use).
#
# Optional: if your project can submit CPU-only jobs on Perlmutter, try instead:
#   #SBATCH --constraint=cpu
#   (and remove the two GPU lines below)
#
# Usage (from project root):
#   mkdir -p logs
#   sbatch scripts/quick_tree_test.sbatch
#
# Optional argv: large n, bench iters for the n=64 run (0 = no timing on 64; default 500):
#   sbatch scripts/quick_tree_test.sbatch 10007 5000

#SBATCH --job-name=tree_test
#SBATCH --account=m4341_g
#SBATCH --constraint="gpu&hbm40g"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --time=00:15:00
#SBATCH --output=logs/tree_test-%j.out
#SBATCH --error=logs/tree_test-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
mkdir -p logs

module load cray-mpich/9.0.1 2>/dev/null || true

export MPICH_GPU_SUPPORT_ENABLED=0

BUILD="${ROOT}/build"
N_LARGE="${1:-10007}"
# Default bench iters kept moderate so the step finishes well under --time even on busy nodes.
BENCH_64="${2:-500}"

echo "=== tree_reduce_test correctness (n=${N_LARGE}) ==="
srun "${BUILD}/tree_reduce_test" "${N_LARGE}" 0

echo "=== tree_reduce_test n=64 + bench (iters=${BENCH_64}) ==="
srun "${BUILD}/tree_reduce_test" 64 "${BENCH_64}"

echo "=== done ==="
