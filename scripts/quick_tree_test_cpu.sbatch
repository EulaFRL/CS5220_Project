#!/bin/bash
# Pure CPU + MPI only (Perlmutter CPU nodes). No GPUs, no CUDA module.
#
# If sbatch says "does not match any supported policy", your allocation may be
# GPU-node-only (use scripts/quick_tree_test.sbatch) or need a different -A / --qos.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/quick_tree_test_cpu.sbatch
#   sbatch scripts/quick_tree_test_cpu.sbatch 10007 500

#SBATCH --job-name=tree_cpu
#SBATCH --account=m4341_g
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --output=logs/tree_cpu-%j.out
#SBATCH --error=logs/tree_cpu-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
mkdir -p logs

module load cray-mpich/9.0.1 2>/dev/null || true
export MPICH_GPU_SUPPORT_ENABLED=0

BUILD="${ROOT}/build"
N_LARGE="${1:-10007}"
BENCH_64="${2:-500}"

echo "=== tree_reduce_test correctness (n=${N_LARGE}) ==="
srun "${BUILD}/tree_reduce_test" "${N_LARGE}" 0

echo "=== tree_reduce_test n=64 + bench (iters=${BENCH_64}) ==="
srun "${BUILD}/tree_reduce_test" 64 "${BENCH_64}"

echo "=== done ==="
