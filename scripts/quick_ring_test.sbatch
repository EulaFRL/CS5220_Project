#!/bin/bash
# Quick Phase-2 sanity: ring_allreduce_test + one short mlp_train with --algo ring.
#
# Usage (from project root, so SLURM_SUBMIT_DIR points here):
#   mkdir -p logs
#   sbatch scripts/quick_ring_test.sbatch
#
# Account/QOS/constraints aligned with course jobs (e.g. hw5/job-leaderboard).

#SBATCH --job-name=ring_test
#SBATCH --account=m4341_g
#SBATCH --constraint="gpu&hbm40g"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --output=logs/ring_test-%j.out
#SBATCH --error=logs/ring_test-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}"
mkdir -p logs

module load cudatoolkit/12.9 2>/dev/null || true
module load cray-mpich/9.0.1 2>/dev/null || true

export MPICH_GPU_SUPPORT_ENABLED=0

BUILD="${ROOT}/build"
DATA="${ROOT}/data/fashion-mnist"

echo "=== ring_allreduce_test (n=10007) ==="
srun "${BUILD}/ring_allreduce_test" 10007

echo "=== mlp_train ring 1 epoch ==="
srun "${BUILD}/mlp_train" \
  --algo ring \
  --epochs 1 \
  --batch 256 \
  --data "${DATA}"

echo "=== done ==="
