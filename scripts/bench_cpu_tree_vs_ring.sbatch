#!/bin/bash
# Perlmutter CPU-only nodes (charges CPU node-hours).
#
# IMPORTANT (NERSC): GPU repos use names like m4341_g; CPU repos use the same number
# WITHOUT _g (e.g. m4341). Using m4341_g + --constraint=cpu fails policy.
# Override on submit:  sbatch -A YOUR_CPU_REPO scripts/bench_cpu_tree_vs_ring.sbatch
#
# If you have NO CPU allocation, use instead:
#   scripts/bench_tree_vs_ring_gpu_nodes.sbatch
#
#   mkdir -p logs/cpu_bench
#   sbatch scripts/bench_cpu_tree_vs_ring.sbatch

#SBATCH --job-name=cpu_tree_ring
#SBATCH --account=m4341
#SBATCH --constraint=cpu
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_bench/slurm-%j.out
#SBATCH --error=logs/cpu_bench/slurm-%j.err

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
