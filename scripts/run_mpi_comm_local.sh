#!/usr/bin/env bash
# MPI smoke tests: tree_reduce_test + ring_allreduce_test.
#
# On Perlmutter + Cray MPICH, $CRAY_MPICH_DIR/bin has mpicc/mpifort only — there is NO mpirun.
# Parallel launch uses `srun` inside a Slurm allocation (salloc / sbatch / interactive job).
#
# Option A — already inside salloc / batch step (SLURM_JOB_ID is set):
#   module load PrgEnv-gnu cray-mpich/9.0.1
#   export MPICH_GPU_SUPPORT_ENABLED=0
#   bash scripts/run_mpi_comm_local.sh
#
# Option B — from login node: grab a short interactive allocation, then run the same script:
#   salloc -A m4341_g -C gpu -q interactive -N 1 --ntasks=4 --gpus-per-node=4 -t 0:30:00
#   # wait for shell on compute node, then:
#   module load PrgEnv-gnu cray-mpich/9.0.1
#   cd /pscratch/sd/r/rb945/CS5220_Project && bash scripts/run_mpi_comm_local.sh
#
# Override launcher:  MPIRUN_CMD=/path/to/mpiexec bash scripts/run_mpi_comm_local.sh
# Rank count:         NPROC=8 bash scripts/run_mpi_comm_local.sh

set -euo pipefail

# GPU-node allocations often set MPICH_GPU_SUPPORT_ENABLED=1. CPU-only binaries are not
# linked with Cray GTL — MPICH then aborts: "GPU_SUPPORT_ENABLED is requested, but GTL
# library is not linked". Always force off for this script.
export MPICH_GPU_SUPPORT_ENABLED=0

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
NPROC="${NPROC:-4}"

resolve_mpi_launcher() {
    if [[ -n "${MPIRUN_CMD:-}" ]]; then
        printf '%s' "$MPIRUN_CMD"
        return 0
    fi
    if command -v mpirun >/dev/null 2>&1; then
        command -v mpirun
        return 0
    fi
    if command -v mpiexec >/dev/null 2>&1; then
        command -v mpiexec
        return 0
    fi
    if [[ -n "${CRAY_MPICH_DIR:-}" ]]; then
        if [[ -x "${CRAY_MPICH_DIR}/bin/mpirun" ]]; then
            printf '%s' "${CRAY_MPICH_DIR}/bin/mpirun"
            return 0
        fi
        if [[ -x "${CRAY_MPICH_DIR}/bin/mpiexec" ]]; then
            printf '%s' "${CRAY_MPICH_DIR}/bin/mpiexec"
            return 0
        fi
    fi
    # Cray MPICH on Slurm: use srun only when we are inside a Slurm job allocation.
    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        command -v srun
        return 0
    fi
    return 1
}

if ! LAUNCHER="$(resolve_mpi_launcher)"; then
    cat >&2 <<EOF
No MPI launcher found.

Perlmutter + cray-mpich: \`\$CRAY_MPICH_DIR/bin\` has compiler wrappers (mpicc, …) but no mpirun/mpiexec.
You must run MPI under Slurm with \`srun\` inside an allocation (so SLURM_JOB_ID is set).

Example:
  salloc -A m4341_g -C gpu -q interactive -N 1 --ntasks=4 --gpus-per-node=4 -t 0:30:00
  module load PrgEnv-gnu cray-mpich/9.0.1
  export MPICH_GPU_SUPPORT_ENABLED=0
  cd ${PROJECT_ROOT}
  bash scripts/run_mpi_comm_local.sh

Manual srun on GPU nodes — you MUST export this first (same as sbatch scripts):
  export MPICH_GPU_SUPPORT_ENABLED=0
  srun -n ${NPROC} ./build/tree_reduce_test 10007 0
EOF
    exit 1
fi

if [[ ! -x "$BUILD_DIR/tree_reduce_test" || ! -x "$BUILD_DIR/ring_allreduce_test" ]]; then
    echo "Missing binaries under $BUILD_DIR — run cmake --build first." >&2
    exit 1
fi

echo "Using MPI launcher: $LAUNCHER"

echo "=== ${NPROC} ranks: tree_reduce_test 10007 0 ==="
"$LAUNCHER" -n "${NPROC}" "$BUILD_DIR/tree_reduce_test" 10007 0

echo "=== ${NPROC} ranks: tree_reduce_test 64 200 ==="
"$LAUNCHER" -n "${NPROC}" "$BUILD_DIR/tree_reduce_test" 64 200

echo "=== ${NPROC} ranks: ring_allreduce_test 10007 ==="
"$LAUNCHER" -n "${NPROC}" "$BUILD_DIR/ring_allreduce_test" 10007

echo "=== done ==="
