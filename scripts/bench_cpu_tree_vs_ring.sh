#!/usr/bin/env bash
# Pure CPU MPI: compare custom tree_reduce vs ring_allreduce over Fashion-MNIST-style
# layer strings (gradient float count n) and process counts P.
#
# Tree primitive is MPI_Reduce semantics (sum to root); ring is full allreduce.
# Timings are custom implementation vs MPI reference in each binary (see stdout).
#
# Usage (inside Slurm allocation on Perlmutter — Cray has no mpirun):
#   module load cray-mpich/9.0.1   # or your site default
#   export MPICH_GPU_SUPPORT_ENABLED=0
#   cd .../CS5220_Project
#   BENCH_ITERS=200 NODE_COUNTS="2 4 8 16" bash scripts/bench_cpu_tree_vs_ring.sh
#
# One rank per node (P = node count) when SLURM_JOB_ID is set:
#   sbatch scripts/bench_cpu_tree_vs_ring.sbatch
#
# All ranks on one node (latency toy model):  ONE_NODE_RANKS=8 BENCH_ITERS=200 bash ...

set -euo pipefail

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-0}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"
BENCH_ITERS="${BENCH_ITERS:-200}"

# Comma-separated layer widths (input … classes), same convention as mlp_train --layers
# Default set: shallow → deep (gradient size grows).
LAYER_SPECS="${LAYER_SPECS:-784,256,10 784,512,256,10 784,1024,512,256,10 784,2048,2048,2048,10}"

# MPI rank / node counts to try (must be ≤ nodes in your allocation when multi-node).
NODE_COUNTS="${NODE_COUNTS:-2 4 8 16}"

# If set to non-empty, ignore multi-node layout and run P ranks on a single node / localhost.
ONE_NODE_RANKS="${ONE_NODE_RANKS:-}"

log() { printf '%s\n' "$*" | tee -a "${LOG_FILE:-/dev/stdout}"; }

grad_n_from_layers() {
    local spec="$1"
    python3 -c "s=[int(x.strip()) for x in '${spec}'.split(',')]; print(sum(s[i+1]*s[i]+s[i+1] for i in range(len(s)-1)))"
}

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
    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        command -v srun
        return 0
    fi
    return 1
}

if [[ ! -x "$BUILD_DIR/tree_reduce_test" || ! -x "$BUILD_DIR/ring_allreduce_test" ]]; then
    echo "Missing $BUILD_DIR/{tree_reduce_test,ring_allreduce_test} — build the project first." >&2
    exit 1
fi

if ! LAUNCHER="$(resolve_mpi_launcher)"; then
    cat >&2 <<'EOF'
No MPI launcher found. On Perlmutter use Slurm (salloc/sbatch) and srun, or set MPIRUN_CMD.
EOF
    exit 1
fi

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/cpu_bench}"
mkdir -p "$OUT_DIR"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/tree_vs_ring_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log}"
export LOG_FILE

log "=== bench_cpu_tree_vs_ring ==="
log "BUILD_DIR=$BUILD_DIR  LAUNCHER=$LAUNCHER  BENCH_ITERS=$BENCH_ITERS"
log "LAYER_SPECS=$LAYER_SPECS"
log "NODE_COUNTS=$NODE_COUNTS  ONE_NODE_RANKS=${ONE_NODE_RANKS:-<unset>}"
log "MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log "Slurm job=$SLURM_JOB_ID nodes=${SLURM_NNODES:-?} JOB_NODELIST=${SLURM_JOB_NODELIST:-?}"
fi
log ""

# MPI_PREFIX must include the launcher as argv[0] (e.g. full path to srun).
run_mpi() { "$@"; }

MAX_ALLOC_NODES="${SLURM_NNODES:-999}"
if [[ -n "${ONE_NODE_RANKS}" ]]; then
    P="${ONE_NODE_RANKS}"
    if [[ "$LAUNCHER" == *srun ]]; then
        MPI_PREFIX=("$LAUNCHER" -n "$P" --cpus-per-task=2)
    else
        MPI_PREFIX=("$LAUNCHER" -n "$P")
    fi
    log "--- single placement: P=$P ranks (ONE_NODE_RANKS) ---"
    for layers in $LAYER_SPECS; do
        N="$(grad_n_from_layers "$layers")"
        log ""
        log "### layers=$layers  n=$N  P=$P"
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/tree_reduce_test" "$N" 0
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/tree_reduce_test" "$N" "$BENCH_ITERS"
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/ring_allreduce_test" "$N" 0
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/ring_allreduce_test" "$N" "$BENCH_ITERS"
    done
    log ""
    log "Wrote log: $LOG_FILE"
    exit 0
fi

for P in $NODE_COUNTS; do
    if [[ "$P" -gt "$MAX_ALLOC_NODES" ]]; then
        log "--- skip P=$P (allocation has only $MAX_ALLOC_NODES nodes) ---"
        continue
    fi
    if [[ "$LAUNCHER" == *srun ]]; then
        MPI_PREFIX=("$LAUNCHER" -N "$P" --ntasks-per-node=1 -n "$P")
    else
        MPI_PREFIX=("$LAUNCHER" -n "$P")
    fi
    log ""
    log "--- P=$P MPI ranks (one rank per node when using srun) ---"
    for layers in $LAYER_SPECS; do
        N="$(grad_n_from_layers "$layers")"
        log ""
        log "### layers=$layers  n=$N  P=$P"
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/tree_reduce_test" "$N" 0
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/tree_reduce_test" "$N" "$BENCH_ITERS"
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/ring_allreduce_test" "$N" 0
        run_mpi "${MPI_PREFIX[@]}" "$BUILD_DIR/ring_allreduce_test" "$N" "$BENCH_ITERS"
    done
done

log ""
log "Done. Log: $LOG_FILE"
log "Grep timings:  grep -E 'tree_reduce:|ring_allreduce:|MPI_' \"$LOG_FILE\""
