#!/bin/bash
#SBATCH --job-name=mpi_mlp
#SBATCH --account=m4776          # replace with your allocation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # 4 MPI ranks per node = 4 GPUs per node
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
mkdir -p logs

# Modules (already loaded in interactive session; explicit here for batch)
module load cudatoolkit/12.9
module load cray-mpich/9.0.1

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"

# Enable CUDA-aware MPI (Cray MPICH supports it natively on Perlmutter)
export MPICH_GPU_SUPPORT_ENABLED=1

# ---------------------------------------------------------------------------
# Experiment 1: Small network (latency-bound, expect tree to win in Phase 2)
# ---------------------------------------------------------------------------
echo "=== Small network {784,64,10} P=8 ==="
srun --ntasks=8 "$BUILD_DIR/mlp_train" \
    --layers 784,64,10 \
    --batch 256 \
    --epochs 5 \
    --lr 0.01 \
    --algo mpi_builtin \
    --data "$PROJECT_ROOT/data/fashion-mnist"

# ---------------------------------------------------------------------------
# Experiment 2: Large network (bandwidth-bound, expect ring to win in Phase 2)
# ---------------------------------------------------------------------------
echo "=== Large network {784,2048,2048,2048,10} P=8 ==="
srun --ntasks=8 "$BUILD_DIR/mlp_train" \
    --layers 784,2048,2048,2048,10 \
    --batch 256 \
    --epochs 5 \
    --lr 0.01 \
    --algo mpi_builtin \
    --data "$PROJECT_ROOT/data/fashion-mnist"

# ---------------------------------------------------------------------------
# Strong scaling sweep: P = 1, 2, 4, 8
# ---------------------------------------------------------------------------
for P in 1 2 4 8; do
    echo "=== Strong scaling P=$P ==="
    srun --ntasks=$P "$BUILD_DIR/mlp_train" \
        --layers 784,512,256,10 \
        --batch 256 \
        --epochs 3 \
        --lr 0.01 \
        --algo mpi_builtin \
        --data "$PROJECT_ROOT/data/fashion-mnist"
done
