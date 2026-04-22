#!/usr/bin/env bash
# Quick local smoke-test: single process, tiny network, 1 epoch.
# Run from the project root: bash scripts/run_local.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

# Build
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Download data if missing
if [ ! -f "$PROJECT_ROOT/data/fashion-mnist/train-images-idx3-ubyte" ]; then
    bash "$PROJECT_ROOT/scripts/download_data.sh"
fi

# Single-rank test: tiny network, 2 epochs, small batch
mpirun -n 1 "$BUILD_DIR/mlp_train" \
    --layers 784,64,10 \
    --batch 128 \
    --epochs 2 \
    --lr 0.01 \
    --algo mpi_builtin \
    --data "$PROJECT_ROOT/data/fashion-mnist"

echo ""
echo "=== 4-rank test (simulates multi-GPU on single node) ==="
mpirun -n 4 "$BUILD_DIR/mlp_train" \
    --layers 784,256,128,10 \
    --batch 256 \
    --epochs 2 \
    --lr 0.01 \
    --algo mpi_builtin \
    --data "$PROJECT_ROOT/data/fashion-mnist"
