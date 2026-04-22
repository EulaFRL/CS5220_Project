#!/bin/bash
#SBATCH -A m4341_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH -o /global/u2/e/eulawang/cs5220-project/logs/smoke_%j.out
#SBATCH -e /global/u2/e/eulawang/cs5220-project/logs/smoke_%j.err
#SBATCH --job-name=mlp_smoke

mkdir -p /global/u2/e/eulawang/cs5220-project/logs

export MPICH_GPU_SUPPORT_ENABLED=1

srun /global/u2/e/eulawang/cs5220-project/build/mlp_train \
  --layers 784,64,10 \
  --batch 128 \
  --epochs 3 \
  --lr 0.01 \
  --data /global/u2/e/eulawang/cs5220-project/data/fashion-mnist
