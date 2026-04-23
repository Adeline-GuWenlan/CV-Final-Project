#!/bin/bash
# 2026-04-20: Save one PDEBench 2D CFD FNO validation rollout under Slurm.
#
# Usage:
#   sbatch analysis/sbatch_save_fno_rollout.sh <REGIME_FILENAME> <ARTIFACT_ROOT>

#SBATCH --account=sw5973
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfsnyu/scratch/wg2381/PDEBench/runs/%x-%j.out
#SBATCH --error=/gpfsnyu/scratch/wg2381/PDEBench/runs/%x-%j.err

set -euo pipefail

REGIME="${1:?missing regime filename}"
ARTIFACT_ROOT="${2:?missing artifact root}"
STEM="${REGIME%.hdf5}"
PROJECT_ROOT="/gpfsnyu/scratch/wg2381/PDEBench"
MODELS_DIR="$PROJECT_ROOT/pdebench/models"
DATA_DIR="/gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand"
OUT_DIR="$ARTIFACT_ROOT/$STEM"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_SAMPLES_MAX="${NUM_SAMPLES_MAX:--1}"

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/runs"

source /gpfsnyu/home/wg2381/conda/miniforge3/etc/profile.d/conda.sh
conda activate pdebench

cd "$MODELS_DIR"

python eval_fno_2dcfd_rollout.py \
  --filename "$REGIME" \
  --data_path "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --num_samples_max "$NUM_SAMPLES_MAX"
