#!/bin/bash
# 2026-04-20: Run one PDEBench 2D CFD FNO eval regime under Slurm and save the
# resulting pickle into a timestamped artifact directory.
#
# Usage:
#   sbatch analysis/sbatch_eval_fno_regime.sh <REGIME_FILENAME> <ARTIFACT_ROOT>

#SBATCH --account=sw5973
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
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

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/runs"

cd "$MODELS_DIR"

python train_models_forward.py \
  +args=config_2DCFD.yaml \
  ++args.model_name='FNO' \
  ++args.if_training=False \
  ++args.data_path="$DATA_DIR" \
  ++args.filename="$REGIME"

cp "${STEM}_FNO.pickle" "$OUT_DIR/"
