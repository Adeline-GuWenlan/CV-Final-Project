#!/bin/bash
# Run fno_fft_bands.py on a single rollout.pt under Slurm.
# Usage: sbatch analysis/sbatch_fno_fft_bands.sh <ROLLOUT_PT> <OUT_DIR>

#SBATCH --account=sw5973
#SBATCH --partition=parallel
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfsnyu/scratch/wg2381/PDEBench/runs/%x-%j.out
#SBATCH --error=/gpfsnyu/scratch/wg2381/PDEBench/runs/%x-%j.err

set -euo pipefail

ROLLOUT="${1:?missing rollout.pt path}"
OUT_DIR="${2:?missing output dir}"

PROJECT_ROOT="/gpfsnyu/scratch/wg2381/PDEBench"

source /gpfsnyu/home/wg2381/conda/miniforge3/etc/profile.d/conda.sh
conda activate pdebench

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/runs"
cd "$PROJECT_ROOT"

python analysis/fno_fft_bands.py \
  --rollout "$ROLLOUT" \
  --out_dir "$OUT_DIR" \
  --model_label "${MODEL_LABEL:-Validation}" \
  --chunk_size "${CHUNK_SIZE:-4}" \
  --skip_initial
