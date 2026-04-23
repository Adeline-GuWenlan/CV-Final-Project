#!/bin/bash
# Submit one FFT-band analysis job per saved U-Net rollout.
# Usage:
#   bash analysis/submit_unet_fft_jobs.sh <ROLLOUT_ROOT>
set -euo pipefail

ROLLOUT_ROOT="${1:?missing rollout root (e.g. artifacts/unet_rollouts/<TS>)}"
PROJECT_ROOT="/gpfsnyu/scratch/wg2381/PDEBench"
SBATCH_SCRIPT="$PROJECT_ROOT/analysis/sbatch_fno_fft_bands.sh"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/artifacts/unet_fft}"

mkdir -p "$OUT_ROOT" "$PROJECT_ROOT/runs"

for dir in "$ROLLOUT_ROOT"/*/; do
  stem=$(basename "$dir")
  rollout="$dir/rollout.pt"
  if [ ! -f "$rollout" ]; then
    echo "skip (no rollout.pt): $stem"
    continue
  fi
  sbatch --job-name="unet-fft-$stem" \
    --export=ALL,MODEL_LABEL="U-Net validation" \
    "$SBATCH_SCRIPT" "$rollout" "$OUT_ROOT/$stem"
done

echo "Rollout root: $ROLLOUT_ROOT"
echo "FFT output root: $OUT_ROOT"
echo "Check queue: squeue -u $USER"
echo "Logs: /gpfsnyu/scratch/wg2381/PDEBench/runs/unet-fft-<stem>-<JOBID>.out"
echo "      /gpfsnyu/scratch/wg2381/PDEBench/runs/unet-fft-<stem>-<JOBID>.err"