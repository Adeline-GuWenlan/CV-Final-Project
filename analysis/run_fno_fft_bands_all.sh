#!/bin/bash
# Apply fno_fft_bands.py to every rollout.pt in the given artifact root.
# Usage:
#   bash analysis/run_fno_fft_bands_all.sh <ARTIFACT_ROOT>
set -euo pipefail

ARTIFACT_ROOT="${1:?missing artifact root (e.g. artifacts/fno_rollouts/<TS>)}"
PROJECT_ROOT="/gpfsnyu/scratch/wg2381/PDEBench"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/artifacts/fno_fft}"

source /gpfsnyu/home/wg2381/conda/miniforge3/etc/profile.d/conda.sh
conda activate pdebench

cd "$PROJECT_ROOT"

for dir in "$ARTIFACT_ROOT"/*/; do
  stem=$(basename "$dir")
  rollout="$dir/rollout.pt"
  if [ ! -f "$rollout" ]; then
    echo "skip (no rollout.pt): $stem"
    continue
  fi
  out="$OUT_ROOT/$stem"
  mkdir -p "$out"
  echo "=== $stem ==="
  python analysis/fno_fft_bands.py \
    --rollout "$rollout" \
    --out_dir "$out" \
    --skip_initial 2>&1 | tail -5
done

echo "Done. fRMSE outputs under: $OUT_ROOT"
