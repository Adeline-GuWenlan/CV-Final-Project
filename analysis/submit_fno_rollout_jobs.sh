#!/bin/bash
# 2026-04-20: Submit one rollout-save job per 2D CFD regime.

set -euo pipefail

PROJECT_ROOT="/gpfsnyu/scratch/wg2381/PDEBench"
SBATCH_SCRIPT="$PROJECT_ROOT/analysis/sbatch_save_fno_rollout.sh"
TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
ARTIFACT_ROOT="$PROJECT_ROOT/artifacts/fno_rollouts/$TS"

mkdir -p "$ARTIFACT_ROOT" "$PROJECT_ROOT/runs"

for REGIME in \
  2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 \
  2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
  2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
do
  sbatch --job-name="fno-rollout-$(basename "$REGIME" .hdf5)" \
    "$SBATCH_SCRIPT" "$REGIME" "$ARTIFACT_ROOT"
done

echo "Artifact root: $ARTIFACT_ROOT"
echo "Check queue: squeue -u $USER"
echo "Logs: /gpfsnyu/scratch/wg2381/PDEBench/runs/fno-rollout-<stem>-<JOBID>.out"
echo "      /gpfsnyu/scratch/wg2381/PDEBench/runs/fno-rollout-<stem>-<JOBID>.err"
