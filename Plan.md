# 2D CFD FNO Pipeline Plan

## Goal

Build a reproducible pipeline for the 2D compressible CFD part of the project in
[Proposal_Plan.tex](./Proposal_Plan.tex), starting from the official PDEBench
FNO baseline and extending it into the FFT-based diagnostic analysis proposed in
the writeup.

The immediate priority is:

1. Reuse the official 2D CFD FNO architecture and pretrained weights.
2. Evaluate that baseline on the three proposal regimes.
3. Save raw rollout outputs, not just scalar metrics.
4. Run a custom FFT error decomposition that matches the proposal.
5. Leave a clean path for adding U-Net and DoG later.

## Proposal Regimes

These are the three regimes selected in the proposal:

1. `M=0.1, eta=0.1`
2. `M=1.0, eta=0.01`
3. `M=1.0, eta~=0` implemented as `Eta1e-08_Zeta1e-08`

Official downloaded data files:

1. `2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5`
2. `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
3. `2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5`

## What The Repo Already Gives Us

### Existing training and eval entrypoints

- `pdebench/models/train_models_forward.py`
  - Unified Hydra entrypoint for FNO and U-Net forward experiments.
- `pdebench/models/fno/train.py`
  - Handles both training and evaluation.
  - If `if_training=False`, it loads `<model_name>.pt` and writes
    `<model_name>.pickle`.
- `pdebench/models/config/args/config_2DCFD.yaml`
  - Ready-made 2D CFD hyperparameter template for baseline settings.
- `pdebench/models/metrics.py`
  - Already computes spatial metrics and coarse FFT low/mid/high metrics.
- `pdebench/models/analyse_result_forward.py`
  - Converts `.pickle` outputs into a CSV and a simple plot.

### Existing baseline hyperparameters for 2D CFD FNO

From `config_2DCFD.yaml`:

- `initial_step=10`
- `t_train=21`
- `reduced_resolution=2`
- `reduced_resolution_t=1`
- `reduced_batch=1`
- `num_channels=4`
- `modes=12`
- `width=20`
- `single_file=True`

These are the first hyperparameters to keep fixed if the goal is faithful
baseline reproduction.

## Important Naming Mismatch

This is the main trap in using the official pretrained FNO weights directly.

### Data filenames

The downloaded 2D CFD HDF5 files are named with `Rand_`, for example:

- `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`

### Official pretrained FNO checkpoint filenames

The official 2D CFD FNO tar contains checkpoint names without `Rand_`, for
example:

- `2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train_FNO.pt`
- `2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train_FNO.pt`
- `2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train_FNO.pt`

### Why this matters

`pdebench/models/fno/train.py` builds the checkpoint path as:

- `model_name = filename_without_hdf5 + "_FNO"`

So if the input file is the real downloaded filename with `Rand_`, the eval
script will look for:

- `2D_CFD_Rand_..._FNO.pt`

but the official pretrained checkpoint is:

- `2D_CFD_..._FNO.pt`

### Recommended fix

Do not change the official checkpoint names.

Instead, create lightweight alias filenames for the HDF5 files without `Rand_`
using symlinks in the same data directory. Example:

```bash
cd /gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand

ln -s 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 \
      2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5

ln -s 2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
      2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5

ln -s 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
      2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
```

With that aliasing, the checkpoint stem and the data stem line up.

## No-`Rand_` HDF5 Symlink Procedure

This section records the exact workflow for making the downloaded 2D CFD HDF5
files compatible with the official pretrained checkpoint names.

### Why use symlinks instead of renaming files

1. The official downloaded HDF5 shards include `Rand_` in the filename stem.
2. The official pretrained 2D CFD checkpoints do not include `Rand_`.
3. PDEBench derives the checkpoint stem from the data filename stem.
   - FNO uses `<stem>_FNO.pt`
   - U-Net uses `<stem>_Unet...pt`
4. If we pass the real downloaded filename directly, PDEBench will look for a
   checkpoint name containing `Rand_`, which does not match the official
   release.
5. A symlink keeps the original large HDF5 file untouched, avoids a duplicate
   copy, and gives us a filename stem that matches the official checkpoints.

### How to create the no-`Rand_` aliases

Run these commands inside the directory that already contains the downloaded
2D CFD HDF5 shards:

```bash
cd /gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand

ln -s 2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 \
      2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5

ln -s 2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
      2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5

ln -s 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5 \
      2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
```

### How to use the aliases at runtime

Keep `data_path` pointing at the real data directory, but pass the symlink name
through `filename`. Example:

```bash
cd /gpfsnyu/scratch/wg2381/PDEBench/pdebench/models

CUDA_VISIBLE_DEVICES=0 python3 train_models_forward.py \
  +args=config_2DCFD.yaml \
  ++args.model_name='FNO' \
  ++args.if_training=False \
  ++args.data_path='/gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand' \
  ++args.filename='2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
```

With that input filename, PDEBench will derive the checkpoint stem
`2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train_FNO.pt`, which matches the
official pretrained file extracted into `pdebench/models/`.

### Quick verification

After creating the symlinks, verify them with:

```bash
ls -l /gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand/2D_CFD_M*.hdf5
```

The output should show each no-`Rand_` filename pointing to the corresponding
real `2D_CFD_Rand_...hdf5` file.

## How To Reuse The Existing FNO Baseline

### Evaluation mode

To evaluate an existing checkpoint, use:

- `if_training=False`

Do not use `continue_training=True` unless the intention is to resume training.

### Where checkpoint loading happens

The FNO eval script loads `<model_name>.pt` from the current working directory.
That means the simplest setup is:

1. Extract the official FNO `.pt` files into `pdebench/models/`
2. Run `train_models_forward.py` from `pdebench/models/`

### Baseline eval command pattern

```bash
cd /gpfsnyu/scratch/wg2381/PDEBench/pdebench/models

CUDA_VISIBLE_DEVICES=0 python train_models_forward.py \
  +args=config_2DCFD.yaml \
  ++args.model_name='FNO' \
  ++args.if_training=False \
  ++args.data_path='/gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand' \
  ++args.filename='2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
```

Repeat the same command with the other two regime filenames.

### Expected outputs

For each regime, PDEBench will generate:

- `<stem>_FNO.pickle`

That `.pickle` stores the aggregate metrics returned by `metrics.py`.

## Limitation Of The Existing Repo Relative To The Proposal

The repo is enough for baseline scalar reproduction, but not enough for the
full proposal analysis.

### What PDEBench already gives us

- spatial RMSE-like metrics
- coarse Fourier low/mid/high error metrics
- autoregressive rollout evaluation on the validation split

### What the proposal still needs

1. Save raw predicted and ground-truth rollout tensors.
2. Compute FFT error decomposition from the actual error field, not just one
   coarse aggregate per run.
3. Aggregate by regime, channel, time, and radial frequency band.
4. Plot spectral curves and bandwise comparisons.
5. Later compare FNO vs U-Net using the same evaluation tensors.
6. Later add DoG localization near shocks.

## Whole Pipeline

## Phase 0: Inventory And Naming

Objective:
Make the official assets runnable without changing upstream PDEBench internals.

Steps:

1. Finish downloading the three HDF5 regime files.
2. Extract the official `2DCFD_FNO.tar` and `2DCFD_Unet.tar`.
3. Create no-`Rand_` HDF5 symlink aliases.
4. Place official FNO and U-Net checkpoints in `pdebench/models/`.
5. Record exact stems for each regime in a small manifest table.

Success condition:

- For each regime, one `.hdf5` alias stem exactly matches one official `.pt`
  checkpoint stem.

## Phase 1: Official FNO Baseline Evaluation

Objective:
Run the pretrained PDEBench FNO exactly as intended on the three selected
regimes.

Steps:

1. Use `config_2DCFD.yaml` as the hyperparameter template.
2. Override only:
   - `if_training=False`
   - `data_path`
   - `filename`
3. Run one eval job per regime.
4. Save the resulting `.pickle` outputs into a dedicated artifact directory.

Success condition:

- Three `.pickle` files exist.
- Metrics are reproducible when rerun.

## Phase 2: Sanity Check Against Proposal

Objective:
Make sure the baseline setup actually matches the proposal design.

Checks:

1. Same equation family:
   - 2D compressible CFD
2. Same three regimes:
   - low Mach smooth
   - sonic transitional
   - inviscid shock-like
3. Same architecture:
   - official PDEBench FNO2d
4. Same rollout setup:
   - autoregressive, `initial_step=10`, `t_train=21`
5. Same train/validation protocol:
   - PDEBench uses the first `10%` of the single HDF5 shard as validation

Decision:

- If this baseline is only for comparison, keep the official hyperparameters.
- If the proposal later asks for fair FNO vs U-Net retraining on identical
  settings, retrain both models from scratch after the pretrained baseline
  evaluation is done.

## Phase 3: Save Raw Rollout Tensors

Objective:
Capture the actual prediction fields needed for FFT analysis.

Why:

The current `.pickle` output is too compressed. The proposal needs error
structure by space, time, channel, and frequency.

Required implementation:

Add a lightweight custom evaluation script, preferably separate from upstream
training code.

Recommended new file:

- `pdebench/models/eval_fno_2dcfd_rollout.py`

This script should:

1. Reuse the existing `FNODatasetSingle` loader.
2. Reuse the existing FNO model construction logic.
3. Load the official checkpoint.
4. Run autoregressive rollout on the validation split.
5. Save:
   - `pred`
   - `target`
   - `grid`
   - metadata such as regime name and checkpoint name

Recommended output format:

- `.npz` or `.pt`

Recommended artifact location:

- `artifacts/fno_rollouts/<regime>/rollout.pt`

Success condition:

- For each regime, there is one raw rollout file with prediction and target
  tensors aligned in shape.

## Phase 4: FFT Error Decomposition

Objective:
Implement the primary contribution from the proposal.

Recommended new file:

- `analysis/fno_fft_bands.py`

Input:

- saved rollout tensors from Phase 3

Core computation:

1. Compute error field:
   - `error = pred - target`
2. For each sample, time step, and channel:
   - compute 2D FFT over spatial dimensions only
3. Convert `(kx, ky)` to radial wavenumber
4. Bin power into annular bands
5. Compute per-band normalized RMSE or spectral energy error

Recommended band design:

Use fixed radial bins across all regimes, not per-regime quantiles.

Suggested starting point:

1. `low`: `0 <= k < k_max/6`
2. `mid`: `k_max/6 <= k < k_max/3`
3. `high`: `k >= k_max/3`

Reason:

- easier regime-to-regime comparison
- cleaner FNO vs U-Net comparison later

Outputs:

1. per-sample spectral metrics
2. per-regime averaged spectral metrics
3. per-channel spectral metrics
4. per-time-step spectral metrics for rollout stability analysis

Recommended artifact files:

- `artifacts/fno_fft/<regime>/band_metrics.csv`
- `artifacts/fno_fft/<regime>/radial_spectrum.npy`

## Phase 5: Figures For The Proposal

Objective:
Produce the figures directly implied by `Proposal_Plan.tex`.

Recommended figure set:

1. Baseline scalar table
   - one row per regime
   - columns for nRMSE and FFT low/mid/high error
2. Radial error spectrum plot for FNO
   - one curve per regime
3. Bandwise bar chart
   - low vs mid vs high for each regime
4. Rollout stability plot
   - error vs autoregressive step

If U-Net is added later:

5. FNO vs U-Net spectral comparison
6. FNO advantage ratio by band and regime

## Phase 6: Extend To U-Net

Objective:
Match the comparison setup in the proposal.

What can already be reused:

- same `train_models_forward.py`
- same `config_2DCFD.yaml` base with U-Net overrides
- same `metrics.py`
- same evaluation split logic

What still needs validation:

- official `2DCFD_Unet.tar` contents
- exact checkpoint naming conventions

Recommended sequencing:

1. Finish FNO baseline and FFT pipeline first.
2. Duplicate the rollout extraction path for U-Net.
3. Run the exact same FFT script on U-Net outputs.
4. Only then compute FNO advantage ratio.

## Phase 7: DoG Analysis

Objective:
Add the proposal's spatial localization component after the FFT pipeline is
stable.

Recommended new file:

- `analysis/dog_shock_localization.py`

Input:

- saved rollout tensors

Computation:

1. build Gaussian pyramid on target and error fields
2. compute DoG response maps
3. detect shock-like regions or steep-gradient regions
4. correlate local error magnitude with shock proximity

Outputs:

- regime-level correlation summaries
- example heatmaps

This phase is explicitly lower priority than the FFT pipeline.

## Concrete Directory Layout

Recommended project layout:

```text
PDEBench/
  Plan.md
  artifacts/
    fno_rollouts/
    fno_fft/
    tables/
    figures/
  analysis/
    fno_fft_bands.py
    plot_fno_fft.py
    dog_shock_localization.py
  pdebench/
    models/
      eval_fno_2dcfd_rollout.py
```

## Immediate Next Actions

1. Finish validating and extracting the official FNO checkpoints.
2. Create the no-`Rand_` HDF5 symlink aliases.
3. Run the three pretrained FNO eval jobs.
4. Add a raw-rollout export script for FNO.
5. Add the FFT-band analysis script.
6. Generate the first regime-by-band table and spectrum plot.

## Minimal Success Definition

The plan is successful when all of the following are true:

1. The official FNO checkpoint can be evaluated on all three proposal regimes.
2. We can reproduce a scalar baseline table for those three regimes.
3. We can save raw FNO predictions and targets for the validation split.
4. We can produce FFT low/mid/high error results per regime.
5. The output is organized so U-Net can be dropped in later without changing the
   FFT analysis code.

## Notes On Scope

This plan intentionally prioritizes:

1. official FNO baseline reuse
2. faithful regime selection
3. proposal-aligned FFT analysis

It does not assume we must immediately retrain every model from scratch. The
pretrained FNO is useful as a fast baseline, while scratch training can be added
as a later fairness check if needed for the final report.
