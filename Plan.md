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

## Validated Reading Of The Current Fourier Metric

This section records what the repo actually does today, based on the code
paths that are active during official evaluation.

### Where it is implemented

- `pdebench/models/metrics.py`
  - `metric_func()` computes the low/mid/high Fourier metric.
- `pdebench/models/fno/train.py`
  - evaluation calls `metrics(...)` and pickles only the returned aggregates.
- `pdebench/models/config/config_rdb.yaml`
  - Hydra keeps the run directory at `.` so checkpoint lookup still depends on
    the shell working directory.

### What happens numerically in 2D

From `pdebench/models/metrics.py`:

1. The first `initial_step` frames are dropped before any metric is computed.
2. Tensors are permuted to `[batch, channel, x, y, time]`.
3. A 2D FFT is taken over spatial axes only:
   - `pred_F = torch.fft.fftn(pred, dim=[2, 3])`
   - `target_F = torch.fft.fftn(target, dim=[2, 3])`
4. The squared complex coefficient error is formed:
   - `_err_F = torch.abs(pred_F - target_F) ** 2`
   - by linearity this is equivalent to the Fourier-domain energy of
     `pred - target`, but the code immediately aggregates it.
5. Only the first quadrant is binned:
   - loops over `i in range(nx // 2)`, `j in range(ny // 2)`
6. Each coefficient is assigned to a radial shell with
   - `it = floor(sqrt(i**2 + j**2))`
7. Shell energies are summed, not normalized by the number of coefficients in
   that shell.
8. Shells are collapsed into three fixed shell-index ranges:
   - `low = [0, 4)`
   - `mid = [4, 12)`
   - `high = [12, end)`
9. With `if_mean=True`, the result is averaged over channel and time, so the
   final Fourier output for one run is only three numbers.

### Practical consequence

- The current PDEBench FFT output is a coarse benchmark summary:
  - one low / one mid / one high scalar per run
- It is not yet a diagnostic tool:
  - no per-sample spectra
  - no per-channel spectra
  - no per-time-step spectra
  - no ground-truth spectrum
  - no FNO vs U-Net advantage curves
  - no location information

### Important implementation caution

- `metrics()` accumulates batch metrics inside `enumerate(val_loader)` and
  divides by `itot` at the end.
- Since `itot` is the last batch index rather than the batch count, this
  averaging is slightly off.
- For proposal analysis, reuse the official metric only as a baseline sanity
  check, not as the final spectral analysis.

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
through `filename`. Use this inside the GPU-allocated shell from the baseline
eval section. Example:

```bash
cd /gpfsnyu/scratch/wg2381/PDEBench/pdebench/models

python train_models_forward.py \
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

The working directory matters:

- `pdebench/models/fno/train.py` derives `model_name = <filename_stem>_FNO`
- it then loads `<model_name>.pt` and writes `<model_name>.pickle` in the
  current working directory
- `pdebench/models/config/config_rdb.yaml` sets `hydra.run.dir: .`, so Hydra
  does not relocate the run; the shell cwd is still the checkpoint cwd

Because GPU work must not run on the login node, use an interactive Slurm
allocation on the Shanghai `gpu` partition first.

Request the interactive GPU:

```bash
export TS=$(date +%Y%m%d-%H%M%S)
export ARTIFACT_ROOT=/gpfsnyu/scratch/wg2381/PDEBench/artifacts/baseline_eval/$TS
mkdir -p "$ARTIFACT_ROOT"

srun --account=sw5973 --partition=gpu --qos=normal \
  --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --pty bash
```

Then inside the allocated GPU shell:

```bash
set -euo pipefail
# activate the PDEBench env here if needed

cd /gpfsnyu/scratch/wg2381/PDEBench/pdebench/models

for REGIME in \
  2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5 \
  2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
  2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
do
  STEM="${REGIME%.hdf5}"
  OUT_DIR="$ARTIFACT_ROOT/$STEM"
  mkdir -p "$OUT_DIR"

  python train_models_forward.py \
    +args=config_2DCFD.yaml \
    ++args.model_name='FNO' \
    ++args.if_training=False \
    ++args.data_path='/gpfsnyu/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand' \
    ++args.filename="$REGIME"

  cp "${STEM}_FNO.pickle" "$OUT_DIR/"
done

python analyse_result_forward.py
cp Results.csv Results.pdf "$ARTIFACT_ROOT/"
```

Why this pattern is safer:

- respects the cluster rule that GPU jobs must run under Slurm, not on the
  login node
- keeps the cwd at `pdebench/models/`, where checkpoint lookup expects the
  `.pt` files
- stores timestamped baseline outputs in a structured artifact directory for
  later FFT comparison

### Expected outputs

For each regime, PDEBench will generate:

- `<stem>_FNO.pickle`

The timestamped artifact root will also store:

- `Results.csv`
- `Results.pdf`

That `.pickle` still stores only the aggregate metrics returned by
`metrics.py`; it is enough for a baseline table but not enough for the
proposal's FFT diagnostic analysis.

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

## Phase 0: Inventory And Naming - DONE

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

Since the proposal later asks for fair FNO vs U-Net retraining on identical
settings, retrain both models from scratch after the pretrained baseline
evaluation is done.

## Caveat: What The `2D_rand` Generator Actually Does

The earlier version of this note described the initialization as a hard square
cut-and-paste. The code is more nuanced.

Validated evidence from
`pdebench/data_gen/data_gen_NLE/utils.py:init_multi_HD_2DRand(...)`:

- the generator first builds full-domain random density / pressure / velocity
  fields from low-order sinusoidal modes
- it then randomly decides per sample whether to apply a windowed rectangular
  mask:
  - `cond = random.choice(..., p=[0.5, 0.5])`
- when the window is used, the mask is not a hard box
  - `x_win` and `y_win` are tanh windows
  - the transition width is `trns = 0.01`
  - the random field is multiplied by this mask
  - outside the mask, density and pressure are filled back with constant
    background values `d0` and `d0 * T0`
- the mask bounds `xL, xR, yL, yR` are random, so the support is rectangular
  and roughly central, but not fixed

Interpretation for our FFT work:

- the benchmark is not purely globally smooth turbulence-like data
- it also is not a literal hard square discontinuity in all samples
- instead, some samples contain steep geometry-imposed window edges at `t=0`,
  while others remain full-domain random fields
- these benchmark-induced edges can contribute mid/high-frequency ground-truth
  power before shock evolution

The goal is not to add one more benchmark number; it is to turn FFT into a
diagnostic that explains why FNO vs U-Net fail differently as the regime
changes.

Updated proposal-aligned FFT plan:

Step 1: establish what the benchmark spectrum actually looks like
- compute ground-truth radial power spectra for each regime, channel, and time
  step
- inspect `t=0` separately from later rollout times
- check whether high-band power comes from evolved dynamics, window edges, or
  both

Step 2: reproduce PDEBench's coarse FFT metric as a benchmark-only sanity check
- report the official low/mid/high numbers from the `.pickle` outputs
- do not over-interpret them, because they are fixed-shell aggregate
  statistics

Step 3: turn FFT into a diagnostic
- save raw rollouts
- compute radial spectra from `error = pred - target`
- keep per-sample, per-channel, and per-time outputs
- use normalized or physical wavenumber bins, not PDEBench's hard-coded shell
  cutoffs `4` and `12`

Step 4: compare error spectrum against ground-truth spectrum
- when ground-truth high-band content rises, which model degrades faster?
- is FNO failure more concentrated in higher normalized bands?
- is U-Net relatively better on localized sharp structures while weaker on
  global low-frequency organization?

Step 5: connect frequency failure to spatial localization
- FFT answers `which frequencies fail`
- DoG answers `where they fail`

Working hypothesis:

- smooth regime:
  - FNO should benefit from global spectral mixing and win mainly in low bands
- transitional regime:
  - FNO's advantage should narrow as sharper gradients push error toward
    mid/high bands
- hardest regime:
  - both models should degrade, but the signatures should differ
  - FNO should look more band-limited
  - U-Net should look more localization-limited

Claim discipline:

- do not equate high-frequency energy with shocks by default
- verify whether the ground-truth structure comes from solver-evolved fronts,
  initial-condition geometry, or both

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
2. Also compute ground-truth spectra:
   - `gt_fft = FFT(target)`
3. For each sample, time step, and channel:
   - compute 2D FFT over spatial dimensions only
4. Convert `(kx, ky)` to radial wavenumber
5. Bin power into annular bands
6. Compute per-band normalized RMSE or spectral energy error

Recommended band design:

Use fixed normalized radial bins across all regimes, not per-regime quantiles.

Do not reuse PDEBench's built-in cutoffs `iLow=4` and `iHigh=12` for proposal
analysis:

- those thresholds come from code defaults, not from physics
- they correspond to different relative scales on `128 x 128` and `512 x 512`
  grids
- they are appropriate only for reproducing the official coarse benchmark
  summary

Suggested starting point:

1. define normalized radius `rho = k / k_Nyquist`
2. `low`: `0 <= rho < 1/6`
3. `mid`: `1/6 <= rho < 1/3`
4. `high`: `1/3 <= rho <= 1`
5. also save the full radial curve before collapsing to 3 bands

Reason:

- easier regime-to-regime comparison
- comparable interpretation across `128` and `512` grids
- cleaner FNO vs U-Net comparison later

Outputs:

1. per-sample spectral metrics
2. per-regime averaged spectral metrics
3. per-channel spectral metrics
4. per-time-step spectral metrics for rollout stability analysis
5. ground-truth radial spectra for the same samples

Recommended artifact files:

- `artifacts/fno_fft/<regime>/band_metrics.csv`
- `artifacts/fno_fft/<regime>/error_radial_spectrum.npy`
- `artifacts/fno_fft/<regime>/gt_radial_spectrum.npy`

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
4. We can produce full radial spectra plus normalized low/mid/high FFT results
   per regime.
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
