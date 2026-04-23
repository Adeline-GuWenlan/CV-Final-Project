# FFT Comparison Design for PDEBench Course Project

This document designs the next-stage FFT-based comparison on top of the
baseline FNO pipeline already built in this repo, grounded in the code in
[metrics.py](pdebench/models/metrics.py) and in the project notes in
[Plan.md](Plan.md) and [Proposal_Plan.tex](Proposal_Plan.tex).

## 0. Precheck: What we already have, and what we need next

Before specifying the design, the following was inspected:

* baseline pickles exist at
  [artifacts/baseline_eval/20260420-214734/](artifacts/baseline_eval/20260420-214734/),
  one per regime, each containing the 6-tuple returned by
  `metric_func` in [metrics.py:164-306](pdebench/models/metrics.py#L164-L306):
  `(err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F)`.
  The first five entries are scalar `np.ndarray` of shape `()`; the last is a
  3-vector of shape `(3,)` containing `[fft_low, fft_mid, fft_high]`. Verified
  by loading
  `2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train_FNO.pickle`.
* a CSV rollup of these scalars is at
  [artifacts/baseline_eval/20260420-214734/baseline_summary.csv](artifacts/baseline_eval/20260420-214734/baseline_summary.csv).
* the pickles do **not** contain predicted or ground-truth fields. The tensor
  `pred`/`target` is constructed in
  [fno/train.py:155-181](pdebench/models/fno/train.py#L155-L181) inside
  `metrics()` and discarded after `metric_func` returns six aggregates.
* FFT band boundaries and the radial binning loop are hard-coded in
  [metrics.py:264-295](pdebench/models/metrics.py#L264-L295).

Reasoning about what is sufficient:

1. **Are the current saved outputs enough for FFT evaluation?**
   No. The existing pickles collapse each regime to 3 scalar band numbers per
   run (after averaging over channel and time). We cannot recover per-sample,
   per-channel, per-time, or full radial spectra from those scalars; the
   tensors we would need are never serialized. What we have is a scalar
   benchmark summary, not a diagnostic.

2. **Are new predictions needed, or can we reuse what we have?**
   We need one more pass over the validation split to serialize raw rollouts
   (`pred`, `target`, and ideally `grid` and regime metadata). Weights,
   checkpoints, and data loaders are already in place; this is a data
   serialization step, not retraining. Existing `.pt` checkpoints in
   [pdebench/models/](pdebench/models/) are the source of truth; we should
   not retrain to get FFT.

3. **Smallest next experiment.**
   Add a `eval_fno_2dcfd_rollout.py` script next to the FNO trainer that
   reuses the existing FNO dataset and model construction, runs the same
   autoregressive rollout as `metrics()`, and writes
   `{pred, target, grid, meta}` to `artifacts/fno_rollouts/<regime>/rollout.pt`
   for one regime. Then validate the FFT redesign by re-deriving the official
   low/mid/high numbers from that saved tensor and confirming they match the
   `.pickle` values within floating-point tolerance. Once that sanity check
   passes, run it for the remaining two regimes.

## 1. Objective

PDE prediction error is intrinsically scale-dependent. The proposal in
[Proposal_Plan.tex](Proposal_Plan.tex) claims that FNO and U-Net have
different inductive biases across spatial frequencies: FNO is globally
band-limited by its spectral convolution truncation; U-Net is locally
capped by its filter receptive field. Aggregate nRMSE cannot distinguish
these failure modes.

FFT-based evaluation serves two purposes:

1. **Diagnostic.** Decompose error energy across wavenumbers so we can
   attribute degradation to a specific band, and track how that distribution
   shifts as the flow regime moves from smooth to shock-dominated.
2. **Comparison.** Give FNO vs U-Net a like-for-like metric keyed on
   wavenumber instead of pixel position, so a model's advantage can be
   described as frequency-selective rather than as a single scalar.

The goal is a metric that is defensible (physically motivated binning,
resolution-invariant), interpretable (a curve rather than 3 numbers), and
feasible (runnable from the saved rollout tensors without retraining).

## 2. What the current implementation does

The current FFT metric lives in
[metric_func](pdebench/models/metrics.py#L164-L306). For 2D CFD:

1. **Input shape.** `pred, target` enter as `[b, x, y, t, c]`.
2. **Context trimming.** The first `initial_step=10` frames are dropped
   ([metrics.py:174-175](pdebench/models/metrics.py#L174-L175)), so the FFT is
   only computed on rollout steps `t = 10 ... 20` under `t_train=21`.
3. **Axis order.** Tensors are permuted to `[b, c, x, y, t]`
   ([metrics.py:180-182](pdebench/models/metrics.py#L180-L182)).
4. **Spatial 2D FFT.**
   `pred_F = torch.fft.fftn(pred, dim=[2,3])` and analogously for target
   ([metrics.py:265-266](pdebench/models/metrics.py#L265-L266)).
   Only spatial dims are transformed; time and channel are not.
5. **Error spectrum.** `_err_F = |pred_F - target_F| ** 2`. By linearity of
   the FFT this equals the power spectrum of the error field
   `e = pred - target`.
6. **First-quadrant radial binning.**
   ```python
   for i in range(nx // 2):
       for j in range(ny // 2):
           it = floor(sqrt(i**2 + j**2))
           err_F[:, :, it] += _err_F[:, :, i, j]
   ```
   ([metrics.py:270-275](pdebench/models/metrics.py#L270-L275)).
   Only `(i, j) ∈ [0, nx/2) × [0, ny/2)` is visited; the other three
   quadrants (which for a real input contain conjugate-symmetric copies of
   the first quadrant **and** the independent high-ky modes) are not
   included. Shells are summed, not averaged over bin multiplicity.
7. **Normalization.**
   `_err_F = sqrt(mean_over_batch(err_F)) / (nx*ny) * Lx * Ly`
   ([metrics.py:276](pdebench/models/metrics.py#L276)). With `Lx=Ly=1`, this
   divides by `nx*ny`. It does not normalize by ground-truth spectral power,
   nor by the number of modes per shell.
8. **Three-band collapse.**
   ```python
   err_F[:, 0] = mean(_err_F[:, :iLow])     # iLow=4
   err_F[:, 1] = mean(_err_F[:, iLow:iHigh]) # iHigh=12
   err_F[:, 2] = mean(_err_F[:, iHigh:])
   ```
   with defaults `iLow=4, iHigh=12` ([metrics.py:292-295](pdebench/models/metrics.py#L292-L295)).
9. **Aggregation.** With `if_mean=True` the result is averaged over channel
   and time ([metrics.py:304](pdebench/models/metrics.py#L304)), returning
   three scalars per run.
10. **Batch accumulation bug.** In `metrics()` at
    [metrics.py:460-465](pdebench/models/metrics.py#L460-L465), the running
    sum is divided by `itot`, which is the last `enumerate` index, not the
    number of batches. For `N` batches this is an `N/(N-1)` bias. Harmless
    for trend comparison, but not for reporting absolute numbers.

### Practical consequence

The baseline pickle for each regime stores only the three scalars
`[fft_low, fft_mid, fft_high]` in `err_F` (verified from
`baseline_summary.csv`: e.g. transitional regime = `[0.1104, 0.0402, 0.00720]`).
Everything else is lost: no per-sample, per-channel, per-time, per-shell, or
ground-truth-spectrum content survives. The metric is a benchmark table
entry, not a diagnostic tool.

## 3. Limitations of the current design

1. **Hard-coded shell cutoffs (`4, 12`) are grid-relative, not physics-relative.**
   On `128×128` (after `reduced_resolution=2`, effective `64×64`), Nyquist
   shell is `~32`, so bands cover roughly `[0, 1/8)`, `[1/8, 3/8)`, `[3/8, 1]`
   in normalized radius. On a native `512×512` grid the same cutoffs
   correspond to very different physical scales. Cross-regime comparison
   under these cutoffs confounds grid resolution with actual spectral
   content.

2. **Only 3 bands.** Collapsing a full `min(nx,ny)/2 ≈ 32` or `256`-sample
   radial spectrum into 3 numbers discards the shape of the spectrum.
   A model can have the same low/mid/high summary while failing at very
   different wavenumbers inside each band.

3. **First-quadrant-only binning is biased.** The loop touches only
   `i ∈ [0, nx/2), j ∈ [0, ny/2)`. For a real-valued spatial field, the
   second half of `kx` (i.e. `i ∈ [nx/2, nx)`) is the conjugate of the
   first half, but the second half of `ky` when `kx = 0` is **not** a
   duplicate — independent high-ky modes with `kx < nx/2, ky ≥ ny/2` are
   silently dropped. The metric therefore ignores roughly half of the
   distinct high-ky directions.

4. **Shell energy is summed, not averaged over mode count.** Different
   shells contain different numbers of `(i, j)` cells (shell `r=0` has 1
   mode; shell `r=20` has ~`2πr`). Summing rather than averaging means
   higher shells get bigger numbers just from having more modes, inflating
   high-band numbers relative to low-band, unrelated to actual error energy.

5. **No ground-truth spectrum.** The error spectrum in isolation does not
   tell us whether the model is missing a high-frequency feature that the
   ground truth actually has, or overshooting in a band where ground truth
   is near zero. The proposal specifically asks to compare `|error|^2`
   against `|target|^2` per band; the current code never computes the
   latter.

6. **No per-time-step curve.** Averaging over time hides whether error
   grows exponentially in rollout or sits flat. Rollout stability is one of
   the four evaluation axes in the proposal but is not derivable from the
   current FFT numbers.

7. **No per-channel breakdown in the saved output.** `metric_func` computes
   per-channel spectra, but `metrics()` returns after `mean(dim=0,-1)` so
   only a channel-averaged scalar survives. Density, velocity, and pressure
   are expected to have very different spectra (pressure in particular
   often has sharper shock signatures); collapsing them is a lost
   diagnostic.

8. **Radial isotropy assumption.** Axial structures (e.g. a front aligned
   with `x`) have anisotropic power in `(kx, ky)`. Radial shell averaging
   collapses this. For CFD in the proposal regimes this is not a hard
   failure (shocks can form in many orientations), but it is information
   we are throwing away.

9. **Amplitude-only.** `|pred_F − target_F|^2` in a fixed frame compares
   the real+imaginary parts of the Fourier coefficients, so it is
   phase-aware — but because we never separately report `||pred_F| −
   |target_F||` versus the full error, we cannot distinguish an amplitude
   error from a spatial translation error. A phase-shifted prediction
   would look just as bad as a blurred prediction in the current metric.

10. **No normalization against ground-truth spectrum.** The reported
    numbers are absolute RMSE in Fourier space. The proposal needs
    **relative** band error (fRMSE = `||e_band|| / ||target_band||`) to
    compare low-Mach smooth regimes (small target amplitudes at high k)
    with inviscid shocks (large target amplitudes at high k) on equal
    footing.

11. **`initial_step` context is dropped.** This is deliberate (the model
    sees `t=0..9` as conditioning), but it means the `t=0` target
    spectrum — which is exactly where the tanh-window initial condition
    lives, per
    [Plan.md:436-469](Plan.md) — is never looked at. The benchmark
    spectrum at `t=0` is needed to interpret high-band content.

12. **Averaging-bug in `metrics()`.** Division by `itot` instead of
    `itot + 1` produces a small but consistent bias.

13. **Averaging across channel+time hides correlated growth.** A
    high-frequency error that is small at `t=10` and large at `t=20` will
    look the same in the final scalar as a uniform mid-range error, which
    destroys the rollout-stability signal.

## 4. Does it satisfy the caveats in `Plan.md`?

[Plan.md](Plan.md) lists specific caveats under
"Updated proposal-aligned FFT plan" and "Phase 4: FFT Error Decomposition".
For each, the current code is:

| Caveat (from Plan.md) | Status | Why |
| --- | --- | --- |
| Do not reuse hard-coded shell cutoffs 4 and 12 for proposal analysis | **not handled** | `metric_func` uses defaults `iLow=4, iHigh=12` in every call path. |
| Use normalized radial bins comparable across 128 and 512 grids | **not handled** | Bins are in raw shell index. No `k/k_Nyquist` normalization. |
| Save the full radial curve before collapsing to 3 bands | **not handled** | The intermediate `_err_F[:, :min(nx,ny)//2]` exists locally in `metric_func` but is discarded; only the 3-band collapse is returned. |
| Compute ground-truth radial spectra per regime, channel, time | **not handled** | `target_F` is computed only to form `|pred_F − target_F|^2`. Its own power spectrum is never reduced or stored. |
| Inspect `t=0` separately to see whether high-band power is from evolved shocks or from the tanh-window mask | **not handled** | `initial_step` frames (including `t=0`) are sliced off before FFT. |
| Per-sample, per-channel, per-time outputs | **partially handled** | `metric_func(if_mean=False)` returns a per-time, per-channel tensor, but `metrics()` always calls it with `if_mean=True` and further averages across time. Per-sample is never preserved. |
| Report per-time-step spectral error for rollout stability | **not handled** | Time dimension is averaged out. |
| Normalize for cross-regime comparison | **not handled** | Output is absolute RMSE in Fourier space, not fRMSE normalized against `||target_F||`. |
| Compute FFT on error field, not on an aggregate | **already handled** | `|pred_F − target_F|^2 = |FFT(e)|^2` by linearity. |
| FFT on 2D spatial axes only, not on time | **already handled** | `fftn(..., dim=[2,3])` is spatial only. |
| First-quadrant binning is sufficient for a real-valued field | **partially handled / bias** | Conjugate symmetry would justify half-plane binning, but the current loop drops the independent `ky ∈ [ny/2, ny)` modes, not just the conjugate ones. |
| Diagnostic across regimes, channels, time | **not handled** | Output is one scalar triple per run. |
| Reusable by FNO vs U-Net | **partially handled** | Same `metrics()` is called from both pipelines, so it is consistent; but the output is too coarse to support the "FNO advantage ratio per band" claim in the proposal. |

Net assessment: the current FFT metric meets only the caveats that come for
free from using `torch.fft` (linearity, spatial-only transform). It fails
on every caveat that the proposal and `Plan.md` flag as important for
diagnostic interpretation.

## 5. Proposed redesign

Scope discipline: this is an evaluation redesign. The model code, training
loop, and checkpoints stay untouched. The new code reads saved rollout
tensors and computes spectra offline.

### 5.1 Minimum viable design

Goal: produce per-regime radial error and target spectra plus normalized
3-band fRMSE numbers from saved rollouts, within a weekend.

Keep from current design:

* 2D spatial FFT with `torch.fft.fftn(..., dim=(-2,-1))` on the spatial
  axes.
* Error = `pred − target`, and `FFT(error) = pred_F − target_F`.
* `metric_func(if_mean=False)` for per-channel, per-time retention.

Change immediately:

1. **Save raw rollouts.** Add
   [pdebench/models/eval_fno_2dcfd_rollout.py](pdebench/models/eval_fno_2dcfd_rollout.py)
   that reuses `FNODatasetSingle` and the FNO model builder from
   [fno/train.py](pdebench/models/fno/train.py#L1-L181), loads
   `<stem>_FNO.pt`, runs rollout over the full validation split, and
   writes `artifacts/fno_rollouts/<stem>/rollout.pt` with:
   ```
   {"pred": [N, X, Y, T, C] float32,
    "target": [N, X, Y, T, C] float32,
    "grid":   [X, Y, 2]      float32,
    "meta":   {"regime": stem, "initial_step": 10, "t_train": 21,
               "dx": 1/X, "dy": 1/Y, "Lx": 1.0, "Ly": 1.0}}
   ```
   `pred` must include `t ∈ [0, T)` where `t < initial_step` is the
   ground-truth warm-up (i.e. save pred as the tensor built inside
   `metrics()` at `fno/train.py` — it already has this shape).
2. **Radial spectrum helper.** Add
   [analysis/fft_spectrum.py](analysis/fft_spectrum.py) with:
   * `radial_bins(nx, ny, n_bins=None) -> (rho, bin_edges, bin_index_map)`
     where `rho = sqrt(kx^2 + ky^2) / k_Nyquist`, with `kx, ky` in
     `torch.fft.fftfreq` convention (so half-plane symmetry is handled
     correctly).
   * `radial_spectrum(field_2d) -> power_1d` that computes
     `|fft|^2` on the last two dims, averages over the full 2D plane per
     shell (not just the first quadrant), and divides by shell mode count.
   * `band_indices(rho, edges=(0, 1/6, 1/3, 1)) -> list[bool mask]` for
     the normalized bands recommended in
     [Plan.md:606-614](Plan.md#L606-L614).
3. **Offline FFT driver.** Add
   [analysis/fno_fft_bands.py](analysis/fno_fft_bands.py) that:
   * loads `rollout.pt`,
   * computes radial spectra for `error`, `target`, and `pred` per
     `(sample, channel, time, shell)`,
   * saves `error_radial_spectrum.npy`, `gt_radial_spectrum.npy`,
     `pred_radial_spectrum.npy` at
     `artifacts/fno_fft/<stem>/`,
   * saves `band_metrics.csv` with columns
     `sample, channel, time, band, err_power, target_power, fRMSE`.
4. **Normalization.** Default per-band metric is
   `fRMSE_band = sqrt(sum_shell_in_band(|FFT(e)|^2)) /
                 sqrt(sum_shell_in_band(|FFT(target)|^2) + eps)`,
   computed per `(sample, channel, time)` and then averaged over sample
   **last**. This is the per-band analogue of nRMSE and is
   grid-resolution invariant.
5. **Regression test against official numbers.** A short test that calls
   `metric_func` on a fixed small tensor and also calls the new radial
   spectrum helper with its cutoffs set to the official `(4, 12)`
   raw-shell bins on the same tensor, verifies agreement to the level
   allowed by the first-quadrant bias difference. This is a sanity check
   that the new pipeline reproduces the old one when configured to match.

Deliverables at this stage:

* three `rollout.pt` files,
* three `band_metrics.csv`,
* three `error_radial_spectrum.npy` and `gt_radial_spectrum.npy`,
* one summary figure per regime (log-log error spectrum curve with
  three vertical band markers).

### 5.2 Stronger comparison design

Goal: produce the diagnostics the proposal actually claims, suitable for
being the core result in the final report. Adds per-time, per-channel,
and cross-model analysis on top of 5.1.

1. **Per-time-step spectral error curves.** For each regime, plot
   `fRMSE_band(t)` for `t ∈ [initial_step, t_train)` on the same axes,
   one curve per band. This gives the rollout-stability diagnostic that
   the proposal's evaluation axis explicitly requires.
2. **Per-channel spectra.** Maintain the channel dimension in
   `band_metrics.csv` (density, vx, vy, pressure). Plot separate curves
   per channel in supplementary figures; keep the four-channel mean as
   the headline number.
3. **FNO vs U-Net advantage ratio.** Once U-Net rollouts are saved in the
   same `rollout.pt` schema under `artifacts/unet_rollouts/<stem>/`,
   compute
   `advantage(band, regime) = fRMSE_unet / fRMSE_fno`
   per-regime, per-band, and per-time. Plot a heatmap
   `band × regime → log2(advantage)` and a curve `advantage(t)` for each
   band. This is the main proposal claim, and the redesign supports it
   without new modeling.
4. **Ground-truth spectrum diagnostics.** Plot `|target_F|(k)` on a
   log-log axis per regime at `t=0` and `t=T-1`, separately. The `t=0`
   curve answers the window-mask question from
   [Plan.md:436-469](Plan.md#L436-L469): if `t=0` already has
   non-trivial high-k power, high-band error at later times cannot be
   attributed to shock evolution alone.
5. **Cumulative spectral error curve.** `C(k) = sum_{k' ≤ k}
   |FFT(e)(k')|^2 / sum_{k'} |FFT(target)(k')|^2`. This is a single
   monotone curve per regime whose shape tells the whole story: if it
   saturates at low k, error is low-band dominated; if it stays near
   zero until the Nyquist quarter and jumps, error is high-band
   dominated.
6. **Raw-shell cutoff reproducibility column.** Keep the official
   `(iLow=4, iHigh=12)` raw-shell numbers as one row in the final table
   so the baseline is reproducible against `baseline_summary.csv`. All
   primary analysis uses normalized bins.
7. **Batch accumulation fix.** The offline driver computes everything as
   stacked `numpy`/`torch` operations, so the `itot`-divisor bug in
   `metrics()` is not inherited.

### 5.3 Optional extensions

Only if time permits. None are required for the course project.

* **Anisotropic 2D spectrum.** Keep the full `|FFT|^2` map on
  `(kx, ky)` rather than collapsing to radius for one or two example
  samples per regime. Plot as a 2D heatmap to check whether the inviscid
  shock regime has directional structure that the radial view hides.
  Cheap to produce, strong to have in the appendix, weak as a headline
  metric.
* **Phase vs amplitude decomposition.** Report
  `amp_err = ||pred_F| − |target_F||` and
  `phase_err = |pred_F − target_F| − amp_err` separately, per band.
  Lets us separately localize "blurred" from "shifted" failure modes.
* **Windowing.** Add a Hann window before FFT for non-periodic samples.
  The PDEBench 2D CFD uses periodic BCs, so periodic FFT is already
  correct here; this matters only if we later look at non-periodic
  shards.
* **Log-spectrum diagnostic.** Plot `log(|target_F|^2)` and
  `log(|pred_F|^2)` on the same axes to expose spectral roll-off
  differences between regimes and models. This is the standard
  turbulence-spectrum view.
* **Spectral error at exact FNO modes.** Read the FNO `modes=12`
  hyperparameter from
  [config_2DCFD.yaml](pdebench/models/config/args/config_2DCFD.yaml#L21)
  and annotate the error spectrum with a dashed line at that
  wavenumber. This makes the "FNO is truncated at a specific k" story
  visually direct.

## 6. Recommended metrics

For each metric: what it is, what it measures, and why it is in the
design.

1. **Full radial error spectrum `E_err(k)`.**
   `E_err(k) = mean_shell_modes(|pred_F − target_F|^2)` over the 2D
   plane, averaged per-`(sample, channel, time)` then over sample.
   A 1D curve over all wavenumbers. Single richest diagnostic we can
   produce. Everything else is a reduction of this curve.
2. **Full radial target spectrum `E_tgt(k)`.**
   Same computation on `target`. Needed as a denominator for
   normalization and as a sanity check for whether high-k energy is
   actually present in the ground truth of each regime.
3. **Per-band fRMSE.**
   `fRMSE_band = sqrt(sum_k∈band E_err(k)) / sqrt(sum_k∈band E_tgt(k))`.
   Resolution-invariant, regime-comparable, and directly maps to the
   proposal's "FNO advantage ratio per band" language. Replaces the
   current 3-scalar output as the headline band metric.
4. **Cumulative spectral error `C(k)`.**
   `C(k) = sum_{k' ≤ k} E_err(k') / sum_{k'} E_tgt(k')`.
   Monotone, shape encodes the full story, works as a single curve in a
   comparison plot.
5. **Per-timestep band fRMSE `fRMSE_band(t)`.**
   Same as (3) but keeps the time axis. Directly supports the
   "rollout stability" evaluation axis in
   [Proposal_Plan.tex:101-110](Proposal_Plan.tex#L101-L110).
6. **Per-channel band fRMSE.** Same as (3) but keeps the channel axis.
   Supports the hypothesis that pressure fails before density or
   velocity near shocks.
7. **FNO advantage ratio.**
   `adv(band, regime) = fRMSE_band(unet) / fRMSE_band(fno)`.
   Reporting in `log2` makes the symmetry of "FNO better" vs
   "U-Net better" readable.
8. **Anisotropic directional spectrum (optional).**
   `E(kx, ky)` retained uncollapsed for 2–4 illustrative samples.
   Only needed if a regime shows strong directional structure in
   visual inspection.
9. **Log-spectrum diagnostic (optional).**
   `log10 E_tgt(k)` and `log10 E_pred(k)` per regime at representative
   times. Exposes FNO's spectral truncation cliff around
   `k ~ modes * Δk`.
10. **Reproducibility row: official low/mid/high.**
    Raw-shell (`iLow=4, iHigh=12`) numbers computed by the new pipeline
    from saved rollouts. Included in the results table to document that
    the pipeline reproduces the benchmark, not as a scientific claim.

## 7. Implementation plan

Phase A: enable offline FFT (minimum viable design).

1. **New file:**
   [pdebench/models/eval_fno_2dcfd_rollout.py](pdebench/models/eval_fno_2dcfd_rollout.py).
   * Reuse `FNODatasetSingle` from
     [fno/utils.py:161](pdebench/models/fno/utils.py#L161).
   * Reuse FNO model construction and checkpoint loading from
     [fno/train.py:155-159](pdebench/models/fno/train.py#L155-L159).
   * Copy the rollout loop from
     [metrics.py:398-412](pdebench/models/metrics.py#L398-L412) without
     the `metric_func` call; keep `pred, target, grid` as tensors.
   * Save `rollout.pt` under
     `artifacts/fno_rollouts/<stem>/rollout.pt`. Use `torch.save` with
     the dict schema in §5.1.
   * CLI args: `--filename`, `--data_path`, `--out_dir`,
     `--num_samples_max` (to support a small smoke run first).
2. **New file:** [analysis/fft_spectrum.py](analysis/fft_spectrum.py) with
   three helpers as specified in §5.1. Pure `torch`/`numpy`, no model
   dependencies. Unit tests with small fake tensors should verify that:
   * `radial_spectrum` of `cos(2π k₀ x)` puts all power in shell
     `round(k₀)`,
   * Parseval holds up to `Lx * Ly / (nx * ny)`,
   * the new implementation reproduces `metric_func`'s 3-band output
     when given raw-shell cutoffs `(4, 12)` on the same input. Run
     this test against one `rollout.pt` sample and compare against the
     matching row in `baseline_summary.csv`.
3. **New file:**
   [analysis/fno_fft_bands.py](analysis/fno_fft_bands.py).
   * loads `rollout.pt`,
   * computes error, target, and pred radial spectra,
   * writes `band_metrics.csv`,
     `error_radial_spectrum.npy`,
     `gt_radial_spectrum.npy`,
     `pred_radial_spectrum.npy`
     under `artifacts/fno_fft/<stem>/`.
4. **Slurm wrapper:**
   [analysis/sbatch_save_fno_rollout.sh](analysis/sbatch_save_fno_rollout.sh)
   mirroring the existing
   [analysis/sbatch_eval_fno_regime.sh](analysis/sbatch_eval_fno_regime.sh)
   but calling `eval_fno_2dcfd_rollout.py` instead.

Phase B: plots and tables (stronger comparison design).

5. **New file:** [analysis/plot_fno_fft.py](analysis/plot_fno_fft.py).
   Produces the figures in §8.
6. **New file:**
   [analysis/make_regime_table.py](analysis/make_regime_table.py) that
   emits `artifacts/tables/regime_band_fRMSE.csv` with one row per
   regime and columns for each normalized band, each time-averaged and
   final-time variant, plus the reproducibility "raw-shell" columns.

Phase C: cross-model (only after U-Net rollouts exist).

7. Duplicate (1) as `eval_unet_2dcfd_rollout.py` and run for the same
   three regimes. The analysis code in (3), (5), (6) is written
   model-agnostically, keyed on regime and model, and needs no change.
8. **New file:**
   [analysis/fno_vs_unet_advantage.py](analysis/fno_vs_unet_advantage.py)
   that reads both model's `band_metrics.csv` and writes the advantage
   table and heatmap.

Experiments to run:

* One smoke-run: `eval_fno_2dcfd_rollout.py --num_samples_max 4` on the
  smoothest regime, confirm the rollout tensor shape matches the
  expectation, run the regression test against the saved pickle's
  `fft_low/mid/high`.
* Full run: all three regimes, all validation samples, save rollouts
  and spectra.
* Plot figures in §8, assemble table in §6-10.

Outputs saved (directory layout; compatible with
[Plan.md:706-723](Plan.md#L706-L723)):

```
PDEBench/
  artifacts/
    fno_rollouts/<stem>/rollout.pt
    fno_fft/<stem>/band_metrics.csv
    fno_fft/<stem>/error_radial_spectrum.npy
    fno_fft/<stem>/gt_radial_spectrum.npy
    fno_fft/<stem>/pred_radial_spectrum.npy
    tables/regime_band_fRMSE.csv
    figures/*.pdf
  analysis/
    fft_spectrum.py
    fno_fft_bands.py
    plot_fno_fft.py
    make_regime_table.py
    fno_vs_unet_advantage.py
  pdebench/models/
    eval_fno_2dcfd_rollout.py
```

## 8. Recommended visualizations

Primary (ship these in the course report):

1. **Radial error spectrum per regime (log-log).**
   `E_err(k)` curve for all three regimes on a single axis, colored by
   regime. Normalized wavenumber `rho = k / k_Nyquist` on the x-axis so
   `128` and `512` grids overlay correctly.
2. **Ground-truth spectrum per regime at `t=0` and `t=T-1` (log-log).**
   Shows whether high-band target content is present at the initial
   condition (window mask) or appears over time (shocks).
3. **Cumulative spectral error `C(k)` per regime.**
   One monotone curve per regime on a linear y-axis.
4. **Per-time-step band fRMSE.**
   One subplot per regime, three curves (low, mid, high bands) vs
   rollout step.
5. **Regime × band heatmap of fRMSE** for FNO. After U-Net exists,
   FNO-advantage-ratio version of the same heatmap.

Secondary (supplementary, not primary):

6. **Raw-shell low/mid/high bar chart** as a reproducibility exhibit
   that matches `baseline_summary.csv`.
7. **Per-channel stacked bars** of band fRMSE, showing whether pressure
   or density dominates the high band.
8. **Sample-level comparison panels.** For 2 samples per regime:
   `target[:, :, -1, :]`, `pred[:, :, -1, :]`, `abs(error)[:, :, -1, :]`
   in the left column; their 2D `|FFT|^2` maps in the right column.
   Keeps the spatial-vs-spectral story attached to a concrete example.
9. **Optional:** anisotropic 2D `|FFT|^2` heatmap for one inviscid-shock
   sample if directional structure is visually present.

## 9. Project-scoped recommendation

For the course project, adopt exactly this:

**Core (required) FFT design:**

* Save raw FNO rollouts per regime (one pass, existing checkpoint).
* Normalized radial bins with edges
  `{0, 1/6, 1/3, 1}` in `k/k_Nyquist`, plus the full radial curve
  retained.
* Metrics: per-band fRMSE, full radial error spectrum, ground-truth
  spectrum, cumulative spectral error `C(k)`, per-time-step band fRMSE.
* Aggregation: keep per-sample, per-channel, per-time internally;
  report regime-level averages for headline tables.
* Figures 1, 2, 3, 4, 5 from §8.
* Reproducibility row in the results table using the official raw-shell
  `(4, 12)` cutoffs, computed by the new pipeline.

**Core FFT design applied to U-Net** once U-Net rollouts are saved;
produce the `log2` advantage-ratio heatmap as the primary comparison
figure. The proposal claim in
[Proposal_Plan.tex:89-91](Proposal_Plan.tex#L89-L91) that "FNO's
superiority is frequency-selective and regime-dependent" is either
supported or refuted by that one figure; plan the report around it.

**Appendix / optional analysis:**

* anisotropic 2D spectrum,
* phase vs amplitude decomposition,
* log-spectrum plot with FNO `modes=12` annotation,
* DoG localization (scope of Phase 7 in [Plan.md](Plan.md), separate
  from FFT).

**What to treat as supplementary only:**

* the raw-shell `(4, 12)` three-number summary. It is useful as
  a reproducibility check against
  [baseline_summary.csv](artifacts/baseline_eval/20260420-214734/baseline_summary.csv)
  and nothing more. Do not let it drive any scientific claim.

**What is out of scope for this evaluation redesign:**

* changing the model, the training loop, the loss, or the dataset.
  The FFT redesign is strictly an offline post-hoc evaluation on top of
  saved rollouts.
