"""Offline FFT driver for the FFT redesign in FFT_design.md section 5.1.

Loads a saved `rollout.pt`, computes radial error/target/pred power spectra,
writes `band_metrics.csv` plus three radial-spectrum `.npy` files, and emits
one summary figure per regime.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from analysis.fft_spectrum import (
    band_masks,
    band_masks_raw_shell,
    radial_bins,
    radial_sum_spectrum,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute radial FFT spectra and per-band fRMSE from a saved "
            "rollout.pt bundle."
        )
    )
    parser.add_argument("--rollout", required=True, help="Path to rollout.pt")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--model_label", default="Validation",
                        help="Label prefix used in the summary figure title.")
    parser.add_argument("--initial_step", type=int, default=None,
                        help="Override initial_step; defaults to meta.initial_step")
    parser.add_argument("--skip_initial", action="store_true",
                        help="Drop the first `initial_step` frames before FFT "
                             "(matches legacy metric_func).")
    parser.add_argument("--figure", default=None,
                        help="Output figure path; defaults to "
                             "<out_dir>/fft_error_spectrum.png")
    parser.add_argument("--eps", type=float, default=1e-30)
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Samples per FFT chunk (lower = less RAM).")
    return parser


def load_rollout(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for required in ("pred", "target", "meta"):
        if required not in payload:
            msg = f"rollout.pt at {path} missing '{required}'"
            raise KeyError(msg)
    return payload


def compute_spectra(pred: torch.Tensor, target: torch.Tensor,
                    chunk_size: int = 4):
    """Expect shape [N, X, Y, T, C]. Returns dict of numpy arrays shaped
    [N, C, T, n_bins] for err/target/pred shell-summed power, plus the
    RadialBinning object. Iterates over samples in chunks to limit RAM."""

    assert pred.shape == target.shape, (pred.shape, target.shape)
    n, x, y, t, c = pred.shape
    binning = radial_bins(x, y, n_bins=None)
    nbins = binning.shell_counts.numel()
    norm = (1.0 / (x * y)) ** 2

    err_all = np.empty((n, c, t, nbins), dtype=np.float32)
    tgt_all = np.empty((n, c, t, nbins), dtype=np.float32)
    pred_all = np.empty((n, c, t, nbins), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        pr = pred[start:end].permute(0, 4, 3, 1, 2).contiguous()
        tg = target[start:end].permute(0, 4, 3, 1, 2).contiguous()
        er = pr - tg
        err_all[start:end] = (radial_sum_spectrum(er, binning).numpy() * norm
                              ).astype(np.float32)
        tgt_all[start:end] = (radial_sum_spectrum(tg, binning).numpy() * norm
                              ).astype(np.float32)
        pred_all[start:end] = (radial_sum_spectrum(pr, binning).numpy() * norm
                               ).astype(np.float32)
        logger.info("FFT chunk %d-%d / %d", start, end, n)
    return {
        "err": err_all,
        "target": tgt_all,
        "pred": pred_all,
        "binning": binning,
    }


def per_band_frmse(err_spec: np.ndarray, tgt_spec: np.ndarray,
                   masks: list[torch.Tensor], eps: float) -> np.ndarray:
    """Returns array shaped [N, C, T, n_bands] of fRMSE per sample/chan/time."""

    out = np.empty(err_spec.shape[:-1] + (len(masks),), dtype=np.float32)
    for b, m in enumerate(masks):
        mask_np = m.cpu().numpy()
        num = err_spec[..., mask_np].sum(axis=-1)
        den = tgt_spec[..., mask_np].sum(axis=-1)
        out[..., b] = np.sqrt(num) / np.sqrt(den + eps)
    return out


def per_band_raw_rmse(err_spec: np.ndarray, masks: list[torch.Tensor]
                      ) -> np.ndarray:
    """Legacy raw-shell RMSE analogue: sqrt(mean over shells in band)."""

    out = np.empty(err_spec.shape[:-1] + (len(masks),), dtype=np.float32)
    for b, m in enumerate(masks):
        mask_np = m.cpu().numpy()
        region = err_spec[..., mask_np]
        out[..., b] = np.sqrt(region.mean(axis=-1))
    return out


def write_band_csv(path: Path, frmse: np.ndarray, channel_names: list[str],
                   initial_step: int, band_labels: list[str]) -> None:
    n, c, t, nb = frmse.shape
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "channel_idx", "channel_name", "time", "band",
                    "fRMSE"])
        for si in range(n):
            for ci in range(c):
                cname = channel_names[ci] if ci < len(channel_names) else f"c{ci}"
                for ti in range(t):
                    for bi in range(nb):
                        w.writerow([si, ci, cname, ti, band_labels[bi],
                                    float(frmse[si, ci, ti, bi])])


def make_figure(out_path: Path, regime: str, rho: np.ndarray,
                err_spec_mean: np.ndarray, tgt_spec_mean: np.ndarray,
                band_edges: tuple[float, ...], model_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    ax.loglog(rho, err_spec_mean + 1e-40, label=r"$E_{\mathrm{err}}(k)$",
              color="tab:red")
    ax.loglog(rho, tgt_spec_mean + 1e-40, label=r"$E_{\mathrm{target}}(k)$",
              color="tab:blue")
    for edge in band_edges[1:-1]:
        ax.axvline(edge, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$\rho = k / k_{\mathrm{Nyquist}}$")
    ax.set_ylabel("radial power (shell-summed, $|FFT/N|^2$)")
    ax.set_title(f"{model_label} FFT spectrum: {regime}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def regression_vs_metric_func(err_spec: np.ndarray, tgt_spec: np.ndarray,
                              nbins: int, nx: int, ny: int,
                              initial_step: int) -> dict:
    """Approximate reproduction of metric_func's 3-scalar output using the
    same raw-shell cutoffs. Used as a sanity check for the new pipeline.

    The legacy code scans only the first quadrant and sums (no mode-count
    divide); here we reuse the full-plane shell-sum and average over shells
    in the band. Numbers will not match exactly, but should stay within an
    order of magnitude — a true regression test would require first-quadrant
    binning which is intentionally not reproduced (it was a bug).
    """

    masks = band_masks_raw_shell(nbins, iLow=4, iHigh=12)
    err_rollout = err_spec[:, :, initial_step:, :]
    tgt_rollout = tgt_spec[:, :, initial_step:, :]
    raw = per_band_raw_rmse(err_rollout, masks)
    return {
        "raw_shell_low": float(raw[..., 0].mean()),
        "raw_shell_mid": float(raw[..., 1].mean()),
        "raw_shell_high": float(raw[..., 2].mean()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    rollout_path = Path(args.rollout).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_rollout(rollout_path)
    meta = payload["meta"]
    regime = meta.get("regime", rollout_path.parent.name)
    channel_names = meta.get("channel_names",
                             [f"c{i}" for i in range(payload["pred"].shape[-1])])
    initial_step = args.initial_step if args.initial_step is not None \
        else int(meta.get("initial_step", 10))

    logger.info("Loaded rollout %s (pred=%s)", rollout_path,
                tuple(payload["pred"].shape))

    spectra = compute_spectra(payload["pred"].float(), payload["target"].float(),
                              chunk_size=args.chunk_size)
    err_spec = spectra["err"]     # [N, C, T, Bin]
    tgt_spec = spectra["target"]
    pred_spec = spectra["pred"]
    binning = spectra["binning"]
    rho = binning.rho.cpu().numpy()

    if args.skip_initial:
        t_slice = slice(initial_step, err_spec.shape[2])
        err_spec = err_spec[:, :, t_slice, :]
        tgt_spec = tgt_spec[:, :, t_slice, :]
        pred_spec = pred_spec[:, :, t_slice, :]
        logger.info("Sliced off first %d (initial) steps; T=%d remains",
                    initial_step, err_spec.shape[2])

    np.save(out_dir / "error_radial_spectrum.npy", err_spec)
    np.save(out_dir / "gt_radial_spectrum.npy", tgt_spec)
    np.save(out_dir / "pred_radial_spectrum.npy", pred_spec)
    np.save(out_dir / "rho.npy", rho)

    band_edges_norm = (0.0, 1.0 / 6.0, 1.0 / 3.0, 1.0)
    masks = band_masks(binning.rho, edges=band_edges_norm)
    frmse = per_band_frmse(err_spec, tgt_spec, masks, args.eps)
    band_labels = ["low", "mid", "high"]
    write_band_csv(out_dir / "band_metrics.csv", frmse, channel_names,
                   initial_step, band_labels)

    mean_over_stc = lambda arr: arr.mean(axis=(0, 1, 2))
    err_mean = mean_over_stc(err_spec)
    tgt_mean = mean_over_stc(tgt_spec)

    figure_path = Path(args.figure).resolve() if args.figure is not None \
        else out_dir / "fft_error_spectrum.png"
    make_figure(figure_path, regime, rho, err_mean, tgt_mean,
                band_edges_norm, args.model_label)

    headline = {
        "regime": regime,
        "num_samples": int(err_spec.shape[0]),
        "num_channels": int(err_spec.shape[1]),
        "num_time_after_slice": int(err_spec.shape[2]),
        "nbins": int(err_spec.shape[3]),
        "frmse_low_mean": float(frmse[..., 0].mean()),
        "frmse_mid_mean": float(frmse[..., 1].mean()),
        "frmse_high_mean": float(frmse[..., 2].mean()),
    }
    reg_input_err = spectra["err"]
    reg_input_tgt = spectra["target"]
    nx = payload["pred"].shape[1]
    ny = payload["pred"].shape[2]
    reg = regression_vs_metric_func(reg_input_err, reg_input_tgt,
                                    nbins=err_spec.shape[3], nx=nx, ny=ny,
                                    initial_step=initial_step)
    headline.update(reg)

    with open(out_dir / "headline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(headline.keys()))
        w.writerow(list(headline.values()))

    logger.info("Wrote band_metrics.csv (%d rows)",
                err_spec.shape[0] * err_spec.shape[1] * err_spec.shape[2] * 3)
    logger.info("Wrote %s", out_dir / "band_metrics.csv")
    logger.info("Figure: %s", figure_path)
    logger.info("Headline: %s", headline)


if __name__ == "__main__":
    main()
