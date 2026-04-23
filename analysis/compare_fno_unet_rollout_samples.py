from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CHANNEL_NAMES = ["density", "pressure", "Vx", "Vy"]
DEFAULT_FNO_ROOT = Path(
    "/scratch/wg2381/PDEBench/artifacts/fno_rollouts/20260423-153221"
)
DEFAULT_UNET_ROOT = Path(
    "/scratch/wg2381/PDEBench/artifacts/unet_rollouts/20260423-160214"
)
DEFAULT_OUT_ROOT = Path(
    "/scratch/wg2381/PDEBench/artifacts/figures/fno_unet_rollout_compare"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select matching samples from FNO and U-Net rollout bundles, then "
            "save per-regime temporal RMSE curves and per-channel contact sheets."
        )
    )
    parser.add_argument("--fno_root", type=Path, default=DEFAULT_FNO_ROOT)
    parser.add_argument("--unet_root", type=Path, default=DEFAULT_UNET_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        help="Optional subset of regime directory names to render.",
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=CHANNEL_NAMES,
        help="Subset of channels to render in the contact sheets.",
    )
    parser.add_argument(
        "--skip_initial",
        action="store_true",
        help="Only render autoregressive rollout steps after initial_step.",
    )
    return parser


def safe_torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def pick_sample_indices(n_total: int, n_select: int) -> np.ndarray:
    n_select = max(1, min(n_select, n_total))
    idx = np.linspace(0, n_total - 1, n_select, dtype=int)
    return np.unique(idx)


def resolve_time_indices(meta: dict, total_steps: int, skip_initial: bool) -> np.ndarray:
    initial_step = int(meta.get("initial_step", 0))
    start = initial_step if skip_initial else 0
    return np.arange(start, total_steps, dtype=int)


def select_subset(
    path: Path,
    sample_indices: np.ndarray | None,
    time_indices: np.ndarray | None,
    include_target: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    payload = safe_torch_load(path)
    pred = payload["pred"]
    target = payload["target"]
    meta = dict(payload.get("meta", {}))

    if sample_indices is None:
        sample_indices = pick_sample_indices(pred.shape[0], 5)
    sample_tensor = torch.as_tensor(sample_indices, dtype=torch.long)

    if time_indices is None:
        time_indices = resolve_time_indices(meta, pred.shape[3], skip_initial=False)
    time_tensor = torch.as_tensor(time_indices, dtype=torch.long)

    pred_sel = pred.index_select(0, sample_tensor).index_select(3, time_tensor).float().numpy()
    result: dict[str, np.ndarray] = {"pred": pred_sel}

    if include_target:
        tgt_sel = (
            target.index_select(0, sample_tensor).index_select(3, time_tensor).float().numpy()
        )
        result["target"] = tgt_sel

    del payload, pred, target
    gc.collect()
    return result, meta


def inspect_rollout(path: Path) -> tuple[int, int, dict]:
    payload = safe_torch_load(path)
    pred = payload["pred"]
    meta = dict(payload.get("meta", {}))
    num_samples = int(pred.shape[0])
    total_steps = int(pred.shape[3])
    del payload, pred
    gc.collect()
    return num_samples, total_steps, meta


def temporal_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2, 4)))


def plot_temporal_rmse(
    regime: str,
    sample_indices: np.ndarray,
    step_numbers: np.ndarray,
    fno_rmse: np.ndarray,
    unet_rmse: np.ndarray,
    out_path: Path,
) -> None:
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(10, 2.2 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, sample_idx, fno_curve, unet_curve in zip(
        axes, sample_indices, fno_rmse, unet_rmse
    ):
        ax.plot(step_numbers, fno_curve, marker="o", linewidth=1.6, label="FNO")
        ax.plot(step_numbers, unet_curve, marker="s", linewidth=1.6, label="U-Net")
        ax.set_ylabel(f"s{sample_idx}\nRMSE")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("time step")
    fig.suptitle(
        f"FNO vs U-Net rollout RMSE per time step\n{regime}   samples={sample_indices.tolist()}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_channel_contact_sheet(
    regime: str,
    channel_name: str,
    channel_idx: int,
    sample_indices: np.ndarray,
    step_numbers: np.ndarray,
    target: np.ndarray,
    fno_pred: np.ndarray,
    unet_pred: np.ndarray,
    out_path: Path,
) -> None:
    n_samples = len(sample_indices)
    n_steps = len(step_numbers)
    fig, axes = plt.subplots(
        n_samples * 3,
        n_steps,
        figsize=(1.55 * n_steps, 1.25 * n_samples * 3),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"{regime}   channel={channel_name}   samples={sample_indices.tolist()}",
        fontsize=12,
    )

    for row_group, sample_idx in enumerate(sample_indices):
        ref = target[row_group, :, :, :, channel_idx]
        vmin = float(ref.min())
        vmax = float(ref.max())
        image_blocks = [
            ("target", target[row_group]),
            ("FNO", fno_pred[row_group]),
            ("U-Net", unet_pred[row_group]),
        ]
        for block_offset, (label, block) in enumerate(image_blocks):
            row = row_group * 3 + block_offset
            for col, step in enumerate(step_numbers):
                ax = axes[row, col]
                ax.imshow(
                    block[:, :, col, channel_idx],
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    ax.set_title(f"t={step}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"s{sample_idx}\n{label}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def common_regimes(fno_root: Path, unet_root: Path) -> list[str]:
    fno = {path.name for path in fno_root.iterdir() if path.is_dir()}
    unet = {path.name for path in unet_root.iterdir() if path.is_dir()}
    return sorted(fno & unet)


def channel_index_map(meta: dict) -> dict[str, int]:
    channel_names = meta.get("channel_names", CHANNEL_NAMES)
    return {str(name): idx for idx, name in enumerate(channel_names)}


def write_selection_summary(summary_rows: list[dict[str, str]], out_root: Path) -> None:
    out_path = out_root / "selected_samples.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "sample_indices", "time_steps"])
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = build_parser().parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    regimes = common_regimes(args.fno_root, args.unet_root)
    if args.regimes:
        regimes = [regime for regime in regimes if regime in set(args.regimes)]
    if not regimes:
        raise ValueError("No common regime directories found for FNO/U-Net rollouts.")

    summary_rows: list[dict[str, str]] = []

    for regime in regimes:
        regime_out = args.out_root / regime
        regime_out.mkdir(parents=True, exist_ok=True)

        fno_rollout = args.fno_root / regime / "rollout.pt"
        unet_rollout = args.unet_root / regime / "rollout.pt"

        num_samples, total_steps, fno_meta = inspect_rollout(fno_rollout)
        sample_indices = pick_sample_indices(num_samples, args.n_samples)
        time_indices = resolve_time_indices(
            fno_meta,
            total_steps=int(fno_meta.get("time_steps", total_steps)),
            skip_initial=args.skip_initial,
        )

        fno_subset, fno_meta = select_subset(
            fno_rollout,
            sample_indices=sample_indices,
            time_indices=time_indices,
            include_target=True,
        )
        unet_subset, unet_meta = select_subset(
            unet_rollout,
            sample_indices=sample_indices,
            time_indices=time_indices,
            include_target=False,
        )

        if fno_subset["pred"].shape != unet_subset["pred"].shape:
            raise RuntimeError(
                f"Prediction shapes do not match for {regime}: "
                f"{fno_subset['pred'].shape} vs {unet_subset['pred'].shape}"
            )

        target = fno_subset["target"]
        fno_pred = fno_subset["pred"]
        unet_pred = unet_subset["pred"]
        step_numbers = time_indices

        plot_temporal_rmse(
            regime=regime,
            sample_indices=sample_indices,
            step_numbers=step_numbers,
            fno_rmse=temporal_rmse(fno_pred, target),
            unet_rmse=temporal_rmse(unet_pred, target),
            out_path=regime_out / "rmse_5samples_per_timestep.png",
        )

        channel_map = channel_index_map(fno_meta)
        for channel_name in args.channels:
            if channel_name not in channel_map:
                raise KeyError(f"Channel '{channel_name}' not found in rollout metadata.")
            plot_channel_contact_sheet(
                regime=regime,
                channel_name=channel_name,
                channel_idx=channel_map[channel_name],
                sample_indices=sample_indices,
                step_numbers=step_numbers,
                target=target,
                fno_pred=fno_pred,
                unet_pred=unet_pred,
                out_path=regime_out / f"contact_sheet_{channel_name}.png",
            )

        summary_rows.append(
            {
                "regime": regime,
                "sample_indices": " ".join(str(idx) for idx in sample_indices.tolist()),
                "time_steps": " ".join(str(step) for step in step_numbers.tolist()),
            }
        )

        del fno_subset, unet_subset, target, fno_pred, unet_pred
        gc.collect()

    write_selection_summary(summary_rows, args.out_root)


if __name__ == "__main__":
    main()