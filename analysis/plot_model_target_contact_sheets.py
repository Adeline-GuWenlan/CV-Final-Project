from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_ROOTS = {
    "FNO": Path("/scratch/wg2381/PDEBench/artifacts/fno_rollouts/20260423-153221"),
    "UNET": Path("/scratch/wg2381/PDEBench/artifacts/unet_rollouts/20260423-160214"),
}

DEFAULT_OUT_ROOT = Path("/scratch/wg2381/PDEBench/artifacts/figures/model_target_contact")

CHANNEL_MAP = {
    "2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train": "density",
    "2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train": "pressure",
    "2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train": "pressure",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render target-vs-model contact sheets for selected PDEBench rollouts."
    )
    parser.add_argument("--model", choices=["FNO", "UNET"], required=True)
    parser.add_argument(
        "--selection_csv",
        type=Path,
        default=Path(
            "/scratch/wg2381/PDEBench/artifacts/figures/fno_unet_rollout_compare/selected_samples.csv"
        ),
    )
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def parse_selection_csv(path: Path) -> list[tuple[str, list[int], list[int]]]:
    rows: list[tuple[str, list[int], list[int]]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            regime = row["regime"]
            sample_indices = [int(x) for x in row["sample_indices"].split()]
            time_steps = [int(x) for x in row["time_steps"].split()]
            rows.append((regime, sample_indices, time_steps))
    return rows


def safe_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def render_sheet(
    regime: str,
    model_name: str,
    payload: dict,
    sample_indices: list[int],
    time_steps: list[int],
    out_path: Path,
) -> None:
    pred = payload["pred"]
    target = payload["target"]
    meta = payload.get("meta", {})
    channel_names = list(meta.get("channel_names", ["density", "pressure", "Vx", "Vy"]))
    channel_name = CHANNEL_MAP[regime]
    channel_idx = channel_names.index(channel_name)

    sample_tensor = torch.as_tensor(sample_indices, dtype=torch.long)
    time_tensor = torch.as_tensor(time_steps, dtype=torch.long)
    pred = pred.index_select(0, sample_tensor).index_select(3, time_tensor).float().numpy()
    target = target.index_select(0, sample_tensor).index_select(3, time_tensor).float().numpy()

    n_samples = len(sample_indices)
    n_steps = len(time_steps)

    fig, axes = plt.subplots(
        n_samples * 2,
        n_steps,
        figsize=(1.55 * n_steps, 1.35 * n_samples * 2),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"{regime}   channel={channel_name}   compare target vs {model_name}",
        fontsize=12,
    )

    for row_group, sample_idx in enumerate(sample_indices):
        ref = target[row_group, :, :, :, channel_idx]
        vmin = float(ref.min())
        vmax = float(ref.max())
        blocks = [("target", target[row_group]), (model_name, pred[row_group])]
        for block_offset, (label, block) in enumerate(blocks):
            row = row_group * 2 + block_offset
            for col, step in enumerate(time_steps):
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    model_key = args.model.upper()
    root = DEFAULT_ROOTS[model_key]
    model_label = "FNO" if model_key == "FNO" else "U-Net"

    for regime, sample_indices, time_steps in parse_selection_csv(args.selection_csv):
        rollout_path = root / regime / "rollout.pt"
        payload = safe_load(rollout_path)
        out_path = args.out_root / model_key.lower() / regime / "target_vs_pred.png"
        render_sheet(regime, model_label, payload, sample_indices, time_steps, out_path)


if __name__ == "__main__":
    main()
