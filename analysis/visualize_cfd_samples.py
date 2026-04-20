"""
Visualize a few samples from the 2D CFD HDF5 datasets.

For each of the three proposal regimes, plots a 4-channel snapshot
(density, pressure, Vx, Vy) at t=0 and t=T for 3 randomly chosen samples.

Usage:
    python analysis/visualize_cfd_samples.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/scratch/wg2381/pdebench_data/2D/CFD/2D_Train_Rand")
OUT_DIR  = Path("/scratch/wg2381/PDEBench/artifacts/figures/cfd_samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = {
    "M0.1_Eta0.1":   "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5",
    "M1.0_Eta0.01":  "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
    "M1.0_Eta0":     "2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
}

CHANNELS = ["density", "pressure", "Vx", "Vy"]
N_SAMPLES = 3
RNG = np.random.default_rng(42)


def plot_regime(regime_tag: str, hdf5_path: Path) -> None:
    print(f"  Opening {hdf5_path.name} ...")
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["density"].shape[0]
        n_t     = f["density"].shape[1]
        idx     = RNG.choice(n_total, size=N_SAMPLES, replace=False)
        idx.sort()

        t_steps = [0, n_t // 2, n_t - 1]   # first, mid, last snapshot
        t_coord = f["t-coordinate"][:]

        # load: (N_SAMPLES, n_t, H, W, 4)
        data = np.stack(
            [f[ch][idx] for ch in CHANNELS], axis=-1
        )  # shape: (N_SAMPLES, n_t, H, W, 4)

    n_rows = N_SAMPLES * len(CHANNELS)
    n_cols = len(t_steps)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 2.5),
        constrained_layout=True,
    )
    fig.suptitle(f"Regime: {regime_tag}   (samples {idx.tolist()})", fontsize=11)

    for si in range(N_SAMPLES):
        for ci, ch in enumerate(CHANNELS):
            row = si * len(CHANNELS) + ci
            vmin = data[si, :, :, :, ci].min()
            vmax = data[si, :, :, :, ci].max()
            for col, ti in enumerate(t_steps):
                ax = axes[row, col]
                im = ax.imshow(
                    data[si, ti, :, :, ci],
                    origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax,
                )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if row == 0:
                    ax.set_title(f"t={t_coord[ti]:.3f}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"s{idx[si]} {ch}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

    out_file = OUT_DIR / f"{regime_tag}_samples.png"
    fig.savefig(out_file, dpi=120)
    plt.close(fig)
    print(f"  Saved -> {out_file}")


def main() -> None:
    for tag, fname in REGIMES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue
        print(f"\nRegime: {tag}")
        plot_regime(tag, path)

    print("\nDone. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
