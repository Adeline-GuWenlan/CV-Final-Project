"""
batch_inspect_seeds.py

Batch-plot key time frames for a range of seeds in the 2D diff-react dataset.
Saves one PNG per seed and prints per-seed statistics to answer:

  A. Are most samples flat?
  B. Do any samples show obvious large-scale structures?
  C. Is channel 1 nearly constant across time?

Usage:
    python3 batch_inspect_seeds.py
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "/gpfsnyu/scratch/wg2381/pdebench_data/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
OUT_DIR   = Path("/gpfsnyu/scratch/wg2381/PDEBench/pdebench/data_download/seed_snapshots")
SEEDS     = list(range(0, 6)) + [10, 20, 30] + list(range(12, 21))  # deduplicated below
SEEDS     = sorted(set(SEEDS))
TIMES     = [0, 10, 30, 60, 100]   # time-step indices (out of 101)

# Structure-detection threshold: std of channel-0 spatial field at t > 0
# (t=0 is always random noise with std≈1; we care about evolved structure)
STRUCTURE_STD_THRESH = 0.05   # spatial std at t>0 that counts as "structured"
CONSTANT_STD_THRESH  = 0.02   # channel-1 "nearly constant" threshold at t>0

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_seed(f, seed: int) -> np.ndarray:
    """Return data array [T, H, W, C] for given seed."""
    key = str(seed).zfill(4)
    if key not in f:
        return None
    return np.array(f[key]["data"], dtype=np.float32)


def analyse_seed(data: np.ndarray, seed: int) -> dict:
    """Return diagnostic stats for one seed."""
    ch0 = data[..., 0]   # [T, H, W]
    ch1 = data[..., 1]

    # Spatial std per time step
    ch0_spatial_std = ch0.reshape(len(ch0), -1).std(axis=1)   # [T]
    ch1_spatial_std = ch1.reshape(len(ch1), -1).std(axis=1)

    # Overall (across time) std of channel values
    ch0_total_std = ch0.std()
    ch1_total_std = ch1.std()

    # Max spatial std EXCLUDING t=0 (t=0 is always pure noise ~std 1.0)
    ch0_max_spatial_std     = ch0_spatial_std.max()       # includes t=0
    ch0_max_spatial_std_ev  = ch0_spatial_std[1:].max()   # evolved: t>0
    ch1_max_spatial_std     = ch1_spatial_std.max()
    ch1_max_spatial_std_ev  = ch1_spatial_std[1:].max()

    # Is channel 1 nearly constant at evolved times (t > 0)?
    ch1_is_constant = bool(ch1_max_spatial_std_ev < CONSTANT_STD_THRESH)

    # Does channel 0 show spatial structure at evolved times (t > 0)?
    ch0_has_structure = bool(ch0_max_spatial_std_ev > STRUCTURE_STD_THRESH)

    return {
        "seed": seed,
        "ch0_total_std": ch0_total_std,
        "ch1_total_std": ch1_total_std,
        "ch0_max_spatial_std": ch0_max_spatial_std,
        "ch0_max_spatial_std_ev": ch0_max_spatial_std_ev,
        "ch1_max_spatial_std": ch1_max_spatial_std,
        "ch1_max_spatial_std_ev": ch1_max_spatial_std_ev,
        "ch1_is_constant": ch1_is_constant,
        "ch0_has_structure": ch0_has_structure,
        "ch0_spatial_std_per_t": ch0_spatial_std.tolist(),
        "ch1_spatial_std_per_t": ch1_spatial_std.tolist(),
    }


def plot_seed(data: np.ndarray, seed: int, stats: dict, out_dir: Path):
    """Save a [|times| × 2] grid of snapshots for this seed."""
    T = len(TIMES)
    fig, axes = plt.subplots(T, 2, figsize=(8, 3 * T))

    ch0 = data[..., 0]
    ch1 = data[..., 1]
    vmin0, vmax0 = ch0.min(), ch0.max()
    vmin1, vmax1 = ch1.min(), ch1.max()

    for r, t in enumerate(TIMES):
        if t >= data.shape[0]:
            continue
        im0 = axes[r, 0].imshow(ch0[t], cmap="viridis", vmin=vmin0, vmax=vmax0,
                                  interpolation="nearest")
        axes[r, 0].set_title(f"seed={seed}  ch0  t={t}", fontsize=8)
        axes[r, 0].axis("off")
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(ch1[t], cmap="plasma", vmin=vmin1, vmax=vmax1,
                                  interpolation="nearest")
        axes[r, 1].set_title(f"seed={seed}  ch1  t={t}", fontsize=8)
        axes[r, 1].axis("off")
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

    # Annotation box
    tag_lines = [
        f"ch0 max spatial std (t>0) = {stats['ch0_max_spatial_std_ev']:.4f}  "
        f"{'[HAS STRUCTURE]' if stats['ch0_has_structure'] else '[flat]'}",
        f"ch1 max spatial std (t>0) = {stats['ch1_max_spatial_std_ev']:.6f}  "
        f"{'[CONSTANT]' if stats['ch1_is_constant'] else '[varies]'}",
    ]
    fig.text(0.5, 0.01, "\n".join(tag_lines), ha="center", va="bottom",
             fontsize=8, color="navy",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = out_dir / f"seed_{seed:04d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = []

    with h5py.File(DATA_PATH, "r") as f:
        total_seeds_in_file = len(f.keys())
        print(f"File contains {total_seeds_in_file} seeds total.\n")

        for seed in SEEDS:
            data = load_seed(f, seed)
            if data is None:
                print(f"  seed {seed:4d} — NOT FOUND in file, skipping.")
                continue

            stats = analyse_seed(data, seed)
            all_stats.append(stats)
            out = plot_seed(data, seed, stats, OUT_DIR)
            flag_struct = "*** HAS STRUCTURE ***" if stats["ch0_has_structure"] else "flat"
            flag_ch1    = "CH1~CONST" if stats["ch1_is_constant"] else "ch1 varies"
            print(
                f"  seed {seed:3d} | "
                f"ch0_ev_std={stats['ch0_max_spatial_std_ev']:.4f} {flag_struct:22s} | "
                f"ch1_ev_std={stats['ch1_max_spatial_std_ev']:.6f} {flag_ch1} | "
                f"saved → {out.name}"
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    n_total     = len(all_stats)
    n_structure = sum(s["ch0_has_structure"]  for s in all_stats)
    n_flat      = n_total - n_structure
    n_ch1_const = sum(s["ch1_is_constant"]    for s in all_stats)

    print(f"\nTotal seeds examined : {n_total}")
    print(f"\n[A] Flat (ch0 max spatial std t>0 < {STRUCTURE_STD_THRESH}) : "
          f"{n_flat}/{n_total} ({100*n_flat/n_total:.0f}%)")
    print(f"[A] Has evolved structure (t>0)                      : "
          f"{n_structure}/{n_total} ({100*n_structure/n_total:.0f}%)")

    structured_seeds = [s["seed"] for s in all_stats if s["ch0_has_structure"]]
    print(f"[B] Seeds with obvious large-scale structure         : {structured_seeds}")

    print(f"\n[C] ch1 nearly constant (spatial std < {CONSTANT_STD_THRESH:.0e}) : "
          f"{n_ch1_const}/{n_total} ({100*n_ch1_const/n_total:.0f}%)")

    # Rank seeds by evolved (t>0) structure strength
    ranked = sorted(all_stats, key=lambda s: s["ch0_max_spatial_std_ev"], reverse=True)
    print("\nTop 5 most structured seeds (ch0 max spatial std, t>0):")
    for s in ranked[:5]:
        print(f"  seed {s['seed']:3d}  ch0_ev_std={s['ch0_max_spatial_std_ev']:.4f}")

    print(f"\nAll plots saved to: {OUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
