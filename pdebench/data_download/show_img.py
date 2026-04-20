import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path

path = "/gpfsnyu/scratch/wg2381/pdebench_data/2D/diffusion-reaction"
seed = 100

with h5py.File(Path(path) / "2D_diff-react_NA_NA.h5", "r") as f:
    data = np.array(f[str(seed).zfill(4)]["data"], dtype="f")

print(data.shape)  # [101, 128, 128, 2]

times = [0, 10, 30, 60, 100]

fig, axes = plt.subplots(len(times), 2, figsize=(8, 3 * len(times)))

vmin0, vmax0 = data[..., 0].min(), data[..., 0].max()
vmin1, vmax1 = data[..., 1].min(), data[..., 1].max()

for r, t in enumerate(times):
    axes[r, 0].imshow(data[t, ..., 0], cmap="viridis", vmin=vmin0, vmax=vmax0, interpolation="nearest")
    axes[r, 0].set_title(f"channel 0, t={t}")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(data[t, ..., 1], cmap="viridis", vmin=vmin1, vmax=vmax1, interpolation="nearest")
    axes[r, 1].set_title(f"channel 1, t={t}")
    axes[r, 1].axis("off")

plt.tight_layout()
plt.savefig("reacdiff_debug.png", dpi=200)