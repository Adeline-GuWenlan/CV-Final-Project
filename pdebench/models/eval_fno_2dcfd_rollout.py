from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Support direct script execution from `pdebench/models`, where Python would
# otherwise miss the repo root needed for `import pdebench...`.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pdebench.models.fno.fno import FNO2d
from pdebench.models.fno.utils import FNODatasetSingle

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run autoregressive FNO rollout extraction for PDEBench 2D CFD "
            "and save raw predictions plus targets for offline FFT analysis."
        )
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="HDF5 regime filename, for example 2D_CFD_M0.1_Eta0.1_..._Train.hdf5",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Directory containing the 2D CFD HDF5 files.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory where rollout.pt will be written.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help=(
            "Optional explicit checkpoint path. If omitted, the script expects "
            "<stem>_FNO.pt in the current working directory."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--num_channels", type=int, default=4)
    parser.add_argument("--initial_step", type=int, default=10)
    parser.add_argument("--t_train", type=int, default=21)
    parser.add_argument("--reduced_resolution", type=int, default=2)
    parser.add_argument("--reduced_resolution_t", type=int, default=1)
    parser.add_argument("--reduced_batch", type=int, default=1)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio used by PDEBench's single-file loader.",
    )
    parser.add_argument(
        "--num_samples_max",
        type=int,
        default=-1,
        help=(
            "Maximum number of validation samples to save. "
            "Use -1 to save the full validation split."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cuda, cuda:0, or cpu.",
    )
    return parser


def derive_model_name(filename: str) -> str:
    return Path(filename).stem + "_FNO"


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(filename: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).resolve()

    default_path = Path.cwd() / f"{derive_model_name(filename)}.pt"
    if not default_path.exists():
        msg = (
            f"Checkpoint not found at {default_path}. "
            "Run this script from pdebench/models/ or pass --checkpoint_path."
        )
        raise FileNotFoundError(msg)
    return default_path.resolve()


def build_val_dataset(args: argparse.Namespace) -> FNODatasetSingle:
    loader_num_samples = -1
    if args.num_samples_max > 0:
        loader_num_samples = math.ceil(args.num_samples_max / args.test_ratio)

    dataset = FNODatasetSingle(
        args.filename,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=args.reduced_batch,
        initial_step=args.initial_step,
        if_test=True,
        test_ratio=args.test_ratio,
        num_samples_max=loader_num_samples,
        saved_folder=args.data_path,
    )

    if args.num_samples_max > 0 and len(dataset) > args.num_samples_max:
        dataset.data = dataset.data[: args.num_samples_max]

    if len(dataset) == 0:
        msg = (
            "Validation split is empty. "
            "Increase --num_samples_max or check the dataset split settings."
        )
        raise ValueError(msg)

    if dataset.data.ndim != 5 or dataset.grid.shape[-1] != 2:
        msg = (
            "This script only supports 2D CFD data with tensors shaped "
            "[batch, x, y, t, c]."
        )
        raise ValueError(msg)

    return dataset


def build_model(args: argparse.Namespace, device: torch.device) -> FNO2d:
    model = FNO2d(
        num_channels=args.num_channels,
        width=args.width,
        modes1=args.modes,
        modes2=args.modes,
        initial_step=args.initial_step,
    ).to(device)
    return model


def rollout_batch(
    model: FNO2d,
    xx: torch.Tensor,
    yy: torch.Tensor,
    grid: torch.Tensor,
    initial_step: int,
    t_train: int,
) -> torch.Tensor:
    pred = yy[..., :initial_step, :]
    inp_shape = list(xx.shape[:-2]) + [-1]

    for _ in range(initial_step, t_train):
        inp = xx.reshape(inp_shape)
        im = model(inp, grid)
        pred = torch.cat((pred, im), dim=-2)
        xx = torch.cat((xx[..., 1:, :], im), dim=-2)

    return pred


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.filename, args.checkpoint_path)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading validation split from %s", args.filename)
    dataset = build_val_dataset(args)
    full_target = dataset.data
    grid_single = dataset.grid
    t_train = min(args.t_train, full_target.shape[-2])
    target = full_target[..., :t_train, :]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    logger.info(
        "Loaded %d validation samples with reduced grid %sx%s and %d time steps",
        len(dataset),
        target.shape[1],
        target.shape[2],
        target.shape[3],
    )

    model = build_model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_cpu: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (xx, yy, grid) in enumerate(loader, start=1):
            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            grid = grid.to(device, non_blocking=True)

            pred = rollout_batch(model, xx, yy, grid, args.initial_step, t_train)
            preds_cpu.append(pred.cpu())
            logger.info("Finished batch %d/%d", batch_idx, len(loader))

    pred = torch.cat(preds_cpu, dim=0)
    if pred.shape != target.shape:
        msg = f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        raise RuntimeError(msg)

    dx = float(grid_single[1, 0, 0] - grid_single[0, 0, 0]) if grid_single.shape[0] > 1 else None
    dy = float(grid_single[0, 1, 1] - grid_single[0, 0, 1]) if grid_single.shape[1] > 1 else None

    payload = {
        "pred": pred,
        "target": target,
        "grid": grid_single,
        "meta": {
            "regime": Path(args.filename).stem,
            "filename": args.filename,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "model_name": derive_model_name(args.filename),
            "initial_step": args.initial_step,
            "t_train": int(t_train),
            "num_samples": int(pred.shape[0]),
            "num_channels": int(pred.shape[-1]),
            "channel_names": ["density", "pressure", "Vx", "Vy"],
            "spatial_shape": [int(pred.shape[1]), int(pred.shape[2])],
            "time_steps": int(pred.shape[3]),
            "reduced_resolution": args.reduced_resolution,
            "reduced_resolution_t": args.reduced_resolution_t,
            "reduced_batch": args.reduced_batch,
            "test_ratio": args.test_ratio,
            "dx": dx,
            "dy": dy,
            "Lx": 1.0,
            "Ly": 1.0,
            "device": str(device),
        },
    }

    rollout_path = out_dir / "rollout.pt"
    torch.save(payload, rollout_path)
    logger.info("Saved rollout tensor bundle to %s", rollout_path)


if __name__ == "__main__":
    main()
