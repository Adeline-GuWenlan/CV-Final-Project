from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from pdebench.models.unet.unet import UNet2d
from pdebench.models.unet.utils import UNetDatasetSingle

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run autoregressive U-Net rollout extraction for PDEBench 2D CFD "
            "and save raw predictions plus targets for offline FFT analysis."
        )
    )
    parser.add_argument("--filename", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help=(
            "Optional explicit checkpoint path. If omitted, the script expects "
            "<stem>_Unet-PF-20.pt in the current working directory."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--out_channels", type=int, default=4)
    parser.add_argument("--initial_step", type=int, default=10)
    parser.add_argument("--t_train", type=int, default=21)
    parser.add_argument("--unroll_step", type=int, default=20)
    parser.add_argument("--reduced_resolution", type=int, default=2)
    parser.add_argument("--reduced_resolution_t", type=int, default=1)
    parser.add_argument("--reduced_batch", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--num_samples_max", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint_suffix",
        default="_Unet-PF-20",
        help="Suffix appended to the regime stem when auto-resolving the checkpoint.",
    )
    return parser


def derive_model_name(filename: str, suffix: str) -> str:
    return Path(filename).stem + suffix


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(filename: str, checkpoint_path: str | None,
                            suffix: str) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).resolve()
    default_path = Path.cwd() / f"{derive_model_name(filename, suffix)}.pt"
    if not default_path.exists():
        msg = (
            f"Checkpoint not found at {default_path}. "
            "Run from pdebench/models/ or pass --checkpoint_path."
        )
        raise FileNotFoundError(msg)
    return default_path.resolve()


def build_val_dataset(args: argparse.Namespace) -> UNetDatasetSingle:
    loader_num_samples = -1
    if args.num_samples_max > 0:
        loader_num_samples = math.ceil(args.num_samples_max / args.test_ratio)

    dataset = UNetDatasetSingle(
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
        msg = "Validation split is empty; check --num_samples_max / test_ratio."
        raise ValueError(msg)

    if dataset.data.ndim != 5:
        msg = "This script only supports 2D CFD data with tensors shaped [b,x,y,t,c]."
        raise ValueError(msg)

    return dataset


def build_model(args: argparse.Namespace, device: torch.device) -> UNet2d:
    model = UNet2d(args.in_channels * args.initial_step, args.out_channels).to(device)
    return model


def rollout_batch(model: UNet2d, xx: torch.Tensor, yy: torch.Tensor,
                  initial_step: int, t_train: int) -> torch.Tensor:
    """Mirror the rollout loop in metrics.py for mode='Unet'.

    xx: [b, X, Y, initial_step, c]
    yy: [b, X, Y, T, c]
    returns pred: [b, X, Y, t_train, c]
    """

    pred = yy[..., :initial_step, :]
    inp_shape = list(xx.shape)[:-2] + [-1]
    for _ in range(initial_step, t_train):
        inp = xx.reshape(inp_shape)                       # [b, X, Y, init*c]
        temp_shape = [0, -1] + list(range(1, len(inp.shape) - 1))
        inp = inp.permute(temp_shape)                     # [b, init*c, X, Y]
        permute_back = [0] + list(range(2, len(inp.shape))) + [1]
        im = model(inp).permute(permute_back).unsqueeze(-2)  # [b, X, Y, 1, c]
        pred = torch.cat((pred, im), dim=-2)
        xx = torch.cat((xx[..., 1:, :], im), dim=-2)
    return pred


def build_synthetic_grid(nx: int, ny: int) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, nx + 1)[:-1]
    ys = torch.linspace(0.0, 1.0, ny + 1)[:-1]
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.filename, args.checkpoint_path,
                                              args.checkpoint_suffix)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading validation split from %s", args.filename)
    dataset = build_val_dataset(args)
    full_target = dataset.data
    t_train = min(args.t_train, full_target.shape[-2])
    target = full_target[..., :t_train, :]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    nx, ny = target.shape[1], target.shape[2]
    logger.info("Loaded %d samples with grid %sx%s and %d time steps",
                len(dataset), nx, ny, target.shape[3])

    model = build_model(args, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds_cpu: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, (xx, yy) in enumerate(loader, start=1):
            xx = xx.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            pred = rollout_batch(model, xx, yy, args.initial_step, t_train)
            preds_cpu.append(pred.cpu())
            logger.info("Finished batch %d/%d", batch_idx, len(loader))

    pred = torch.cat(preds_cpu, dim=0)
    if pred.shape != target.shape:
        msg = f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        raise RuntimeError(msg)

    grid_single = build_synthetic_grid(nx, ny)

    payload = {
        "pred": pred,
        "target": target,
        "grid": grid_single,
        "meta": {
            "regime": Path(args.filename).stem,
            "filename": args.filename,
            "model_family": "UNet",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "model_name": derive_model_name(args.filename, args.checkpoint_suffix),
            "initial_step": args.initial_step,
            "t_train": int(t_train),
            "unroll_step": args.unroll_step,
            "num_samples": int(pred.shape[0]),
            "num_channels": int(pred.shape[-1]),
            "channel_names": ["density", "pressure", "Vx", "Vy"],
            "spatial_shape": [int(pred.shape[1]), int(pred.shape[2])],
            "time_steps": int(pred.shape[3]),
            "reduced_resolution": args.reduced_resolution,
            "reduced_resolution_t": args.reduced_resolution_t,
            "reduced_batch": args.reduced_batch,
            "test_ratio": args.test_ratio,
            "dx": 1.0 / nx,
            "dy": 1.0 / ny,
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
