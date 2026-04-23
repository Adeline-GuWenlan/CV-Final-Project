"""Radial-FFT helpers for the FFT redesign in FFT_design.md section 5.1.

Pure torch/numpy with no model dependencies. Used by
`analysis/fno_fft_bands.py` and the regression test that reproduces
`metric_func`'s (iLow=4, iHigh=12) raw-shell 3-band output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RadialBinning:
    rho: torch.Tensor
    bin_edges: torch.Tensor
    bin_index_map: torch.Tensor
    shell_counts: torch.Tensor


def radial_bins(nx: int, ny: int, n_bins: int | None = None) -> RadialBinning:
    """Build a radial binning in normalized wavenumber rho = k / k_Nyquist.

    Uses `torch.fft.fftfreq` conventions so half-plane symmetry of a real
    field is respected (the full `[nx, ny]` FFT plane is binned, not just
    the first quadrant that the legacy `metric_func` used).
    """

    kx = torch.fft.fftfreq(nx) * nx
    ky = torch.fft.fftfreq(ny) * ny
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    radius = torch.sqrt(kx_grid**2 + ky_grid**2)

    k_nyquist = min(nx, ny) / 2.0
    if n_bins is None:
        n_bins = int(k_nyquist)

    bin_edges = torch.linspace(0.0, float(k_nyquist), n_bins + 1)
    bin_index_map = torch.bucketize(radius, bin_edges, right=False) - 1
    bin_index_map = bin_index_map.clamp(min=0, max=n_bins - 1)

    shell_counts = torch.bincount(bin_index_map.reshape(-1), minlength=n_bins).to(
        torch.float32
    )
    rho = (bin_edges[:-1] + bin_edges[1:]) * 0.5 / k_nyquist
    return RadialBinning(
        rho=rho,
        bin_edges=bin_edges,
        bin_index_map=bin_index_map,
        shell_counts=shell_counts,
    )


def radial_spectrum(field_2d: torch.Tensor, binning: RadialBinning) -> torch.Tensor:
    """Full-plane radial power spectrum, averaged over shell mode count.

    Works on arbitrary leading dims; the last two dims must be (X, Y).
    Returns a tensor shaped `[..., n_bins]`.
    """

    leading = field_2d.shape[:-2]
    nx, ny = field_2d.shape[-2], field_2d.shape[-1]
    assert binning.bin_index_map.shape == (nx, ny)

    fft = torch.fft.fftn(field_2d, dim=(-2, -1))
    power = fft.real**2 + fft.imag**2
    flat_power = power.reshape(-1, nx * ny)
    index = binning.bin_index_map.reshape(-1)
    n_bins = binning.shell_counts.numel()

    out = torch.zeros(flat_power.shape[0], n_bins, dtype=flat_power.dtype,
                      device=flat_power.device)
    out.index_add_(1, index.to(out.device), flat_power)

    counts = binning.shell_counts.to(out.device).clamp(min=1.0)
    out = out / counts
    return out.reshape(*leading, n_bins)


def radial_sum_spectrum(field_2d: torch.Tensor, binning: RadialBinning) -> torch.Tensor:
    """Full-plane radial power, summed within shells (no mode-count average).

    This is the band integral used to compute fRMSE numerators/denominators
    without double-dividing by shell counts.
    """

    leading = field_2d.shape[:-2]
    nx, ny = field_2d.shape[-2], field_2d.shape[-1]
    fft = torch.fft.fftn(field_2d, dim=(-2, -1))
    power = fft.real**2 + fft.imag**2
    flat_power = power.reshape(-1, nx * ny)
    index = binning.bin_index_map.reshape(-1).to(flat_power.device)
    n_bins = binning.shell_counts.numel()
    out = torch.zeros(flat_power.shape[0], n_bins, dtype=flat_power.dtype,
                      device=flat_power.device)
    out.index_add_(1, index, flat_power)
    return out.reshape(*leading, n_bins)


def band_masks(rho: torch.Tensor, edges=(0.0, 1.0 / 6.0, 1.0 / 3.0, 1.0)
               ) -> list[torch.Tensor]:
    """Boolean masks over shells for the normalized bands recommended in
    Plan.md (default low/mid/high = [0,1/6), [1/6,1/3), [1/3,1])."""

    masks: list[torch.Tensor] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (rho >= lo) & (rho <= hi)
        else:
            mask = (rho >= lo) & (rho < hi)
        masks.append(mask)
    return masks


def band_masks_raw_shell(n_bins: int, iLow: int = 4, iHigh: int = 12
                         ) -> list[torch.Tensor]:
    """Legacy reproducibility: raw-shell cutoffs (iLow=4, iHigh=12)."""

    idx = torch.arange(n_bins)
    return [idx < iLow, (idx >= iLow) & (idx < iHigh), idx >= iHigh]


def sanity_parseval(field: torch.Tensor) -> tuple[float, float]:
    """Return (spatial_energy, spectral_energy/Nxy) for a real 2D field.

    They should agree up to float tolerance if the transform is unitary
    under the 1/N normalization we use below.
    """

    nx, ny = field.shape[-2], field.shape[-1]
    fft = torch.fft.fftn(field, dim=(-2, -1))
    spatial_energy = float((field**2).sum())
    spectral_energy = float((fft.real**2 + fft.imag**2).sum()) / (nx * ny)
    return spatial_energy, spectral_energy


__all__ = [
    "RadialBinning",
    "band_masks",
    "band_masks_raw_shell",
    "radial_bins",
    "radial_spectrum",
    "radial_sum_spectrum",
    "sanity_parseval",
]
