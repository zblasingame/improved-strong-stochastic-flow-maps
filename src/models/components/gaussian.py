"""Standard-normal draws that stay outside mixed-precision autocast."""

from __future__ import annotations

import torch
from torch import Tensor


def randn(
    *size: int | tuple[int, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``N(0, 1)`` in float32, then cast to ``dtype``.

    Autocast would otherwise emit bf16/fp16 Gaussians whose quantization
    greys out generated images. The returned tensor matches ``dtype`` (or
    float32 when omitted) so callers can still run the rest of the step in AMP.
    """
    shape: tuple[int, ...]
    if len(size) == 1 and isinstance(size[0], tuple):
        shape = size[0]
    else:
        shape = tuple(int(dim) for dim in size)
    device_type = 'cpu'
    if device is not None:
        device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, enabled=False):
        noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if dtype is not None and noise.dtype != dtype:
        return noise.to(dtype)
    return noise


def randn_like(tensor: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
    """Sample ``N(0, 1)`` like ``tensor``, drawing the Gaussian in float32."""
    return randn(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype, generator=generator)
