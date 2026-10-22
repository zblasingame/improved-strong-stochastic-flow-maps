"""Importance-weight utilities for FKC particle systems.

FKC simulates a weighted particle system: each particle carries a log weight
``w`` updated by the potential ``g`` along its trajectory. Terminal (or
intermediate) expectations are self-normalized, so only differences of log
weights matter. These helpers keep that normalization stable via ``logsumexp``
and provide systematic resampling for the optional SMC step.
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalized_weights(log_weights: Tensor) -> Tensor:
    """Return the stable self-normalized weights ``softmax(log_weights)``.

    :param log_weights: Log weights of shape ``(n,)``.
    :return: Normalized weights of shape ``(n,)`` summing to one.
    """
    if log_weights.ndim != 1:
        raise ValueError(f'log_weights must be 1D of shape (n,); got {tuple(log_weights.shape)}.')
    return torch.softmax(log_weights, dim=-1)


def effective_sample_size(log_weights: Tensor) -> Tensor:
    """Return the Kish effective sample size ``1 / sum_i w_i**2``.

    :param log_weights: Log weights of shape ``(n,)``.
    :return: Scalar effective sample size in ``[1, n]``.
    """
    weights = normalized_weights(log_weights)
    return 1.0 / weights.square().sum()


def systematic_resample(
    log_weights: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Systematic resampling indices for the normalized weights.

    :param log_weights: Log weights of shape ``(n,)``.
    :param generator: Optional RNG for the single offset draw.
    :return: Indices of shape ``(n,)`` in ``[0, n)``.
    """
    weights = normalized_weights(log_weights)
    n = weights.shape[0]
    offset = torch.rand(1, device=weights.device, dtype=weights.dtype, generator=generator)
    positions = (torch.arange(n, device=weights.device, dtype=weights.dtype) + offset) / n
    cumulative = torch.cumsum(weights, dim=-1)
    cumulative[-1] = 1.0
    indices = torch.searchsorted(cumulative, positions)
    return indices.clamp_max(n - 1)
