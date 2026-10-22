"""Shared utilities for discrete Align Your Flow SSFM objectives."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor



def per_sample_mse(residual: Tensor) -> Tensor:
    """Reduce squared residuals over every non-batch dimension."""
    return (residual**2).flatten(1).mean(dim=1)


@contextmanager
def deterministic_evaluation(net: nn.Module) -> Iterator[None]:
    """Disable stochastic training layers temporarily while preserving gradients."""
    was_training = net.training
    net.eval()
    try:
        yield
    finally:
        net.train(was_training)


def sample_delta(s: Tensor, t: Tensor, dt: float) -> Tensor:
    """Draw a small absolute step, keeping both decomposed intervals nonempty."""
    if dt <= 0.0:
        raise ValueError(f'dt must be > 0 for a discrete AYF loss; got {dt}.')
    draw = 0.5 * dt * (1.0 + torch.rand_like(s))
    return torch.minimum(draw, 0.5 * (t - s))


