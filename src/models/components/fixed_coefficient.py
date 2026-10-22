"""Original-DMF-style fixed-coefficient additive conditioning modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def _linear_as(module: nn.Linear, x: Tensor) -> Tensor:
    """Apply ``module`` in ``x.dtype``, casting trainables on the fly.

    Trainable float32 layers may meet a lower-precision token stream when the
    frozen backbone runs in its storage dtype with autocast disabled; the cast
    keeps every op in one dtype while gradients flow back to the float32
    parameters. It is a no-op whenever the dtypes already match.
    """
    if module.weight.dtype != x.dtype or (
        module.bias is not None and module.bias.dtype != x.dtype
    ):
        weight = module.weight.to(x.dtype)
        bias = None if module.bias is None else module.bias.to(x.dtype)
        return F.linear(x, weight, bias)
    return F.linear(x, module.weight, module.bias)


class FixedCoefficientResidual(nn.Module):
    """Map image and coefficient token streams to an additive residual.

    Parameters
    ----------
    hidden_size : int
        Token width of the streams.
    degree : int, optional
        Number of coefficient token streams concatenated into the MLP input
        (``3`` reproduces the original SDMFT ``W, H, K`` design bit-for-bit).
    """

    def __init__(self, hidden_size: int, degree: int = 3) -> None:
        super().__init__()
        if degree < 1:
            raise ValueError(f'degree must be >= 1; got {degree}.')
        self.degree = degree
        streams = 1 + degree
        self.levy_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(streams * hidden_size, streams * hidden_size),
            nn.SiLU(),
            nn.Linear(streams * hidden_size, hidden_size),
        )
        # Match the original SDMFT initialization: stochastic conditioning is a
        # no-op until the final projection learns a useful residual.
        nn.init.zeros_(self.levy_net[-1].weight)
        nn.init.zeros_(self.levy_net[-1].bias)

    def forward(
        self,
        image_tokens: Tensor,
        coefficient_tokens: Tensor,
        s: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Return image tokens with an interval-scaled stochastic residual."""
        if coefficient_tokens.ndim != 4 or coefficient_tokens.shape[1] != self.degree:
            raise ValueError(
                f'coefficient_tokens must have shape (B, {self.degree}, L, D); '
                f'got {tuple(coefficient_tokens.shape)}.'
            )
        if image_tokens.shape[0] != coefficient_tokens.shape[0]:
            raise ValueError('image and coefficient token batches must match')
        if image_tokens.shape[1:] != coefficient_tokens.shape[2:]:
            raise ValueError('image and coefficient token sequences must match')
        streams = torch.cat((image_tokens.unsqueeze(1), coefficient_tokens), dim=1)
        residual = rearrange(streams, 'b streams l d -> b l (streams d)', streams=1 + self.degree)
        for module in self.levy_net:
            residual = (
                _linear_as(module, residual) if isinstance(module, nn.Linear) else module(residual)
            )
        step = (t.reshape(-1) - s.reshape(-1)).reshape(-1, 1, 1)
        return image_tokens + step.to(dtype=residual.dtype) * residual
