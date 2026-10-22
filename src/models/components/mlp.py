"""Small MLPs for 2D flow-matching and stochastic-flow-map experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import TimestepEmbedder
from .ssfm import Parameterization, SSFMNet


def _trunk(in_dim: int, hidden_size: int, depth: int) -> nn.Sequential:
    """Build a SiLU MLP trunk ending in a hidden activation."""
    if depth < 1:
        raise ValueError(f'depth must be at least 1; got {depth}.')
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_size), nn.SiLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
    return nn.Sequential(*layers)


class VectorMLP(nn.Module):
    """Single-time MLP velocity field ``v_theta(x, t)`` for 2D flow matching.

    Concatenates the state with a sinusoidal time embedding and maps to a
    ``dim``-vector. The call signature matches :class:`~src.models.components.dit.DiT`:
    ``forward(x, t)`` with ``x`` of shape ``(B, dim)`` and ``t`` of shape ``(B,)``.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_size: int = 256,
        depth: int = 4,
        time_emb_size: int = 128,
    ) -> None:
        """Initialize a ``VectorMLP``.

        :param dim: State dimensionality.
        :param hidden_size: Width of every hidden layer and of the time embedding.
        :param depth: Number of hidden SiLU layers.
        :param time_emb_size: Fourier width of the sinusoidal time features.
        """
        super().__init__()
        if dim < 1:
            raise ValueError(f'dim must be at least 1; got {dim}.')
        self.dim = dim
        self.time_embed = TimestepEmbedder(hidden_size, emb_size=time_emb_size)
        self.trunk = _trunk(dim + hidden_size, hidden_size, depth)
        self.head = nn.Linear(hidden_size, dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the field at ``x`` for times ``t``.

        :param x: States of shape ``(B, dim)``.
        :param t: Times of shape ``(B,)`` (or broadcastable to it).
        :return: Field of shape ``(B, dim)``.
        """
        t_emb = self.time_embed(t.reshape(-1))
        return self.head(self.trunk(torch.cat((x, t_emb), dim=-1)))


class SSFM_MLP(SSFMNet):
    """Two-time MLP flow-map backbone for vector-valued SSFM experiments.

    Conditions on start time ``s``, terminal time ``t``, and flattened Brownian
    coefficient features ``(B, N, dim)``. Coefficient degree is padded with zeros
    up to :attr:`max_degree` so a single linear width covers degree sweeps.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden_size: int = 256,
        depth: int = 4,
        time_emb_size: int = 128,
        max_degree: int = 4,
        *,
        parameterization: Parameterization = 'euler',
        normalize_coefficients: bool = True,
    ) -> None:
        """Initialize an ``SSFM_MLP``.

        :param dim: State dimensionality.
        :param hidden_size: Width of every hidden layer and of the time embeddings.
        :param depth: Number of hidden SiLU layers.
        :param time_emb_size: Fourier width of the sinusoidal time features.
        :param max_degree: Maximum Brownian polynomial degree ``N`` the net accepts.
        :param parameterization: ``'euler'`` or ``'euler_maruyama'``.
        :param normalize_coefficients: Whether callers should feed unit-variance
            coefficient features (stored for sampler/loss convenience).
        """
        super().__init__(
            parameterization=parameterization, normalize_coefficients=normalize_coefficients
        )
        if dim < 1:
            raise ValueError(f'dim must be at least 1; got {dim}.')
        if max_degree < 1:
            raise ValueError(f'max_degree must be at least 1; got {max_degree}.')
        self.dim = dim
        self.max_degree = max_degree
        self.time_embed = TimestepEmbedder(hidden_size, emb_size=time_emb_size)
        in_dim = dim + 2 * hidden_size + max_degree * dim
        self.trunk = _trunk(in_dim, hidden_size, depth)
        self.drift_head = nn.Linear(hidden_size, dim)
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)
        if parameterization == 'euler_maruyama':
            self.diff_head: nn.Linear | None = nn.Linear(hidden_size, dim)
            nn.init.zeros_(self.diff_head.weight)
            nn.init.zeros_(self.diff_head.bias)
        else:
            self.diff_head = None

    def diagonal_prediction(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluate the Euler velocity head at ``s = t`` with Brownian features silenced.

        :param x: States of shape ``(B, dim)``.
        :param t: Times broadcastable to ``(B,)``.
        :return: Diagonal velocity of shape ``(B, dim)``.
        """
        times = t.reshape(-1)
        coefficients = x.new_zeros(x.shape[0], 1, self.dim)
        self.set_brownian(None)
        try:
            return self.drift(x, times, times, coefficients)
        finally:
            self.set_brownian(None)

    def _condition(self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor) -> Tensor:
        """Assemble the MLP input from state, times, and Brownian features."""
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f'expected x of shape (B, {self.dim}); got {tuple(x.shape)}.')
        if coefficients.shape[0] != x.shape[0] or coefficients.shape[2:] != x.shape[1:]:
            raise ValueError(
                'coefficients must have shape (B, N, dim) matching x; '
                f'got {tuple(coefficients.shape)} vs x {tuple(x.shape)}.'
            )
        degree = coefficients.shape[1]
        if degree > self.max_degree:
            raise ValueError(f'coefficient degree {degree} exceeds max_degree {self.max_degree}.')
        flat = coefficients.reshape(x.shape[0], -1)
        pad = self.max_degree * self.dim - flat.shape[1]
        if pad:
            flat = F.pad(flat, (0, pad))
        s_emb = self.time_embed(s.reshape(-1))
        t_emb = self.time_embed(t.reshape(-1))
        return torch.cat((x, s_emb, t_emb, flat), dim=-1)

    def _field(
        self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Return the drift field and optional diffusion field."""
        hidden = self.trunk(self._condition(x, s, t, coefficients))
        drift = self.drift_head(hidden)
        diffusion = None if self.diff_head is None else self.diff_head(hidden)
        return drift, diffusion
