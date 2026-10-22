"""Uncertainty weighting and binned diagnostics for the SSFM objective.

The *uncertainty trick* (Kendall et al., arXiv:1705.07115; as used in the
``stochastic-flow-map`` reference) reweights a per-sample loss by a learned,
time-dependent log-variance ``l_{s,t}``:

    weighted = exp(-l_{s,t}) * raw + l_{s,t}.

The log-variance is produced by a small MLP on a fixed random Fourier embedding
of the interval endpoints ``(s, t)``. The Fourier features are *non-trained*
buffers; only the final linear read-out is learned (jointly with the network).

Binned diagnostics (:func:`binned_mean`) aggregate per-sample quantities into
uniform bins over ``[0, 1]`` so the raw losses and ``l_{s,t}`` can be logged as a
function of ``s`` or ``h = t - s``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

#: Uniform bin edges over ``[0, 1]`` (10 bins) for binned diagnostics.
BIN_EDGES: Tensor = torch.linspace(0.0, 1.0, 11)


class UncertaintyMLP(nn.Module):
    """Learned per-sample log-variance over interval endpoints ``(s, t)``.

    A fixed random Fourier embedding of ``s`` and ``t`` is fed through a single
    trainable linear layer to produce a scalar ``l_{s,t}`` per sample.

    :param fourier_dim: Number of random Fourier features per endpoint.
    :param scale: Standard deviation of the random frequencies.
    """

    def __init__(self, fourier_dim: int = 256, scale: float = 1.0) -> None:
        super().__init__()
        self.fourier_dim = fourier_dim
        # Fixed (non-trained) random Fourier features for s and t.
        self.register_buffer('freqs_s', torch.randn(fourier_dim) * scale)
        self.register_buffer('phases_s', torch.rand(fourier_dim))
        self.register_buffer('freqs_t', torch.randn(fourier_dim) * scale)
        self.register_buffer('phases_t', torch.rand(fourier_dim))
        self.linear = nn.Linear(2 * fourier_dim, 1)

    def _features(self, x: Tensor, freqs: Tensor, phases: Tensor) -> Tensor:
        """Cosine Fourier features ``cos(2 pi (freqs x + phases))`` of shape ``(B, F)``."""
        proj = rearrange(x, 'b -> b 1') * freqs + phases
        return torch.cos(2.0 * torch.pi * proj)

    def forward(self, s: Tensor, t: Tensor) -> Tensor:
        """Return the per-sample log-variance ``l_{s,t}`` of shape ``(B,)``.

        :param s: Start times of shape ``(B,)``.
        :param t: Terminal times of shape ``(B,)``.
        :return: Scalar log-variance per sample.
        """
        feat_s = self._features(s, self.freqs_s, self.phases_s)
        feat_t = self._features(t, self.freqs_t, self.phases_t)
        feats = torch.cat([feat_s, feat_t], dim=-1)
        return self.linear(feats).squeeze(-1)


def apply_uncertainty(
    raw: Tensor,
    log_var: Tensor,
    form: str = 'log_eps',
    eps: float = 1e-8,
) -> Tensor:
    """Apply the uncertainty reweighting ``f(l, raw)`` per sample.

    Two forms are supported:

    - ``'log_eps'`` (default, numerically stable): ``log(exp(-l) * raw + eps) + l``.
      The gradient ``df/dl = eps / (exp(-l) * raw + eps)`` is always positive and
      vanishes as ``l -> -inf``, so the learned log-variance self-limits instead
      of diverging to ``-inf`` (which the plain form permits once ``raw < 1/e``,
      driving the total loss negative).

    - ``'reference'``: ``exp(-l) * raw + l`` — the plain form used by the
      ``stochastic-flow-map`` reference (Boffi et al.) and the DMF repo's
      ``SDMFLoss``. The minimum over ``l`` is ``1 + log(raw)`` (negative once
      ``raw < 1/e``); relies on a small, magnitude-preserving optimizer (RAdam /
      AdamW at low lr) and gradient clipping to keep ``l`` bounded. Use this form
      for faithful reference reproduction with AdamW lr=1e-4.

    :param raw: Per-sample raw losses of shape ``(B,)``.
    :param log_var: Per-sample log-variance ``l_{s,t}`` of shape ``(B,)``.
    :param form: ``'log_eps'`` (stable, default) or ``'reference'`` (plain).
    :param eps: Small constant inside the log for numerical stability
        (``'log_eps'`` form only).
    :return: Per-sample weighted losses of shape ``(B,)``.
    """
    if form == 'reference':
        return torch.exp(-log_var) * raw + log_var
    if form == 'log_eps':
        return torch.log(torch.exp(-log_var) * raw + eps) + log_var
    raise ValueError(
        f"Unknown uncertainty form {form!r}; expected 'log_eps' or 'reference'."
    )


def binned_mean(values: Tensor, keys: Tensor, edges: Tensor) -> Tensor:
    """Mean of ``values`` grouped into bins of ``keys`` defined by ``edges``.

    Empty bins return ``0``. Useful for logging per-sample losses as a function
    of ``s`` or ``h``.

    :param values: Per-sample quantities of shape ``(B,)``.
    :param keys: Per-sample bin keys of shape ``(B,)`` (e.g. ``s`` or ``h``).
    :param edges: Monotonic bin edges of shape ``(num_bins + 1,)``.
    :return: Per-bin means of shape ``(num_bins,)``.
    """
    num_bins = edges.numel() - 1
    edges = edges.to(keys.device)
    # Inner edges only; clamp indices into [0, num_bins - 1].
    idx = torch.bucketize(keys, edges[1:-1].contiguous())
    idx = idx.clamp(0, num_bins - 1)
    sums = torch.zeros(num_bins, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_bins, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, idx, values)
    counts.scatter_add_(0, idx, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)
