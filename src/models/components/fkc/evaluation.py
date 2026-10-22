"""Weighted evaluation utilities for FKC toy experiments.

The learned sampler returns weighted particles, so quality is measured two
ways: self-normalized weighted moments against the exact target moments, and
(unweighted) raw-particle moments as the proposal baseline. Coupled strong
errors compare the learned map against a fine solver of the SAME truncated
Wong-Zakai path, which is distinct from the high-resolution Ito reference.
"""

from __future__ import annotations


import torch
from torch import Tensor

from .weights import effective_sample_size


def spatial_rmse(estimate: Tensor, reference: Tensor) -> float:
    """Return the per-particle root-mean-squared spatial error."""
    return float((estimate - reference).square().sum(-1).mean().sqrt())


def centered_log_weight_rmse(estimate: Tensor, reference: Tensor) -> float:
    """Return the RMSE of log-weight increments after removing a global gauge.

    Log weights are only defined up to a time-dependent additive constant, so
    both clouds are centered before comparison.
    """
    centered = (estimate - estimate.mean()) - (reference - reference.mean())
    return float(centered.square().mean().sqrt())


def ess_fraction(log_weights: Tensor) -> float:
    """Return the effective sample size as a fraction of the particle count."""
    return float(effective_sample_size(log_weights)) / float(log_weights.shape[0])


# -- Skreta et al. (arXiv:2503.02819) sampling metrics, Appendix F.1 ----------


def energy_values(problem: object, x: Tensor, t: float = 1.0) -> Tensor:
    """Return the FKC energy ``-log pi_t(x)`` up to an additive constant.

    Only differences between two energy clouds matter, so the (unknown)
    normalizer of ``pi_t`` cancels.
    """
    # Chunked: at beta=3 the target has 40**3 = 64k components, so one batch
    # of 10k points would materialize several (B, 64k, 2) tensors (tens of GB).
    chunk = 256
    return torch.cat(
        [-problem.log_target(x[start : start + chunk], t) for start in range(0, x.shape[0], chunk)]
    )


def energy_wasserstein(energy_a: Tensor, energy_b: Tensor, *, order: int = 2) -> float:
    """Exact 1D Wasserstein distance between two energy distributions.

    :param energy_a: Reference energies ``(n,)``.
    :param energy_b: Generated energies ``(m,)``.
    :param order: ``1`` or ``2``.
    :return: ``Energy-W_order``.
    """
    if order not in (1, 2):
        raise ValueError(f'order must be 1 or 2; got {order}.')
    n = min(energy_a.numel(), energy_b.numel())
    if n < 1:
        raise ValueError('energy distributions must be non-empty.')
    quantiles = torch.linspace(0.0, 1.0, n, device=energy_a.device, dtype=energy_a.dtype)
    difference = torch.quantile(energy_a, quantiles) - torch.quantile(energy_b, quantiles)
    return float(difference.abs().pow(order).mean().pow(1.0 / order))


def _histogram2d(points: Tensor, bins: int, lower: Tensor, upper: Tensor) -> Tensor:
    """Return a ``(bins, bins)`` count histogram (torch has no portable 2D hist)."""
    span = (upper - lower).clamp_min(1e-12)
    scaled = (points - lower) / span
    index_x = (scaled[:, 0] * bins).floor().clamp(0, bins - 1).long()
    index_y = (scaled[:, 1] * bins).floor().clamp(0, bins - 1).long()
    flat = index_x * bins + index_y
    counts = torch.bincount(flat, minlength=bins * bins)
    return counts.reshape(bins, bins).to(points.dtype)


def total_variation(
    x: Tensor, y: Tensor, *, bins: int = 64, limits: tuple[Tensor, Tensor] | None = None
) -> float:
    """Empirical total variation between two 2D clouds on a shared grid.

    :param x: Reference points ``(n, 2)``.
    :param y: Generated points ``(m, 2)``.
    :param bins: Grid resolution per axis.
    :param limits: Optional ``(lower (2,), upper (2,))`` grid bounds.
    :return: Total variation distance in ``[0, 1]``.
    """
    if limits is None:
        pooled = torch.cat((x, y), dim=0)
        lower = pooled.min(0).values
        upper = pooled.max(0).values
        pad = 0.05 * (upper - lower).clamp_min(1e-6)
        limits = (lower - pad, upper + pad)
    lower, upper = limits
    hist_x = _histogram2d(x, bins, lower, upper)
    hist_y = _histogram2d(y, bins, lower, upper)
    p = hist_x / hist_x.sum().clamp_min(1.0)
    q = hist_y / hist_y.sum().clamp_min(1.0)
    return float(0.5 * (p - q).abs().sum())


