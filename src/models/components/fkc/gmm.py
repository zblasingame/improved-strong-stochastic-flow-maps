"""Analytic 2D Gaussian mixtures for FKC toy experiments.

Project convention: ``t = 0`` is noise and ``t = 1`` is data. Each mixture
carries an :class:`AffineSchedule` (defaulting to the project's linear path
``alpha_t = t``, ``sigma_t = 1 - t``). An endpoint mixture

    ``q_1(x) = sum_k w_k N(x; m_k, C_k)``

therefore has the exact marginal

    ``q_t(x) = sum_k w_k N(x; alpha_t m_k, alpha_t**2 C_k + sigma_t**2 I)``,

whose density and score are available in closed form. This module provides
that mixture object plus an exact pairwise product of mixtures, which is the
building block for FKC product-of-experts and annealed targets.

All routines are pure tensor algebra. Buffers live in the dtype of the
constructing tensors, so float64 tests keep full precision; ``.to(device,
dtype)`` moves everything consistently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.schedule import AffineSchedule, LinearSchedule

_DIM = 2
_LOG_2PI = math.log(2.0 * math.pi)


def _as_tensor(value: Tensor | Sequence[object], dtype: torch.dtype) -> Tensor:
    """Convert a (possibly nested) sequence/tensor to a tensor of ``dtype``.

    Nested sequences may contain tensors (for example ``[mean_tensor]``); those
    are stacked rather than coerced through Python scalars.
    """
    if isinstance(value, Tensor):
        return value.to(dtype=dtype)
    try:
        return torch.as_tensor(value, dtype=dtype)
    except (TypeError, ValueError):
        return torch.stack([_as_tensor(item, dtype) for item in value])


def _infer_dtype(
    means: Tensor | Sequence[object],
    covariances: Tensor | Sequence[object],
    weights: Tensor | Sequence[object],
    dtype: torch.dtype | None,
) -> torch.dtype:
    """Choose an explicit dtype, else inherit from the first tensor input."""
    if dtype is not None:
        return dtype
    for value in (means, covariances, weights):
        if isinstance(value, Tensor):
            return value.dtype
    return torch.get_default_dtype()


def _eye(x: Tensor) -> Tensor:
    """Return a ``(2, 2)`` identity on ``x``'s device and dtype."""
    return torch.eye(_DIM, device=x.device, dtype=x.dtype)


def _broadcast_time(t: Tensor | float, batch: int, ref: Tensor) -> Tensor:
    """Coerce a scalar or ``(B,)`` time to ``(B,)`` on ``ref``'s device/dtype."""
    t_t = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)
    if t_t.ndim == 0:
        t_t = t_t.expand(batch)
    elif t_t.ndim != 1 or t_t.shape[0] != batch:
        raise ValueError(f'time must be scalar or of shape ({batch},); got {tuple(t_t.shape)}.')
    return t_t


def _component_log_prob(cov: Tensor, diff: Tensor) -> tuple[Tensor, Tensor]:
    """Gaussian log density and precision-times-difference for ``diff = x - m``.

    :param cov: Covariances ``(..., 2, 2)`` (symmetric positive definite).
    :param diff: Centered points ``(..., 2)``.
    :return: ``(log_prob, precision_diff)`` each of shape ``(...)`` / ``(..., 2)``.
    """
    lower = torch.linalg.cholesky(cov)
    solved = torch.linalg.solve_triangular(lower, diff.unsqueeze(-1), upper=False).squeeze(-1)
    quad = (solved**2).sum(-1)
    log_det = 2.0 * torch.log(torch.diagonal(lower, dim1=-2, dim2=-1)).sum(-1)
    log_prob = -0.5 * (quad + log_det + _DIM * _LOG_2PI)
    precision_diff = torch.linalg.solve(cov, diff.unsqueeze(-1)).squeeze(-1)
    return log_prob, precision_diff


class GaussianMixture2D(nn.Module):
    """A full-covariance 2D Gaussian mixture with analytic marginalization.

    Components are stored at the endpoint ``t = 1``. Evaluating at a physical
    time ``t`` uses the exact affine marginal

        ``q_t(x) = sum_k w_k N(x; alpha_t m_k, alpha_t**2 C_k + sigma_t**2 I)``

    for the supplied :class:`AffineSchedule` (defaulting to the project's linear
    path ``alpha_t = t``, ``sigma_t = 1 - t``), whose density and score are
    available in closed form. Weights are normalized on construction;
    covariances must be symmetric positive definite.
    """

    def __init__(
        self,
        means: Tensor | Sequence[Sequence[float]],
        covariances: Tensor | Sequence[Sequence[Sequence[float]]],
        weights: Tensor | Sequence[float],
        *,
        dtype: torch.dtype | None = None,
        schedule: AffineSchedule | None = None,
    ) -> None:
        """Initialize the mixture.

        :param means: Component means of shape ``(K, 2)``.
        :param covariances: Component covariances of shape ``(K, 2, 2)``.
        :param weights: Non-negative component weights of shape ``(K,)``.
        :param dtype: Optional cast for all tensors; inferred otherwise.
        :param schedule: Affine noise schedule used to marginalize to a physical
            time; defaults to the project's linear schedule.
        """
        super().__init__()
        dtype = _infer_dtype(means, covariances, weights, dtype)
        self.schedule: AffineSchedule = schedule or LinearSchedule()
        means_t = _as_tensor(means, dtype).reshape(-1, _DIM)
        covs_t = _as_tensor(covariances, dtype).reshape(-1, _DIM, _DIM)
        weights_t = _as_tensor(weights, dtype).reshape(-1)
        if means_t.shape[0] < 1:
            raise ValueError('a Gaussian mixture needs at least one component.')
        if covs_t.shape[0] != means_t.shape[0] or weights_t.shape[0] != means_t.shape[0]:
            raise ValueError(
                'means, covariances and weights must share the component count; got '
                f'{tuple(means_t.shape)}, {tuple(covs_t.shape)}, {tuple(weights_t.shape)}.'
            )
        if bool(torch.any(weights_t < 0.0)):
            raise ValueError('mixture weights must be non-negative.')
        total = weights_t.sum()
        if not bool(total > 0.0):
            raise ValueError('mixture weights must have positive total mass.')
        self._validate_covariances(covs_t)
        self.register_buffer('means', means_t)
        self.register_buffer('covariances', covs_t)
        self.register_buffer('weights', weights_t / total)

    @staticmethod
    def _validate_covariances(covs: Tensor) -> None:
        """Reject non-symmetric or non-positive-definite covariance matrices."""
        if not torch.allclose(covs, covs.transpose(-1, -2), atol=1e-6, rtol=1e-5):
            raise ValueError('covariances must be symmetric.')
        try:
            _ = torch.linalg.cholesky(covs)
        except RuntimeError as error:
            raise ValueError('covariances must be positive definite.') from error

    def marginal_parameters(self, t: float | Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return endpoint-parameterized marginals at a scalar time ``t``.

        :param t: Scalar physical time.
        :return: ``(means_t (K, 2), covariances_t (K, 2, 2), weights (K,))``.
        """
        t_t = torch.as_tensor(t, device=self.means.device, dtype=self.means.dtype)
        if t_t.ndim != 0:
            raise ValueError(f'marginal_parameters expects a scalar time; got {tuple(t_t.shape)}.')
        alpha_t = self.schedule.alpha(t_t)
        sigma_t = self.schedule.sigma(t_t)
        means_t = alpha_t * self.means
        covs_t = alpha_t * alpha_t * self.covariances + sigma_t * sigma_t * _eye(self.means)
        return means_t, covs_t, self.weights

    def _marginal_batch(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Vectorized marginals for ``(B,)`` times: ``(B, K, 2)`` and ``(B, K, 2, 2)``."""
        alpha_t = self.schedule.alpha(t)
        sigma_t = self.schedule.sigma(t)
        means_t = alpha_t[:, None, None] * self.means[None]
        eye = _eye(self.means)
        covs_t = (
            (alpha_t * alpha_t)[:, None, None, None] * self.covariances[None]
            + (sigma_t * sigma_t)[:, None, None, None] * eye
        )
        return means_t, covs_t

    def _log_components(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(log_component_density (B, K), centered (B, K, 2), cov (B, K, 2, 2))``."""
        means_t, covs_t = self._marginal_batch(t)
        diff = x[:, None, :] - means_t
        log_comp, _ = _component_log_prob(covs_t, diff)
        return log_comp, diff, covs_t

    def log_prob(self, x: Tensor, t: float | Tensor) -> Tensor:
        """Log density ``log q_t(x)`` for ``x`` of shape ``(B, 2)``.

        :param x: Evaluation points ``(B, 2)``.
        :param t: Scalar or ``(B,)`` physical time.
        :return: Log density of shape ``(B,)``.
        """
        if x.ndim != 2 or x.shape[-1] != _DIM:
            raise ValueError(f'expected x of shape (B, {_DIM}); got {tuple(x.shape)}.')
        t_b = _broadcast_time(t, x.shape[0], x)
        log_comp, _, _ = self._log_components(x, t_b)
        return torch.logsumexp(log_comp + torch.log(self.weights)[None], dim=-1)

    def score(self, x: Tensor, t: float | Tensor) -> Tensor:
        """Analytic score ``grad_x log q_t(x)`` for ``x`` of shape ``(B, 2)``.

        :param x: Evaluation points ``(B, 2)``.
        :param t: Scalar or ``(B,)`` physical time.
        :return: Score of shape ``(B, 2)``.
        """
        if x.ndim != 2 or x.shape[-1] != _DIM:
            raise ValueError(f'expected x of shape (B, {_DIM}); got {tuple(x.shape)}.')
        t_b = _broadcast_time(t, x.shape[0], x)
        log_comp, diff, covs = self._log_components(x, t_b)
        log_joint = log_comp + torch.log(self.weights)[None]
        log_norm = torch.logsumexp(log_joint, dim=-1)
        responsibilities = torch.exp(log_joint - log_norm[:, None])
        precision_diff = torch.linalg.solve(covs, diff.unsqueeze(-1)).squeeze(-1)
        return -(responsibilities[..., None] * precision_diff).sum(dim=1)

    def sample(
        self, n: int, t: float | Tensor = 1.0, *, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw ``n`` independent samples from the marginal at time ``t``.

        :param n: Number of samples.
        :param t: Scalar or ``(n,)`` physical time (one time per sample).
        :param generator: Optional RNG for reproducibility.
        :return: Samples of shape ``(n, 2)``.
        """
        if n < 1:
            raise ValueError(f'n must be positive; got {n}.')
        t_b = _broadcast_time(t, n, self.means)
        index = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        means = self.means[index]
        covs = self.covariances[index]
        alpha_t = self.schedule.alpha(t_b)
        sigma_t = self.schedule.sigma(t_b)
        marg_cov = (
            (alpha_t * alpha_t)[:, None, None] * covs
            + (sigma_t * sigma_t)[:, None, None] * _eye(self.means)
        )
        lower = torch.linalg.cholesky(marg_cov)
        eps = torch.randn(
            n, _DIM, device=self.means.device, dtype=self.means.dtype, generator=generator
        )
        return alpha_t[:, None] * means + (lower @ eps.unsqueeze(-1)).squeeze(-1)


def multiply_mixtures(a: GaussianMixture2D, b: GaussianMixture2D) -> GaussianMixture2D:
    """Exact product ``a * b`` renormalized to a Gaussian mixture.

    Uses the Gaussian product identity, including the overlap weight
    ``N(m_i; n_j, C_i + D_j)`` for every component pair (the cross terms that a
    naive per-component covariance shrink would miss).

    :param a: First mixture.
    :param b: Second mixture.
    :return: Normalized mixture with ``K_a * K_b`` components.
    """
    if a.means.dtype != b.means.dtype or a.means.device != b.means.device:
        raise ValueError('mixtures must share dtype and device.')
    eye = _eye(a.means)
    means_a = a.means[:, None, :]
    covs_a = a.covariances[:, None, :, :]
    means_b = b.means[None, :, :]
    covs_b = b.covariances[None, :, :, :]
    sum_cov = covs_a + covs_b
    log_overlap, _ = _component_log_prob(sum_cov, means_a - means_b)

    prec_a = torch.linalg.solve(covs_a, eye)
    prec_b = torch.linalg.solve(covs_b, eye)
    precision = prec_a + prec_b
    cov = torch.linalg.inv(precision)
    natural = torch.linalg.solve(covs_a, means_a.unsqueeze(-1)).squeeze(-1) + torch.linalg.solve(
        covs_b, means_b.unsqueeze(-1)
    ).squeeze(-1)
    shift = (cov @ natural.unsqueeze(-1)).squeeze(-1)
    log_weight = torch.log(a.weights)[:, None] + torch.log(b.weights)[None, :] + log_overlap
    log_weight = log_weight.reshape(-1)
    weight = torch.softmax(log_weight, dim=-1)
    return GaussianMixture2D(
        shift.reshape(-1, _DIM), cov.reshape(-1, _DIM, _DIM), weight, schedule=a.schedule
    )
