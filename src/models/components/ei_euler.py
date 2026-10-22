"""Increment-conditioned exponential-integrator Euler flow maps.

The map is
``Phi_st(x) = (alpha_s / alpha_t) x + (2 A_st / alpha_t) u_st
               + (1 / alpha_t) int_s^t alpha_r nu_r dW_r``.
It uses the physical flow-as-SDE martingale and is separate from the x-pred
SNR-time integral map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .gaussian import randn
from .schedule import AffineSchedule
from .ssfm import SSFMNet


@dataclass(frozen=True)
class EIEulerPath:
    """Polynomial representation of a scaled local EI martingale increment."""

    coeffs: Tensor
    normalized_coeffs: Tensor
    variance: Tensor
    degrees: Tensor | None = None

    @property
    def degree(self) -> int:
        """Return the padded polynomial degree."""
        return self.coeffs.shape[0]

    def degree_mask(self) -> Tensor:
        """Return active modes for each batch member."""
        batch = self.coeffs.shape[1]
        if self.degrees is None:
            return torch.ones(batch, self.degree, device=self.coeffs.device, dtype=torch.bool)
        modes = torch.arange(self.degree, device=self.coeffs.device)
        return modes.unsqueeze(0) < self.degrees.unsqueeze(1)

    def terminal_increment(self) -> Tensor:
        """Return ``alpha_t**-1 int_s^t nu_r dW_r``."""
        return self.coeffs[0]

    def coefficient_features(self, *, normalize: bool = True) -> Tensor:
        """Return martingale polynomial features in ``(B, N, ...)`` layout."""
        features = self.normalized_coeffs if normalize else self.coeffs
        if self.degrees is not None:
            mask = (
                self.degree_mask()
                .transpose(0, 1)
                .reshape(self.degree, features.shape[1], *([1] * (features.ndim - 2)))
            )
            features = features * mask.to(dtype=features.dtype)
        return features.movedim(0, 1)


def sample_ei_euler_path(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    s: Tensor | float,
    t: Tensor | float,
    degree: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    degrees: Tensor | None = None,
    t0: float = 0.0,
) -> EIEulerPath:
    """Sample local scaled martingale polynomial coefficients."""
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    s_t = torch.as_tensor(s, device=device, dtype=dtype)
    t_t = torch.as_tensor(t, device=device, dtype=dtype)
    if torch.any(s_t <= 0):
        raise ValueError('EI-Euler intervals require s > 0.')
    if torch.any(t_t < s_t):
        raise ValueError('EI-Euler intervals require t >= s.')
    if torch.any(t_t > 1):
        raise ValueError('EI-Euler intervals require t <= 1.')

    variance = schedule.ei_driver_variance(s_t, t_t, t0=t0)
    modes = torch.arange(degree, device=device, dtype=dtype or torch.float32)
    denominator = (2.0 * modes + 1.0).reshape(degree, *([1] * len(shape)))
    std = (variance.unsqueeze(0) / denominator).clamp_min(0.0).sqrt()
    normalized = randn((degree, *shape), device=device, dtype=dtype, generator=generator)
    normalized = torch.where(std > 0, normalized, torch.zeros_like(normalized))
    if degrees is not None:
        degrees = torch.as_tensor(degrees, device=device, dtype=torch.long)
        if degrees.shape != (shape[0],):
            raise ValueError(f'degrees must have shape ({shape[0]},); got {tuple(degrees.shape)}.')
        if torch.any(degrees < 1) or torch.any(degrees > degree):
            raise ValueError(f'degrees must lie in [1, {degree}].')
        mask = torch.arange(degree, device=device).unsqueeze(1) < degrees.unsqueeze(0)
        normalized = normalized * mask.reshape(degree, shape[0], *([1] * (len(shape) - 1)))
    return EIEulerPath(normalized * std, normalized, variance, degrees)


def _require_euler(net: SSFMNet) -> None:
    """Require an Euler head because the sampler supplies the stochastic term."""
    if net.parameterization != 'euler':
        raise ValueError('The EI-Euler map requires net.parameterization="euler".')


def ei_euler_map(
    net: SSFMNet,
    schedule: AffineSchedule,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    path: EIEulerPath,
) -> Tensor:
    """Evaluate the increment-conditioned EI-Euler map on ``[s, t]``."""
    _require_euler(net)
    features = path.coefficient_features(normalize=net.normalize_coefficients)
    net.set_brownian(path)
    try:
        velocity = net.drift(x_s, s, t, features)
    finally:
        net.set_brownian(None)
    return apply_ei_euler_prediction(schedule, x_s, s, t, velocity, path)


def apply_ei_euler_prediction(
    schedule: AffineSchedule,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    velocity: Tensor,
    path: EIEulerPath,
) -> Tensor:
    """Assemble an EI-Euler map from an already-computed interval velocity."""
    state_coeff, velocity_coeff = schedule.ei_euler_coefficients(s, t)
    field_ones = (1,) * (x_s.ndim - 1)
    return (
        state_coeff.view(-1, *field_ones) * x_s
        + velocity_coeff.view(-1, *field_ones) * velocity
        + path.terminal_increment()
    )
