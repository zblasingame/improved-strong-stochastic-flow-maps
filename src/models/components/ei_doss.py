"""EI-then-Doss-Sussmann paths: EI-Euler skeleton, martingale added after.

The Holderrieth flow-as-SDE is ``dX = (2u - (α̇/α) X) dt + ν dW``. The
deterministic EI-Euler map on the physical Doss state ``Z_s = X_s`` is

    Z_t = (α_s / α_t) Z_s + (2 A_st / α_t) u,
    A_st = int_s^t α_r dr.

Doss-Sussmann then reattaches the EI-weighted martingale
``M = int α ν dW`` (clock ``ei_weighted_clock`` = ``int α² ν²``) in X-units:

    X_t = Z_t + (M_t - M_s) / α_t.

The backbone predicts the same velocity ``u`` as EI-Euler and consumes
physical ``X``; only the residual (ESD/LSD of this map vs PSD of the
increment-conditioned EI-Euler map) and the fact that the sampler -- not the
map -- owns the martingale distinguish the two arms. This is not the
physical-space Doss path ``X - int ν dW``.

Zeta annealing
--------------
Every object here is the ``zeta = 1`` member of the annealed SDE family
``dX = [(1+ζ²)u - ζ²(α̇/α)X] dt + ζν dW`` (paper, "Annealing the
Doss-Sussman ODE"). Passing ``zeta < 1`` rescales the same machinery:

- the lift becomes ``Y = α_t^{ζ²} X`` and the skeleton uses the annealed
  coefficients ``(α_s/α_t)^{ζ²}`` and ``(1+ζ²) A^{(ζ)} / α_t^{ζ²}`` with
  ``A^{(ζ)} = int α^{ζ²}`` (:meth:`AffineSchedule.annealed_ei_euler_coefficients`);
- the martingale becomes ``M~^ζ = ζ int α^{ζ²} ν dW`` on the annealed clock
  ``ζ² int α^{2ζ²} ν²`` (:meth:`AffineSchedule.annealed_ei_weighted_clock`);
- the reconstruction becomes ``X_t = Z_t + ΔM~^ζ / α_t^{ζ²}``.

``zeta = 0`` degenerates to the probability-flow ODE: the clock is identically
zero, so the martingale polynomial is the zero path (zero features, zero
increments) and the skeleton is the plain Euler map ``x + (t - s) u``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .brownian import BrownianPolynomial
from .doss_sussmann import (
    GlobalMartingaleInterval,
    GlobalMartingalePath,
    MartingalePath,
    _restrict_martingale_polynomial,
)
from .schedule import AffineSchedule, expand_t
from .ssfm import SSFMNet


def _alpha_pow(schedule: AffineSchedule, t: Tensor | float, zeta: float) -> Tensor:
    """Return ``alpha_t ** zeta**2`` (the annealed lift weight)."""
    return schedule.alpha(torch.as_tensor(t)) ** (zeta * zeta)


def ei_y_state(
    schedule: AffineSchedule, x: Tensor, t: Tensor | float, zeta: float = 1.0
) -> Tensor:
    """Map physical ``X`` at time ``t`` to the EI state ``Y = α_t^{ζ²} X``."""
    return expand_t(_alpha_pow(schedule, t, zeta), x) * x


def ei_x_state(
    schedule: AffineSchedule, y: Tensor, t: Tensor | float, zeta: float = 1.0
) -> Tensor:
    """Map EI state ``Y`` at time ``t`` back to ``X = Y / α_t^{ζ²}``."""
    alpha = expand_t(_alpha_pow(schedule, t, zeta), y)
    return y / alpha.clamp_min(torch.finfo(y.dtype).tiny)


def ei_doss_reconstruct(
    schedule: AffineSchedule,
    z: Tensor,
    t: Tensor | float,
    delta_m: Tensor,
    zeta: float = 1.0,
) -> Tensor:
    """Reattach the annealed martingale in X-units: ``X = Z + ΔM~^ζ / α_t^{ζ²}``."""
    alpha = expand_t(_alpha_pow(schedule, t, zeta), z)
    return z + delta_m / alpha.clamp_min(torch.finfo(z.dtype).tiny)


def ei_doss_map(
    net: SSFMNet,
    schedule: AffineSchedule,
    z: Tensor,
    s: Tensor,
    t: Tensor,
    coefficients: Tensor,
    zeta: float = 1.0,
) -> Tensor:
    """Annealed EI-Euler skeleton on the physical Doss state ``Z_s = X_s``.

    The map is the deterministic half of the annealed EI-Euler step,

        ``Z_t = (α_s / α_t)^{ζ²} Z_s + ((1+ζ²) A^{(ζ)}_{s,t} / α_t^{ζ²}) u``,

    using :meth:`AffineSchedule.annealed_ei_euler_coefficients`. The backbone
    consumes physical ``Z`` and predicts velocity ``u``. The sampler / LSD
    teacher reattaches ``ΔM~^ζ / α_t^{ζ²}`` via :func:`ei_doss_reconstruct`.

    At the diagonal the annealed coefficients are ``(1, 0)`` for every ζeta,
    so the map is the identity and ``∂_t Z|_{t=s}`` recovers the annealed
    physical drift ``b^ζ = (1+ζ²)u - ζ²(α̇/α)Z``. ``zeta = 1`` is exactly the
    un-annealed :func:`ei_doss_map`; ``zeta = 0`` is the Euler ODE skeleton
    ``x + (t-s) u``.
    """
    velocity = net.drift(z, s, t, coefficients)
    state_coeff, velocity_coeff = schedule.annealed_ei_euler_coefficients(s, t, zeta)
    return expand_t(state_coeff, z) * z + expand_t(velocity_coeff, z) * velocity


@dataclass(frozen=True)
class EIMartingalePath(MartingalePath):
    """Local ``M_r - M_s`` on the EI-weighted clock ``int α² ν²``.

    Under zeta annealing the clock is the annealed variant
    ``ζ² int α^{2ζ²} ν²`` of the annealed lifted martingale
    ``M~^ζ = ζ int α^{ζ²} ν dW``; ``zeta = 1`` (default) is the un-annealed
    EI-weighted martingale.
    """

    zeta: float = 1.0

    def clock(self, r: Tensor) -> Tensor:
        """Map physical time to the (possibly annealed) EI martingale clock."""
        return self.schedule.annealed_ei_weighted_clock(r, self.t0, self.zeta)


@dataclass(frozen=True)
class GlobalEIMartingalePath(GlobalMartingalePath):
    """One globally anchored EI-weighted martingale, ``M_{t0} = 0``.

    ``zeta < 1`` samples the annealed lifted martingale ``M~^ζ`` on the
    annealed clock; ``zeta = 0`` is the identically-zero path (the annealed
    ODE carries no noise).
    """

    zeta: float = 1.0

    def clock(self, r: Tensor) -> Tensor:
        """Map physical time to the (possibly annealed) EI martingale clock."""
        return self.schedule.annealed_ei_weighted_clock(r, self.t0, self.zeta)

    def select(self, index: Tensor) -> GlobalEIMartingalePath:
        """Return the selected batch members of the same EI realization."""
        return GlobalEIMartingalePath(
            path=BrownianPolynomial(
                coeffs=self.path.coeffs[:, index],
                s=self.path.s,
                t=self.path.t,
            ),
            schedule=self.schedule,
            t0=self.t0,
            zeta=self.zeta,
        )

    def interval(self, s: Tensor | float, t: Tensor | float) -> GlobalMartingaleInterval:
        """Restrict this realization to ``[s, t]`` in EI-clock coordinates."""
        s_t, t_t = self._as_time(s), self._as_time(t)
        if torch.any(s_t < self.t0):
            raise ValueError(f'interval views require s >= t0; got s < {self.t0}.')
        if torch.any(t_t > 1.0):
            raise ValueError('interval views require t <= 1.')
        if torch.any(s_t >= t_t):
            raise ValueError('interval views require s < t; use diagonal() at s == t.')
        clock_s, clock_t = self.clock(s_t), self.clock(t_t)
        restricted = _restrict_martingale_polynomial(self.path, clock_s, clock_t)
        local = EIMartingalePath(
            path=restricted, schedule=self.schedule, t0=self.t0, s=s_t, t=t_t, zeta=self.zeta
        )
        return GlobalMartingaleInterval(realization=self, local=local)


def sample_ei_martingale_path(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    s: Tensor | float,
    t: Tensor | float,
    degree: int,
    t0: float,
    *,
    zeta: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> EIMartingalePath:
    """Sample ``I^M_{s,t}`` on the (annealed) EI clock of ``[clock(s), clock(t)]``."""
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    s_t = torch.as_tensor(s, device=device, dtype=dtype)
    t_t = torch.as_tensor(t, device=device, dtype=dtype)
    path = BrownianPolynomial.sample(
        shape,
        s=schedule.annealed_ei_weighted_clock(s_t, t0, zeta),
        t=schedule.annealed_ei_weighted_clock(t_t, t0, zeta),
        degree=degree,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return EIMartingalePath(path=path, schedule=schedule, t0=t0, s=s_t, t=t_t, zeta=zeta)


def sample_global_ei_martingale_path(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    t0: float,
    degree: int,
    *,
    zeta: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> GlobalEIMartingalePath:
    """Sample one (annealed) EI martingale on ``[clock(t0), clock(1)]`` with ``M_{t0}=0``.

    At ``zeta = 0`` the annealed clock is identically zero, so the sampled
    polynomial is the zero path: zero coefficient features and zero increments,
    i.e. the annealed probability-flow ODE limit.
    """
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    if not 0.0 <= t0 < 1.0:
        raise ValueError(f'EI-Doss clock origin must satisfy 0 <= t0 < 1; got {t0}.')
    if not 0.0 <= zeta <= 1.0:
        raise ValueError(f'zeta must lie in [0, 1]; got {zeta}.')
    origin = torch.as_tensor(t0, device=device, dtype=dtype)
    end = torch.as_tensor(1.0, device=device, dtype=dtype)
    path = BrownianPolynomial.sample(
        shape,
        s=schedule.annealed_ei_weighted_clock(origin, t0, zeta),
        t=schedule.annealed_ei_weighted_clock(end, t0, zeta),
        degree=degree,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return GlobalEIMartingalePath(path=path, schedule=schedule, t0=t0, zeta=zeta)
