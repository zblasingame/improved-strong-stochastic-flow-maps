"""Globally anchored polynomial martingale for the 1D non-linear SDE toy.

The toy SDE ``dX = f(t, X) dt + sqrt(beta_t) dW`` has a *known* diffusion
coefficient, so its noise integral — the closed-form martingale

    ``M_r = int_{t0}^r sqrt(beta_u) dW_u``

— is a Brownian motion on the clock ``Sigma(t) = int_{t0}^t beta_u du``,
which is quadratic in physical time (``beta_t`` is linear):

    ``Sigma(t) = beta_max t - (beta_max - beta_min) t^2 / 2``.

This module realizes ``M`` as one degree-``N`` polynomial Brownian path on
``[Sigma(0), Sigma(1)]`` and exposes the same view semantics the 2D toy's
martingales use: differentiable interval restrictions (exact Chen-consistent
increments and coefficient features for any sub-interval) and the physical
driver ``Mdot_r = dM/dr`` (the Wong-Zakai white-noise term, in closed form
via the chain rule ``dM/dSigma . beta_r``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .brownian import BrownianPolynomial, gauss_legendre_rule, shifted_legendre
from .nonlinear_sde import NonlinearSDE


def toy_clock(sde: NonlinearSDE, t: Tensor | float) -> Tensor:
    """Return the closed-form clock ``Sigma(t) = int_0^t beta_u du``."""
    t_t = torch.as_tensor(t)
    value = sde.beta_max * t_t - 0.5 * (sde.beta_max - sde.beta_min) * t_t.square()
    return value if t_t.ndim else value.reshape(())


@dataclass(frozen=True)
class GlobalToyMartingale:
    """One globally anchored ``M_r = int_0^r sqrt(beta) dW`` realization, ``M_0 = 0``.

    ``path`` is a degree-``N`` polynomial Brownian path on the clock interval
    ``[0, Sigma(1)]``. Interval views re-express the same realization on
    physical sub-intervals via exact clock-coordinate restrictions, so
    adjacent views share global martingale values and their increments
    compose (Chen) into the parent view.
    """

    path: BrownianPolynomial
    sde: NonlinearSDE

    @property
    def degree(self) -> int:
        """Polynomial degree of the clock-time path."""
        return self.path.degree

    def clock(self, r: Tensor | float) -> Tensor:
        """Map physical time ``r`` to the martingale clock."""
        return toy_clock(self.sde, r)

    def global_value(self, r: Tensor | float) -> Tensor:
        """Return ``M_r`` for ``r in [0, 1]`` (differentiable in ``r``)."""
        r_t = torch.as_tensor(r, device=self.path.coeffs.device, dtype=self.path.coeffs.dtype)
        if torch.any(r_t < 0.0) or torch.any(r_t > 1.0):
            raise ValueError('global toy martingale values require r in [0, 1].')
        return self.path.increment(self.path.s.expand(self.path.coeffs.shape[1:]), self.clock(r_t))

    def driver(self, r: Tensor, reference: Tensor) -> Tensor:
        """Return the physical-time driver ``Mdot_r = dM/dSigma . beta_r``.

        ``r`` may be per-sample ``(B,)`` times; the result broadcasts against
        ``reference`` — the Wong-Zakai white-noise term of the smoothed RODE
        ``dX/dr = f(r, X) + Mdot_r``.
        """
        r_field = r.reshape(-1, *([1] * (reference.ndim - 1))).to(
            device=self.path.coeffs.device, dtype=self.path.coeffs.dtype
        )
        clock_r = self.clock(r_field)
        beta_r = self.sde.beta(r_field)
        return self.path.derivative(clock_r) * beta_r

    def interval(self, s: Tensor | float, t: Tensor | float) -> BrownianPolynomial:
        """Restrict the realization to physical ``[s, t]`` in clock coordinates.

        The returned polynomial lives on ``[Sigma(s), Sigma(t)]``: its raw
        increment is the exact closed-form martingale increment ``M_t - M_s``
        and its coefficient features are the network conditioning.
        """
        s_t = torch.as_tensor(s, device=self.path.coeffs.device, dtype=self.path.coeffs.dtype)
        t_t = torch.as_tensor(t, device=self.path.coeffs.device, dtype=self.path.coeffs.dtype)
        if torch.any(s_t < 0.0) or torch.any(t_t > 1.0):
            raise ValueError('toy martingale intervals must lie in [0, 1].')
        if torch.any(s_t >= t_t):
            raise ValueError('toy martingale intervals require s < t.')
        return _restrict_polynomial(self.path, self.clock(s_t), self.clock(t_t))


def _restrict_polynomial(path: BrownianPolynomial, s: Tensor, t: Tensor) -> BrownianPolynomial:
    """Re-express one clock polynomial on the child interval ``[s, t]``.

    Same construction as :func:`src.models.components.doss_sussmann._restrict_martingale_polynomial`
    (degree-``N`` Gauss-Legendre quadrature of the derivative against the child
    shifted-Legendre basis, with the zeroth child coefficient assigned from the
    exact parent increment).
    """
    degree = path.degree
    nodes, weights = gauss_legendre_rule(degree)
    span = (t - s).clamp_min(0.0)
    field_span = (
        span if span.ndim == 0 else span.reshape(span.shape[0], *([1] * (path.coeffs.ndim - 2)))
    )
    evals = torch.stack([path.derivative(s + x * span) for x in nodes], dim=0)
    node_t = path.coeffs.new_tensor(nodes)
    weight_t = path.coeffs.new_tensor(weights)
    basis = torch.stack([shifted_legendre(n, node_t) for n in range(degree)])
    weighted = (basis * weight_t).reshape(degree, degree, *([1] * (evals.ndim - 1)))
    projection = (weighted * evals.unsqueeze(0)).sum(dim=1)
    zeroth = path.increment(s, t)
    coeffs = torch.cat((zeroth.unsqueeze(0), field_span * projection[1:]), dim=0)
    return BrownianPolynomial(coeffs=coeffs, s=s, t=t)


def sample_global_toy_martingale(
    sde: NonlinearSDE,
    shape: tuple[int, ...],
    degree: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> GlobalToyMartingale:
    """Sample one global closed-form martingale on ``[0, Sigma(1)]``."""
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    origin = torch.zeros((), device=device, dtype=dtype)
    end = toy_clock(sde, torch.ones((), device=device, dtype=dtype))
    path = BrownianPolynomial.sample(
        shape, origin, end, degree, device=device, dtype=dtype, generator=generator
    )
    return GlobalToyMartingale(path=path, sde=sde)


def sample_piecewise_toy_martingale(
    sde: NonlinearSDE,
    shape: tuple[int, ...],
    degree: int,
    steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[Tensor, list[BrownianPolynomial]]:
    """Sample independent clock-polynomial leaves on a fine physical grid.

    Each leaf ``k`` lives on the clock span of the physical interval
    ``[k/steps, (k+1)/steps]``; its raw increment is an exact draw of the
    closed-form martingale increment over that interval — the SDE's own noise
    — so a fine Euler-Maruyama reference riding the leaves is pathwise
    coupled to any map consuming restrictions of the same realization.
    """
    if steps < 1:
        raise ValueError(f'steps must be >= 1; got {steps}.')
    nodes = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    clock_nodes = toy_clock(sde, nodes)
    leaves = [
        BrownianPolynomial.sample(
            shape,
            clock_nodes[index],
            clock_nodes[index + 1],
            degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for index in range(steps)
    ]
    return clock_nodes, leaves


