"""Local and globally anchored clock-time paths for the Doss-Sussmann random ODE.

For ``dX_t = b(t, X_t) dt + nu_t dW_t``, let
``Sigma(t) = int_{t0}^t nu_r**2 dr``.  On a build interval ``[s, t]`` the
martingale increment is a standard Brownian motion on
``[Sigma(s), Sigma(t)]``.  Subtracting this *local* increment gives

    Z_r^s = X_r - (M_r - M_s),
    dZ_r^s = b(r, Z_r^s + M_r - M_s) dr,

with ``Z_s^s = X_s``.  The polynomial coefficients supplied to the network are
therefore sampled directly on the build interval in clock time, exactly as the
ordinary SSFM samples Brownian polynomials on native-time build intervals.

A *globally anchored* variant instead realizes the whole martingale
``M_r = int_{t0}^r nu_u dW_u`` on the full clock interval
``[Sigma(t0), Sigma(1)]`` as one polynomial path with ``M_{t0} = 0``
(:class:`GlobalMartingalePath`).  Interval and diagonal *views* re-express that
single realization on sub-intervals of the clock, so adjacent views share the
same global martingale values and their physical reconstructions compose,
while diagonal evaluation reads nondegenerate coefficients from a build window
ending at the diagonal time instead of sampling a zero-span path.  The
drift-RODE state, clock, and reconstruction semantics are unchanged: views
still evolve the transformed state ``Z`` and reattach ``M_t - M_s``; nothing
here uses the x-prediction integral-map coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .brownian import BrownianPolynomial, gauss_legendre_rule, shifted_legendre
from .schedule import AffineSchedule


@dataclass(frozen=True)
class MartingalePath:
    """Polynomial realization of ``M_r - M_s`` on one native interval ``[s, t]``."""

    path: BrownianPolynomial
    schedule: AffineSchedule
    t0: float
    s: Tensor
    t: Tensor

    @property
    def degree(self) -> int:
        """Polynomial degree of the clock-time path."""
        return self.path.degree

    def clock(self, r: Tensor) -> Tensor:
        """Map native time ``r`` to the martingale clock anchored at ``t0``."""
        return self.schedule.martingale_clock(r, self.t0)

    def martingale(self, r: Tensor) -> Tensor:
        """Return the local increment ``M_r - M_s`` for ``r in [s, t]``."""
        clock_r = self.clock(r)
        clock_s = self.path.s.expand_as(clock_r)
        return self.path.increment(clock_s, clock_r)

    def terminal_increment(self) -> Tensor:
        """Return the exact full build-interval increment ``M_t - M_s``."""
        return self.path.raw_increment()

    def coefficient_features(self, *, normalize: bool = True) -> Tensor:
        """Return build-interval clock-polynomial features as ``(B, N, ...)``."""
        return self.path.coefficient_features(normalize=normalize)


def _restrict_martingale_polynomial(
    path: BrownianPolynomial,
    s: Tensor,
    t: Tensor,
) -> BrownianPolynomial:
    """Re-express one polynomial realization on the child clock interval ``[s, t]``.

    The path derivative is a degree-``N - 1`` polynomial on the parent
    interval, hence also on any child interval after the affine coordinate
    change, so an ``N``-node Gauss--Legendre rule integrates its products with
    the child shifted-Legendre basis exactly.  The zeroth child coefficient is
    assigned from the exact parent increment so increment additivity across
    adjacent children holds independently of quadrature round-off, following
    the :meth:`BrownianPolynomial.concat_arbitrary` precedent.

    :param path: Parent polynomial on a clock interval containing ``[s, t]``.
    :param s: Child interval start on the parent clock, broadcastable with the
        sample shape (pre-shaped per-sample tensors, e.g. ``(B, 1, 1, 1)``).
    :param t: Child interval end, with ``t > s`` elementwise.
    :return: The same realization expressed in the child basis on ``[s, t]``.
    """
    degree = path.degree
    nodes, weights = gauss_legendre_rule(degree)
    span = (t - s).clamp_min(0.0)
    field_span = (
        span
        if span.ndim == 0
        else span.reshape(span.shape[0], *([1] * (path.coeffs.ndim - 2)))
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


@dataclass(frozen=True)
class GlobalMartingalePath:
    """One globally anchored martingale realization on the schedule clock.

    Wraps a single polynomial Brownian realization of the diffusion martingale
    ``M_r = int_{t0}^r nu_u dW_u`` on the full clock interval
    ``[Sigma(t0), Sigma(1)]``, anchored so ``M_{t0} = 0``.  Interval and
    diagonal views derived from this object re-express the *same* realization
    on clock sub-intervals: adjacent views share the same global martingale
    values, their increments compose into their parent view's increment, and
    no view ever samples a fresh or zero-span path.
    """

    path: BrownianPolynomial
    schedule: AffineSchedule
    t0: float

    @property
    def degree(self) -> int:
        """Polynomial degree of the global clock-time path."""
        return self.path.degree

    def _as_time(self, value: Tensor | float) -> Tensor:
        """Convert a physical time to the realization's device and dtype."""
        return torch.as_tensor(value, device=self.path.coeffs.device, dtype=self.path.coeffs.dtype)

    def clock(self, r: Tensor) -> Tensor:
        """Map physical time ``r`` to the martingale clock anchored at ``t0``."""
        return self.schedule.martingale_clock(r, self.t0)

    def global_value(self, r: Tensor | float) -> Tensor:
        """Return the global martingale value ``M_r`` for ``r in [t0, 1]``.

        Differentiable in ``r``; exactly zero at ``r = t0``.
        """
        r_t = self._as_time(r)
        if torch.any(r_t < self.t0):
            raise ValueError(f'global martingale values require r >= t0; got r < {self.t0}.')
        if torch.any(r_t > 1.0):
            raise ValueError('global martingale values require r <= 1.')
        clock_r = self.clock(r_t)
        origin = self.path.s.expand_as(clock_r)
        return self.path.increment(origin, clock_r)

    def select(self, index: Tensor) -> GlobalMartingalePath:
        """Return the selected batch members of the same global realization."""
        return GlobalMartingalePath(
            path=BrownianPolynomial(
                coeffs=self.path.coeffs[:, index],
                s=self.path.s,
                t=self.path.t,
            ),
            schedule=self.schedule,
            t0=self.t0,
        )

    def interval(self, s: Tensor | float, t: Tensor | float) -> GlobalMartingaleInterval:
        """Return the coherent view of this realization on ``[s, t]``.

        The view's local path is the exact restriction of the global
        polynomial to ``[Sigma(s), Sigma(t)]``, so its increments, features,
        and reconstruction match the established local drift-RODE semantics
        while remaining coherent with every other view of this realization.
        Diagonal evaluation (``s == t``) must use :meth:`diagonal`, which
        retains nondegenerate conditioning from a nonzero build window.
        """
        s_t, t_t = self._as_time(s), self._as_time(t)
        if torch.any(s_t < self.t0):
            raise ValueError(f'interval views require s >= t0; got s < {self.t0}.')
        if torch.any(t_t > 1.0):
            raise ValueError('interval views require t <= 1.')
        if torch.any(s_t >= t_t):
            raise ValueError('interval views require s < t; use diagonal() at s == t.')
        clock_s, clock_t = self.clock(s_t), self.clock(t_t)
        restricted = _restrict_martingale_polynomial(self.path, clock_s, clock_t)
        local = MartingalePath(path=restricted, schedule=self.schedule, t0=self.t0, s=s_t, t=t_t)
        return GlobalMartingaleInterval(realization=self, local=local)

    def diagonal(self, r: Tensor | float, dt: float) -> GlobalMartingaleInterval:
        """Return the diagonal-conditioning view at time ``r``.

        The view covers the nondegenerate build window
        ``[max(r - dt, t0), r]`` of the same global realization, following the
        established drift-RODE diagonal convention: the field is evaluated at
        the diagonal time ``r`` while its coefficient features come from the
        build window, never from a zero-span path.

        :param r: Diagonal time in ``(t0, 1]``.
        :param dt: Positive build-window length in physical time.
        """
        r_t = self._as_time(r)
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError(f'diagonal views require dt > 0; got {dt_f}.')
        if torch.any(r_t <= self.t0):
            raise ValueError(
                f'diagonal views require r > t0 so the build window is nondegenerate; '
                f'got r <= {self.t0}.'
            )
        if torch.any(r_t > 1.0):
            raise ValueError('diagonal views require r <= 1.')
        return self.interval((r_t - dt_f).clamp_min(self.t0), r_t)


@dataclass(frozen=True)
class GlobalMartingaleInterval:
    """Interval view ``[s, t]`` of one globally anchored martingale realization.

    ``local`` re-expresses the shared realization on ``[Sigma(s), Sigma(t)]``
    as an ordinary :class:`MartingalePath`, so interval consumers keep the
    established local semantics: ``martingale(r)`` is the local increment
    ``M_r - M_s``, :meth:`terminal_increment` is the exact ``M_t - M_s``, and
    physical reconstruction stays ``X_t = Z_t^s + (M_t - M_s)``.  Global
    accessors read the shared realization directly, which is what makes
    adjacent views share global martingale values and compose into their
    parent view.
    """

    realization: GlobalMartingalePath
    local: MartingalePath

    @property
    def degree(self) -> int:
        """Polynomial degree of the view path."""
        return self.local.degree

    @property
    def s(self) -> Tensor:
        """Physical start time of the view."""
        return self.local.s

    @property
    def t(self) -> Tensor:
        """Physical terminal time of the view."""
        return self.local.t

    @property
    def path(self) -> BrownianPolynomial:
        """Restricted clock-time polynomial on ``[Sigma(s), Sigma(t)]``."""
        return self.local.path

    def clock(self, r: Tensor) -> Tensor:
        """Map physical time ``r`` to the martingale clock anchored at ``t0``."""
        return self.local.clock(r)

    def martingale(self, r: Tensor) -> Tensor:
        """Return the local increment ``M_r - M_s`` of the shared realization."""
        return self.local.martingale(r)

    def terminal_increment(self) -> Tensor:
        """Return the exact full-view increment ``M_t - M_s``."""
        return self.local.terminal_increment()

    def coefficient_features(self, *, normalize: bool = True) -> Tensor:
        """Return view conditioning features as ``(B, N, ...)``."""
        return self.local.coefficient_features(normalize=normalize)

    def global_value(self, r: Tensor | float) -> Tensor:
        """Return the shared global martingale value ``M_r``."""
        return self.realization.global_value(r)

    def global_start(self) -> Tensor:
        """Return the shared global value ``M_s`` at the view start."""
        return self.realization.global_value(self.s)

    def global_terminal(self) -> Tensor:
        """Return the shared global value ``M_t`` at the view terminal."""
        return self.realization.global_value(self.t)


