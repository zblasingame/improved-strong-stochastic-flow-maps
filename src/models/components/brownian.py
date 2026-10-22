"""Polynomial approximation of Brownian motion (arXiv:2606.01086).

A degree-``N`` polynomial Brownian motion over an interval ``[s, t]`` is built
from independent Gaussian coefficients projected onto the shifted Legendre
basis. The construction has two properties that make it ideal for stochastic
flow maps:

1. The increment over the *full* interval reproduces the raw Brownian
   increment exactly, ``W_{s,t}^{(N)} = W_t - W_s = I^{(0)}``, so the forward
   pass is numerically identical to Euler--Maruyama.
2. The sub-interval increment ``W_{u,v}^{(N,s,t)}`` is a smooth, differentiable
   function of its endpoints ``(u, v)``. Differentiating through it with
   autograd recovers the ``+/- Wdot`` total-derivative terms needed by the
   Eulerian / Lagrangian self-distillation losses (ESD / LSD), which a raw
   (constant) increment cannot provide.

Coefficient distribution: ``I_{s,t}^{(n)} ~ N(0, (t - s) / (2n + 1))``,
independent across ``n`` and across channels/pixels.

A *globally anchored* variant realizes the whole path on one full interval
``[t0, t1]`` as a single :class:`BrownianPolynomial` with ``W_{t0} = 0``
(:class:`GlobalBrownianPath`). Interval, diagonal, and sampling *views*
re-express that single realization on sub-intervals via
:meth:`BrownianPolynomial.restrict`, so adjacent views share the same global
values, their increments compose, and diagonal evaluation reads nondegenerate
conditioning from a nonzero build window instead of sampling a zero-span
path. This is the coherent-path construction the Wong--Zakai (X-space)
objectives and samplers consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch
from torch import Tensor

from .gaussian import randn


def shifted_legendre(n: int, x: Tensor) -> Tensor:
    """Evaluate the ``n``-th shifted Legendre polynomial ``P_tilde_n`` on ``[0, 1]``.

    ``P_tilde_n(x) = P_n(2x - 1)`` where ``P_n`` is the standard Legendre
    polynomial. Computed via the Bonnet recurrence for numerical stability.

    :param n: Polynomial degree (``>= 0``).
    :param x: Evaluation points in ``[0, 1]``, any shape.
    :return: ``P_tilde_n(x)`` with the same shape as ``x``.
    """
    z = 2.0 * x - 1.0  # map [0, 1] -> [-1, 1]
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return z
    p_prev, p_curr = torch.ones_like(z), z
    for k in range(1, n):
        p_next = ((2 * k + 1) * z * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def integrate_shifted_legendre(n: int, a: Tensor, b: Tensor) -> Tensor:
    """Integrate ``P_tilde_n`` over ``[a, b] subset [0, 1]``: ``int_a^b P_tilde_n(r) dr``.

    Uses the antiderivative identity for (shifted) Legendre polynomials. For the
    standard basis, ``int P_n = (P_{n+1} - P_{n-1}) / (2n + 1)``; the shifted /
    rescaled form is applied here.

    :param n: Polynomial degree.
    :param a: Lower limits in ``[0, 1]``, broadcastable with ``b``.
    :param b: Upper limits in ``[0, 1]``.
    :return: Definite integral value(s).
    """
    if n == 0:
        return b - a

    # Antiderivative of the shifted Legendre polynomial:
    #   A_n(r) = (P_tilde_{n+1}(r) - P_tilde_{n-1}(r)) / (2 (2n + 1)),
    # so int_a^b P_tilde_n = A_n(b) - A_n(a). Differentiable in a and b.
    def antideriv(r: Tensor) -> Tensor:
        return (shifted_legendre(n + 1, r) - shifted_legendre(n - 1, r)) / (2.0 * (2 * n + 1))

    return antideriv(b) - antideriv(a)


@cache
def chen_dilation_coefficient(n: int, m: int) -> float:
    """Single dilation coefficient ``c_{n,m}`` from the paper's Chen relations.

    From the shifted-Legendre dilation rule (arXiv:2606.01086, Lemma
    "Dilation rule for shifted Legendre polynomials"):

        P_tilde_n(x) = sum_{m=0}^n c_{n,m} P_tilde_m(2x),

    with

        c_{n,m} = (-1)^n (2m+1) sum_{k=m}^n (-1/2)^k
                  * (n + k)! / ((n - k)! (k - m)! (k + m + 1)!).

    The coefficients are lower-triangular (``c_{n,m} = 0`` for ``m > n``). Computed
    in float64 via :func:`math.factorial` for exactness, then consumed as needed.

    :param n: Parent (outer) degree, ``>= 0``.
    :param m: Child (inner) degree, ``0 <= m <= n``.
    :return: The scalar coefficient ``c_{n,m}``.
    """
    if m > n:
        return 0.0
    total = 0.0
    for k in range(m, n + 1):
        numerator = math.factorial(n + k)
        denominator = math.factorial(n - k) * math.factorial(k - m) * math.factorial(k + m + 1)
        total += (-0.5) ** k * numerator / denominator
    return ((-1.0) ** n) * (2 * m + 1) * total


@cache
def chen_dilation_matrix(degree: int) -> tuple[tuple[float, ...], ...]:
    """Lower-triangular dilation matrix ``c_{n,m}`` for degrees ``0..degree-1``.

    Cached pure-Python representation (tuple-of-tuples so it is hashable for
    :func:`lru_cache`); callers materialise it into a tensor on the right device
    via :meth:`BrownianPolynomial.concat`.

    :param degree: Number of basis terms ``N``.
    :return: ``N x N`` lower-triangular matrix as nested tuples, row ``n`` holding
        ``(c_{n,0}, ..., c_{n,N-1})`` with zeros above the diagonal.
    """
    return tuple(
        tuple(chen_dilation_coefficient(n, m) for m in range(degree)) for n in range(degree)
    )


@cache
def gauss_legendre_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return an ``order``-node Gauss--Legendre rule on ``[0, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return tuple((0.5 * (nodes + 1.0)).tolist()), tuple((0.5 * weights).tolist())


@dataclass(frozen=True)
class BrownianPolynomial:
    """Immutable degree-``N`` polynomial Brownian motion on ``[s, t]``.

    Holds the sampled Legendre coefficients ``I^{(n)}`` (shape
    ``(N, *sample_shape)``) together with the interval endpoints. All accessor
    methods are pure: they read the stored coefficients and return new tensors.

    :param coeffs: Stacked coefficients ``I^{(0..N-1)}``, shape ``(N, B, C, H, W)``.
    :param s: Interval start time (scalar tensor or float).
    :param t: Interval end time (scalar tensor or float).
    :param degrees: Optional active coefficient counts for each batch member,
        shape ``(B,)``. Coefficients above each active count are padded with
        zeros. Omitting this preserves the legacy fixed-degree representation.
    """

    coeffs: Tensor
    s: Tensor
    t: Tensor
    degrees: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate mixed-degree metadata and zero padded coefficients."""
        if self.degree < 1:
            raise ValueError(f'degree must be at least 1; got {self.degree}.')
        if self.degrees is None:
            return
        if self.coeffs.ndim < 2:
            raise ValueError('per-sample degrees require coefficients with a batch dimension.')

        degrees = self.degrees.to(device=self.coeffs.device)
        if degrees.ndim != 1:
            raise ValueError(f'degrees must have shape (B,); got {tuple(degrees.shape)}.')
        if degrees.shape[0] != self.coeffs.shape[1]:
            raise ValueError(
                f'degrees batch size must match coefficients; got {degrees.shape[0]} and '
                f'{self.coeffs.shape[1]}.'
            )
        if torch.is_floating_point(degrees) or torch.is_complex(degrees):
            raise ValueError('degrees must use an integer tensor dtype.')
        if torch.any(degrees < 1) or torch.any(degrees > self.degree):
            raise ValueError(f'degrees must lie in [1, {self.degree}].')

        degrees = degrees.to(dtype=torch.long)
        object.__setattr__(self, 'degrees', degrees)
        mask = (
            self.degree_mask()
            .transpose(0, 1)
            .reshape(self.degree, self.coeffs.shape[1], *([1] * (self.coeffs.ndim - 2)))
        )
        object.__setattr__(self, 'coeffs', self.coeffs * mask.to(dtype=self.coeffs.dtype))

    @property
    def degree(self) -> int:
        """Number of basis terms ``N``."""
        return self.coeffs.shape[0]

    @classmethod
    def sample(
        cls,
        shape: tuple[int, ...],
        s: Tensor | float,
        t: Tensor | float,
        degree: int,
        *,
        degrees: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> BrownianPolynomial:
        """Sample the Legendre coefficients for a fresh polynomial Brownian path.

        ``I^{(n)} ~ N(0, (t - s) / (2n + 1))`` independently per term and per
        element of ``shape``.

        :param shape: Per-coefficient sample shape, e.g. ``(B, C, H, W)``.
        :param s: Interval start time.
        :param t: Interval end time.
        :param degree: Number of stored basis terms ``N``. In mixed-degree mode,
            this is the padded maximum degree.
        :param degrees: Optional integer active degree for each batch member,
            bounded by ``[1, degree]``.
        :return: A new :class:`BrownianPolynomial`.
        """
        if degree < 1:
            raise ValueError(f'degree must be at least 1; got {degree}.')
        if degrees is not None and not shape:
            raise ValueError(
                'per-sample degrees require a non-empty sample shape with batch axis 0.'
            )

        s_t = torch.as_tensor(s, device=device, dtype=dtype)
        t_t = torch.as_tensor(t, device=device, dtype=dtype)
        # span broadcasts against the (N, *shape) coefficient tensor; per-sample
        # times should be passed pre-reshaped (e.g. (B, 1, 1, 1)) to broadcast.
        span = (t_t - s_t).clamp_min(0.0)
        n = torch.arange(degree, device=device, dtype=dtype or torch.float32)
        n = n.reshape(degree, *([1] * len(shape)))
        std = torch.sqrt(span / (2.0 * n + 1.0))  # (N, *broadcast)
        noise = randn((degree, *shape), device=device, dtype=dtype, generator=generator)
        coeffs = noise * std
        return cls(coeffs=coeffs, s=s_t, t=t_t, degrees=degrees)

    def degree_mask(self) -> Tensor:
        """Return the active coefficient mask with shape ``(B, degree)``.

        For legacy fixed-degree paths, every coefficient is active. Mixed-degree
        paths use the optional per-sample ``degrees`` metadata.
        """
        batch_size = self.coeffs.shape[1] if self.coeffs.ndim >= 2 else 1
        if self.degrees is None:
            return torch.ones(batch_size, self.degree, device=self.coeffs.device, dtype=torch.bool)
        order = torch.arange(self.degree, device=self.coeffs.device)
        return order.unsqueeze(0) < self.degrees.unsqueeze(1)

    @staticmethod
    def _compatible_degrees(left: BrownianPolynomial, right: BrownianPolynomial) -> Tensor | None:
        """Validate compatible degree metadata and return the preserved metadata."""
        left_degrees = left.degrees
        right_degrees = right.degrees
        if left_degrees is None and right_degrees is None:
            return None

        full = torch.full(
            (left.coeffs.shape[1],), left.degree, device=left.coeffs.device, dtype=torch.long
        )
        left_effective = full if left_degrees is None else left_degrees
        right_effective = full if right_degrees is None else right_degrees
        if not torch.equal(left_effective, right_effective):
            raise ValueError('Half-path per-sample degrees must match.')
        return left_degrees if left_degrees is not None else right_degrees

    @classmethod
    def concat(cls, left: BrownianPolynomial, right: BrownianPolynomial) -> BrownianPolynomial:
        """Concatenate two adjacent half-paths into the parent via Chen relations.

        The child intervals must have equal span in the coordinate on which the
        Brownian polynomial is defined. The parent coefficients satisfy

        ``I_parent^(n) = sum_m c_nm (I_left^(m) + (-1)^(n+m) I_right^(m))``.

        :param left: Path on ``[s, u]`` (the earlier sub-interval).
        :param right: Path on ``[u, t]`` (the later sub-interval), with
            ``right.s == left.t`` and equal degree/shape/span.
        :return: The reconstructed parent path on ``[left.s, right.t]``.
        :raises ValueError: If the degrees mismatch or the sub-intervals are not
            adjacent (``left.t != right.s``).
        """
        if left.degree != right.degree:
            raise ValueError(
                f'Half-path degrees must match; got {left.degree} and {right.degree}.'
            )
        if not torch.allclose(left.t, right.s):
            raise ValueError('Half-paths must be adjacent: left.t must equal right.s.')
        # Equal spans in exact math; the tolerance absorbs the float32
        # rounding of the midpoint (up to ~1e-7 at times of order 1), which
        # exceeds the default atol when the half-span itself is tiny
        # (e.g. h ~ 1e-3 matching steps).
        if not torch.allclose(left.t - left.s, right.t - right.s, atol=1e-6):
            raise ValueError('Half-paths must have equal spans in their Brownian coordinate.')
        if left.coeffs.shape != right.coeffs.shape:
            raise ValueError('Half-paths must have matching coefficient shapes.')

        degree = left.degree
        coeffs = left.coeffs
        chen = torch.tensor(chen_dilation_matrix(degree), device=coeffs.device, dtype=coeffs.dtype)
        n_idx = torch.arange(degree, device=coeffs.device).view(degree, 1)
        m_idx = torch.arange(degree, device=coeffs.device).view(1, degree)
        signs = torch.where(
            (n_idx + m_idx) % 2 == 0,
            coeffs.new_ones(()),
            -coeffs.new_ones(()),
        )
        contributions = left.coeffs.unsqueeze(0) + signs.reshape(
            degree, degree, *([1] * (coeffs.ndim - 1))
        ) * right.coeffs.unsqueeze(0)
        weighted = chen.reshape(degree, degree, *([1] * (coeffs.ndim - 1))) * contributions
        return cls(
            coeffs=weighted.sum(dim=1),
            s=left.s,
            t=right.t,
            degrees=cls._compatible_degrees(left, right),
        )

    @classmethod
    def concat_arbitrary(
        cls, left: BrownianPolynomial, right: BrownianPolynomial
    ) -> BrownianPolynomial:
        """Project adjacent child paths onto a parent basis for any split ratio.

        An ``N``-node Gauss--Legendre rule integrates the child/parent basis
        products exactly because their degree is at most ``2N - 2``. The
        degree-zero coefficient is assigned directly to preserve exact increment
        additivity independently of floating-point quadrature error.
        """
        if left.degree != right.degree:
            raise ValueError(
                f'Child-path degrees must match; got {left.degree} and {right.degree}.'
            )
        if not torch.allclose(left.t, right.s):
            raise ValueError('Child paths must be adjacent: left.t must equal right.s.')
        if left.coeffs.shape != right.coeffs.shape:
            raise ValueError('Child paths must have matching coefficient shapes.')

        left_span = left.t - left.s
        right_span = right.t - right.s
        # A zero-span child is a zero increment (the exact zeta -> 0 limit of
        # the annealed clock, or any degenerate span): absorb it instead of
        # failing the strict-span check.
        if torch.all(right_span <= 0):
            return left
        if torch.all(left_span <= 0):
            return right
        if torch.any(left_span <= 0) or torch.any(right_span <= 0):
            raise ValueError('Child paths must have strictly positive spans.')

        degree = left.degree
        nodes_raw, weights_raw = gauss_legendre_rule(degree)
        nodes = left.coeffs.new_tensor(nodes_raw)
        weights = left.coeffs.new_tensor(weights_raw)
        split = left_span / (left_span + right_span)

        child_shape = (degree, degree) + (1,) * (left.coeffs.ndim - 1)
        child_basis = torch.stack([shifted_legendre(m, nodes) for m in range(degree)]).reshape(
            child_shape
        )
        weights = weights.reshape((1, degree) + (1,) * (left.coeffs.ndim - 1))
        nodes = nodes.reshape((1, degree) + (1,) * (left.coeffs.ndim - 1))
        split = split.unsqueeze(0).unsqueeze(0)

        left_parent_basis = torch.stack(
            [shifted_legendre(n, split * nodes) for n in range(degree)]
        )
        right_parent_basis = torch.stack(
            [shifted_legendre(n, split + (1.0 - split) * nodes) for n in range(degree)]
        )
        projection_scale = left.coeffs.new_tensor([2 * m + 1 for m in range(degree)]).reshape(
            (1, degree) + (1,) * (left.coeffs.ndim - 1)
        )
        left_projection = (weights * child_basis * left_parent_basis).sum(dim=2)
        right_projection = (weights * child_basis * right_parent_basis).sum(dim=2)
        coeffs = (
            projection_scale
            * (
                left_projection * left.coeffs.unsqueeze(0)
                + right_projection * right.coeffs.unsqueeze(0)
            )
        ).sum(dim=1)
        coeffs[0] = left.raw_increment() + right.raw_increment()
        return cls(
            coeffs=coeffs,
            s=left.s,
            t=right.t,
            degrees=cls._compatible_degrees(left, right),
        )

    @classmethod
    def build_dyadic_levels(
        cls, leaves: list[BrownianPolynomial]
    ) -> list[list[BrownianPolynomial]]:
        """Coarsen a finest-level dyadic path into every coarser resolution.

        Given ``2^k`` independent, adjacent leaf paths tiling ``[0, 1]`` on a
        *uniform* grid (``leaves[i].t == leaves[i + 1].s``), repeatedly fold
        adjacent pairs through the Chen relations (:meth:`concat`). Each coarser
        level therefore rides the *same realized Brownian motion* as the finest
        level, observed at half the time/polynomial resolution. The final level
        holds a single path on ``[0, 1]`` equal to the Chen-composition of all
        leaves. This shared-noise tree powers a step-count sweep where the only
        difference between resolutions is discretization.

        Pairwise folding is exact only when each coarse interval's midpoint is a
        node of the finer grid, i.e. a uniform (un-warped) dyadic grid; callers
        must not apply a non-uniform time ``shift`` to the leaf grid.

        :param leaves: Finest-level paths, length a power of two, in time order.
        :return: Levels from finest to coarsest; ``out[0] is leaves`` (length
            ``2^k``), ``out[-1]`` has length one (the path on ``[0, 1]``).
        :raises ValueError: If ``leaves`` is empty or its length is not a power
            of two.
        """
        num_leaves = len(leaves)
        if num_leaves == 0 or (num_leaves & (num_leaves - 1)) != 0:
            raise ValueError(
                f'Number of leaves must be a positive power of two; got {num_leaves}.'
            )

        levels = [leaves]
        current = leaves
        while len(current) > 1:
            current = [cls.concat(current[i], current[i + 1]) for i in range(0, len(current), 2)]
            levels.append(current)
        return levels

    def restrict(self, s: Tensor | float, t: Tensor | float) -> BrownianPolynomial:
        """Re-express this realization on a child interval ``[s, t]``.

        The path derivative is a degree-``N - 1`` polynomial on the parent
        interval, hence also on any child interval after the affine coordinate
        change, so an ``N``-node Gauss--Legendre rule integrates its products
        with the child shifted-Legendre basis exactly: the restricted path is
        the *same* realization, not an approximation or a resample. The zeroth
        child coefficient is assigned from the exact parent increment so
        increment additivity across adjacent children holds independently of
        quadrature round-off, following the :meth:`concat_arbitrary` precedent.

        Both endpoints may be per-sample tensors broadcastable against the
        sample shape, and the restriction is differentiable in ``s`` and ``t``:
        moving a child endpoint (e.g. inside a JVP) follows the same stored
        coefficients, never a resample.

        :param s: Child interval start, ``>= self.s``, broadcastable with the
            sample shape (pre-shaped per-sample tensors, e.g. ``(B, 1, 1, 1)``).
        :param t: Child interval end, ``<= self.t``, with ``t > s`` elementwise.
        :return: The same realization expressed in the child basis on ``[s, t]``.
        :raises ValueError: If the child interval leaves the parent domain or
            is degenerate (``t <= s``).
        """
        s_t = torch.as_tensor(s, device=self.coeffs.device, dtype=self.coeffs.dtype)
        t_t = torch.as_tensor(t, device=self.coeffs.device, dtype=self.coeffs.dtype)
        if torch.any(s_t < self.s):
            raise ValueError('restriction requires s >= the parent interval start.')
        if torch.any(t_t > self.t):
            raise ValueError('restriction requires t <= the parent interval end.')
        if torch.any(t_t <= s_t):
            raise ValueError('restriction requires a strictly positive child span (t > s).')

        degree = self.degree
        nodes, weights = gauss_legendre_rule(degree)
        span = (t_t - s_t).clamp_min(0.0)
        evals = torch.stack([self.derivative(s_t + x * span) for x in nodes], dim=0)
        node_t = self.coeffs.new_tensor(nodes)
        weight_t = self.coeffs.new_tensor(weights)
        basis = torch.stack([shifted_legendre(n, node_t) for n in range(degree)])
        weighted = (basis * weight_t).reshape(degree, degree, *([1] * (evals.ndim - 1)))
        projection = (weighted * evals.unsqueeze(0)).sum(dim=1)
        zeroth = self.increment(s_t, t_t)
        coeffs = torch.cat((zeroth.unsqueeze(0), span * projection[1:]), dim=0)
        return BrownianPolynomial(coeffs=coeffs, s=s_t, t=t_t, degrees=self.degrees)

    def raw_increment(self) -> Tensor:
        """Full-interval increment ``W_t - W_s = I^{(0)}`` (exact)."""
        return self.coeffs[0]

    def coefficients(self) -> Tensor:
        """Raw stacked Legendre coefficients ``I^{(0..N-1)}``.

        Explicit read-only accessor for the underlying coefficient tensor, kept in
        its native ``(N, *sample_shape)`` layout (degree on the leading axis).

        :return: Coefficients of shape ``(N, *sample_shape)``.
        """
        return self.coeffs

    def coefficient_features(self, *, normalize: bool = True) -> Tensor:
        """Network-ready conditioning features built from the Legendre coefficients.

        The degree axis is moved from the front to position ``1`` so the result is
        ``(B, N, *spatial)`` (e.g. ``(N, B, C, H, W) -> (B, N, C, H, W)``). The ``N``
        coefficients form a dedicated feature axis the network can embed/condition
        on (mirroring how the increment terms condition the SSFM drift head).

        When ``normalize`` is set, each coefficient ``I^{(n)}`` is divided by its
        standard deviation ``sqrt((t - s) / (2n + 1))`` to yield unit-variance
        features. Without this, coefficient variance shrinks like ``1 / (2n + 1)``
        and with the interval span, which is poor conditioning for the network.

        :param normalize: Rescale each coefficient to unit variance (default ``True``).
        :return: Features of shape ``(B, N, *spatial)``.
        """
        coeffs = self.coeffs
        if normalize:
            span = (self.t - self.s).clamp_min(0.0)
            if span.ndim:
                span = span.reshape(1, span.shape[0], *([1] * (coeffs.ndim - 2)))
            n = torch.arange(self.degree, device=coeffs.device, dtype=coeffs.dtype)
            n = n.reshape(self.degree, *([1] * (coeffs.ndim - 1)))
            std = torch.sqrt(span / (2.0 * n + 1.0)).clamp_min(torch.finfo(coeffs.dtype).tiny)
            coeffs = coeffs / std
        if self.degrees is not None:
            mask = (
                self.degree_mask()
                .transpose(0, 1)
                .reshape(self.degree, self.coeffs.shape[1], *([1] * (self.coeffs.ndim - 2)))
            )
            coeffs = coeffs * mask.to(dtype=coeffs.dtype)
        # (N, B, *spatial) -> (B, N, *spatial)
        return coeffs.movedim(0, 1)

    def increment(self, u: Tensor, v: Tensor) -> Tensor:
        """Differentiable sub-interval increment ``W_{u,v}^{(N,s,t)}``.

        ``W_{u,v} = sum_n (2n+1)/(t-s) I^{(n)} int_u^v P_tilde_n((r-s)/(t-s)) dr``.

        Differentiable in ``u`` and ``v`` (autograd yields ``-Wdot_u`` and
        ``+Wdot_v`` respectively). At ``u=s, v=t`` returns :meth:`raw_increment`.

        A degenerate parent span (``t == s``, the zero polynomial -- e.g. a
        martingale sampled on a zero-length clock under zeta annealing at
        ``zeta = 0``) is evaluated on a substitute span of one so the zero
        coefficients yield an exact zero instead of ``0 * inf`` NaNs.

        :param u: Sub-interval start, broadcastable with the sample shape.
        :param v: Sub-interval end.
        :return: Increment tensor with the sample shape.
        """
        raw_span = self.t - self.s
        # Substitute span 1 for degenerate (zero-width) intervals; the stored
        # coefficients there are exactly zero, so any finite span gives 0.
        span = torch.where(raw_span > 0, raw_span, torch.ones_like(raw_span))
        sample_ndim = self.coeffs.ndim - 1
        if u.ndim < sample_ndim:
            u = u.reshape(*u.shape, *([1] * (sample_ndim - u.ndim)))
        if v.ndim < sample_ndim:
            v = v.reshape(*v.shape, *([1] * (sample_ndim - v.ndim)))
        ua = ((u - self.s) / span).clamp(0.0, 1.0)
        vb = ((v - self.s) / span).clamp(0.0, 1.0)
        out = torch.zeros_like(self.coeffs[0])
        for n in range(self.degree):
            # int_u^v P_tilde_n((r-s)/span) dr = span * int_ua^vb P_tilde_n(x) dx,
            # so the span Jacobian cancels the (2n+1)/span prefactor, leaving (2n+1).
            integral = integrate_shifted_legendre(n, ua, vb)
            out = out + (2 * n + 1) * self.coeffs[n] * integral
        return out

    def derivative(self, r: Tensor) -> Tensor:
        """White-noise derivative ``Wdot_r^{(N,s,t)}``.

        ``Wdot_r = sum_n (2n+1)/(t-s) I^{(n)} P_tilde_n((r-s)/(t-s))``.

        A degenerate parent span (``t == s``, the zero polynomial -- e.g. a
        martingale sampled on a zero-length clock under zeta annealing at
        ``zeta = 0``) is evaluated on a substitute span of one, so the zero
        coefficients yield an exact zero derivative (``1 / tiny`` would
        overflow to ``inf`` and poison ``inf * 0`` into NaNs).

        :param r: Evaluation time in ``[s, t]``.
        :return: Derivative tensor with the sample shape.
        """
        raw_span = self.t - self.s
        # Substitute span 1 for degenerate (zero-width) intervals; the stored
        # coefficients there are exactly zero, so any finite span gives 0.
        span = torch.where(raw_span > 0, raw_span, torch.ones_like(raw_span))
        sample_ndim = self.coeffs.ndim - 1
        if r.ndim < sample_ndim:
            r = r.reshape(*r.shape, *([1] * (sample_ndim - r.ndim)))
        x = ((r - self.s) / span).clamp(0.0, 1.0)
        out = torch.zeros_like(self.coeffs[0])
        for n in range(self.degree):
            out = out + (2 * n + 1) / span * self.coeffs[n] * shifted_legendre(n, x)
        return out


@dataclass(frozen=True)
class GlobalBrownianPath:
    """One globally anchored polynomial Brownian realization on ``[t0, t1]``.

    Wraps a single :class:`BrownianPolynomial` on the full native-time
    interval ``[t0, t1]``, anchored so ``W_{t0} = 0``. Interval, diagonal, and
    sampling views derived from this object re-express the *same* stored
    coefficients on sub-intervals (:meth:`BrownianPolynomial.restrict`):
    adjacent views share the same global values ``W_r``, their increments
    compose into their parent view's increment, and no view ever samples a
    fresh or zero-span path.

    Moving a view endpoint — for example differentiating the start time inside
    a JVP — keeps the random realization fixed: the restriction tracks the
    moving endpoint analytically while the stored coefficients stay constant,
    so endpoint dependence remains differentiable wherever the view is
    nondegenerate. Callers that instead want fixed conditioning build the
    coefficient features once, outside the differentiated callable.
    """

    path: BrownianPolynomial

    @property
    def degree(self) -> int:
        """Polynomial degree of the global path."""
        return self.path.degree

    @property
    def t0(self) -> Tensor:
        """Anchor time (left endpoint of the realized interval)."""
        return self.path.s

    @property
    def t1(self) -> Tensor:
        """Terminal time (right endpoint of the realized interval)."""
        return self.path.t

    def _as_time(self, value: Tensor | float) -> Tensor:
        """Convert a time to the realization's device and dtype."""
        return torch.as_tensor(value, device=self.path.coeffs.device, dtype=self.path.coeffs.dtype)

    def _check_in_domain(self, r_t: Tensor, *, name: str) -> None:
        """Validate that times lie on the realized interval ``[t0, t1]``."""
        if torch.any(r_t < self.t0):
            raise ValueError(f'{name} require r >= t0; got r < {float(self.t0)}.')
        if torch.any(r_t > self.t1):
            raise ValueError(f'{name} require r <= t1; got r > {float(self.t1)}.')

    def global_value(self, r: Tensor | float) -> Tensor:
        """Return the global value ``W_r`` for ``r in [t0, t1]``.

        Differentiable in ``r``; exactly zero at ``r = t0``.
        """
        r_t = self._as_time(r)
        self._check_in_domain(r_t, name='global values')
        origin = self.path.s.expand_as(r_t)
        return self.path.increment(origin, r_t)

    def derivative(self, r: Tensor | float) -> Tensor:
        """Return the white-noise derivative ``Wdot_r`` of this realization."""
        r_t = self._as_time(r)
        self._check_in_domain(r_t, name='global derivatives')
        return self.path.derivative(r_t)

    def interval(self, s: Tensor | float, t: Tensor | float) -> GlobalBrownianInterval:
        """Return the coherent view of this realization on ``[s, t]``.

        The view's local path is the exact restriction of the global
        polynomial to ``[s, t]``, so its increments, derivative, and features
        keep the established :class:`BrownianPolynomial` semantics while
        remaining coherent with every other view of this realization. For a
        JVP that moves an endpoint, construct the view *inside* the
        differentiated callable so the restriction tracks the endpoint with
        the same stored coefficients. Diagonal evaluation (``s == t``) must
        use :meth:`diagonal`, which retains nondegenerate conditioning from a
        nonzero build window.
        """
        s_t, t_t = self._as_time(s), self._as_time(t)
        if torch.any(s_t < self.t0):
            raise ValueError(f'interval views require s >= t0; got s < {float(self.t0)}.')
        if torch.any(t_t > self.t1):
            raise ValueError(f'interval views require t <= t1; got t > {float(self.t1)}.')
        if torch.any(s_t >= t_t):
            raise ValueError('interval views require s < t; use diagonal() at s == t.')
        return GlobalBrownianInterval(realization=self, path=self.path.restrict(s_t, t_t))

    def diagonal(self, r: Tensor | float, dt: float) -> GlobalBrownianInterval:
        """Return the diagonal-conditioning view at time ``r``.

        The view covers the nondegenerate build window ``[max(r - dt, t0), r]``
        of the same global realization: the field is evaluated at the diagonal
        time ``r`` while its coefficient features come from the build window,
        never from a zero-span path.

        :param r: Diagonal time in ``(t0, t1]``.
        :param dt: Positive build-window length in native time.
        """
        r_t = self._as_time(r)
        dt_f = float(dt)
        if dt_f <= 0.0:
            raise ValueError(f'diagonal views require dt > 0; got {dt_f}.')
        if torch.any(r_t <= self.t0):
            raise ValueError(
                f'diagonal views require r > t0 so the build window is nondegenerate; '
                f'got r <= {float(self.t0)}.'
            )
        if torch.any(r_t > self.t1):
            raise ValueError(f'diagonal views require r <= t1; got r > {float(self.t1)}.')
        return self.interval((r_t - dt_f).clamp_min(self.t0), r_t)


@dataclass(frozen=True)
class GlobalBrownianInterval:
    """Interval view ``[s, t]`` of one globally anchored Brownian realization.

    ``path`` is the exact restriction of the shared realization to ``[s, t]``,
    so local accessors keep the established :class:`BrownianPolynomial`
    semantics — :meth:`increment` is the sub-increment ``W_{u,v}``,
    :meth:`raw_increment` the exact ``W_t - W_s``, and
    :meth:`coefficient_features` the network conditioning — and the view can
    be handed to ``SSFMNet.set_brownian`` directly for sampling. Global
    accessors read the shared realization, which is what makes adjacent views
    share endpoint values and compose into their parent view.
    """

    realization: GlobalBrownianPath
    path: BrownianPolynomial

    @property
    def degree(self) -> int:
        """Polynomial degree of the view path."""
        return self.path.degree

    @property
    def s(self) -> Tensor:
        """Start time of the view."""
        return self.path.s

    @property
    def t(self) -> Tensor:
        """Terminal time of the view."""
        return self.path.t

    def increment(self, u: Tensor, v: Tensor) -> Tensor:
        """Return the sub-increment ``W_{u,v}`` of the shared realization."""
        return self.path.increment(u, v)

    def raw_increment(self) -> Tensor:
        """Return the exact full-view increment ``W_t - W_s``."""
        return self.path.raw_increment()

    def derivative(self, r: Tensor) -> Tensor:
        """Return the white-noise derivative ``Wdot_r`` for ``r in [s, t]``."""
        return self.path.derivative(r)

    def coefficient_features(self, *, normalize: bool = True) -> Tensor:
        """Return view conditioning features as ``(B, N, ...)``."""
        return self.path.coefficient_features(normalize=normalize)

    def global_value(self, r: Tensor | float) -> Tensor:
        """Return the shared global value ``W_r``."""
        return self.realization.global_value(r)

    def global_start(self) -> Tensor:
        """Return the shared global value ``W_s`` at the view start."""
        return self.realization.global_value(self.s)

    def global_terminal(self) -> Tensor:
        """Return the shared global value ``W_t`` at the view terminal."""
        return self.realization.global_value(self.t)


