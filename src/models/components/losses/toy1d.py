"""1D non-linear SDE toy losses: drift matching + (PSD|ESD|LSD) on (WZ|DS|None).

The toy of arXiv:2606.01086 Section 4.1 — ``dX = (X - X^3) dt + sqrt(beta_t)
dW`` with ``beta_t`` linear from ``beta_max`` at ``t = 0`` to ``beta_min`` at
``t = 1`` — transplanted onto the 2D toy's experimental structure:

- **drift matching** (diagonal anchor): the silenced diagonal head is
  regressed against the *known* drift ``f(t, x) = x - x^3`` at simulated
  states ``X_s ~ p_s``;
- **distillation** selected from ``{psd, esd, lsd, discrete_esd,
  discrete_lsd}`` with the 2D form taxonomy on the WZ arm
  (``imf/ayf_emd/ayf_lmd``), the ``ayf_emd``/``imf`` (ESD) and
  ``ayf_lmd`` (LSD) forms on the DS arm (defaulting to the 2D Doss
  conventions), and the non-derivative family on the plain arm
  (``psd`` semigroup and the discrete-AYF stopped-substep variants
  ``discrete_esd``/``discrete_lsd`` — increments only, never the
  driver ``Mdot``);
- **noise handling, shared by every arm**: the *closed-form martingale*
  ``M = int sqrt(beta) dW`` on the quadratic clock
  ``Sigma(t) = int_0^t beta`` (see
  :mod:`~src.models.components.toy_martingale`) supplies the exact
  increment ``M_t - M_s`` inside the map
  ``Psi = x + (t-s) u(x, s, t; I) + (M_t - M_s)`` (an ``euler`` net; no
  diffusion head is ever learned — the noise is known in closed form):

  - ``wz``: Wong-Zakai — the *derivative* formalism: the driver ``Mdot``
    enters every transport/target field (the smoothed RODE
    ``dX/dr = f + Mdot``); identities:
    ``f_s = u - (t-s) D_s u`` (Eulerian), ``partial_t Psi = u_teacher +
    Mdot_t`` (Lagrangian, drivers cancel);
  - ``ds``: Doss-Sussmann — the Z-space RODE ``dZ = f(r, Z + M_r - M_s) dr``
    trains an increment-free ``euler`` map; the physical step reattaches
    ``X_t = Z_t + M_t - M_s``. The 2D Doss-ESD/LSD conventions (learned
    Z-diagonal tangent; EMA endpoint target) are kept verbatim.
  - ``none``: the plain x-space arm — no derivatives and no RODE
    machinery, just increments: the EMA two-half-step semigroup (``psd``)
    and the discrete-AYF stopped-substep compositions.

Training states are simulated ``X_s ~ p_s`` (tamed fine Euler-Maruyama from
``X_0 ~ N(0, init_std^2)``); the batch splits by ``p`` into the distillation
term and the drift-matching anchor, mirroring the 2D toy's fixed-ratio mix.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.func import jvp

from src.models.components.brownian import BrownianPolynomial
from src.models.components.nonlinear_sde import NonlinearSDE
from src.models.components.schedule import expand_t
from src.models.components.ssfm import SSFMNet
from src.models.components.toy_martingale import (
    GlobalToyMartingale,
    sample_global_toy_martingale,
    toy_clock,
)

from .discrete_ayf import (
    deterministic_evaluation,
    per_sample_mse as _ayf_per_sample_mse,
    sample_delta,
)

ToyArmName = Literal['none', 'wz', 'ds']
ToyDistillationName = Literal['psd', 'esd', 'lsd', 'discrete_esd', 'discrete_lsd']
ToyFormName = Literal['imf', 'ayf_emd', 'ayf_lmd']

_Toy_FORMS = ('imf', 'ayf_emd', 'ayf_lmd')
_TOY_DEFAULT_FORM = {'esd': 'ayf_emd', 'lsd': 'ayf_lmd'}


def _per_sample_mse(pred: Tensor, target: Tensor | None = None) -> Tensor:
    """Mean squared error reduced over all but the batch axis -> ``(B,)``."""
    residual = pred if target is None else pred - target
    return residual.square().flatten(1).mean(1)


def _field(value: Tensor, reference: Tensor) -> Tensor:
    """Reshape a per-sample ``(B,)`` value against a batch tensor."""
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def _silenced(net: SSFMNet, x: Tensor) -> Tensor:
    """Zero coefficient features with the net's padded degree width."""
    width = max(int(getattr(net, 'max_degree', 1)), 1)
    return x.new_zeros(x.shape[0], width, *x.shape[1:])


def _require_parameterization(net: SSFMNet, expected: str, arm: str) -> None:
    if net.parameterization != expected:
        raise ValueError(
            f'toy arm {arm!r} requires net.parameterization={expected!r}; '
            f'got {net.parameterization!r}.'
        )


# ---------------------------------------------------------------------------
# WZ arm: increment inside the map, driver in the fields
# ---------------------------------------------------------------------------


def _wz_map(
    net: SSFMNet,
    sde: NonlinearSDE,
    x: Tensor,
    s: Tensor,
    t: Tensor,
    martingale: GlobalToyMartingale,
) -> tuple[Tensor, Tensor]:
    """Evaluate ``Psi = x + (t-s) u(x, s, t; I) + (M_t - M_s)`` and features."""
    interval = martingale.interval(s, t)
    coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
    velocity = net.drift(x, s, t, coefficients)
    psi = x + expand_t(t - s, x) * velocity + interval.raw_increment()
    return psi, interval


def _wz_esd(
    net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    martingale: GlobalToyMartingale,
    form: str,
) -> Tensor:
    """Enforce ``(partial_s + beta_s . grad) Psi = 0`` on the WZ map."""
    driver_s = martingale.driver(s, x_s).detach()
    if form == 'ayf_emd':
        base = sde.drift(x_s, s).detach()
    else:
        base = net.drift(x_s, s, s, _silenced(net, x_s)).detach()
    tangent = base + driver_s

    if form == 'ayf_emd':

        def _map(state: Tensor, start: Tensor) -> Tensor:
            psi, _ = _wz_map(net, sde, state, start, t, martingale)
            return psi

        psi, d_psi = jvp(_map, (x_s, s), (tangent, torch.ones_like(s)))
        weight = expand_t(t - s, psi)
        return -(weight * psi * d_psi.detach()).flatten(1).sum(1)

    def _interval_velocity(state: Tensor, start: Tensor) -> Tensor:
        interval = martingale.interval(start, t)
        coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
        return net.drift(state, start, t, coefficients)

    u, material_derivative = jvp(_interval_velocity, (x_s, s), (tangent, torch.ones_like(s)))
    # Corrected iMF compound for the increment-inside parameterization:
    # f_s = u - (t - s) D_s u, with the material derivative transporting
    # along the full smoothed field f + Mdot.
    compound = u - expand_t(t - s, u) * material_derivative.detach()
    target = sde.drift(x_s, s).detach()
    return _per_sample_mse(compound, target)


def _wz_lsd(
    net: SSFMNet,
    ema_net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    martingale: GlobalToyMartingale,
    form: str,
) -> Tensor:
    """Enforce ``partial_t Psi = u_teacher + Mdot_t`` on the WZ map."""
    driver_t = martingale.driver(t, x_s).detach()

    def _map(end: Tensor) -> Tensor:
        psi, _ = _wz_map(net, sde, x_s, s, end, martingale)
        return psi

    if form == 'ayf_lmd':
        psi, d_psi_dt = jvp(_map, (t,), (torch.ones_like(t),))
        with torch.inference_mode():
            x_t, _ = _wz_map(ema_net, sde, x_s, s, t, martingale)
            velocity = ema_net.drift(x_t, t, t, _silenced(ema_net, x_t))
        residual = d_psi_dt.detach() - (velocity + driver_t)
        weight = expand_t(t - s, psi)
        return (weight * psi * residual).flatten(1).sum(1)

    _, d_psi_dt = jvp(_map, (t,), (torch.ones_like(t),))
    with torch.inference_mode():
        x_t, _ = _wz_map(ema_net, sde, x_s, s, t, martingale)
        velocity = ema_net.drift(x_t, t, t, _silenced(ema_net, x_t))
    target = velocity + driver_t
    return _per_sample_mse(d_psi_dt, target.clone().detach())


# ---------------------------------------------------------------------------
# DS arm: Z-space RODE map, physical step reattaches the increment
# ---------------------------------------------------------------------------


def _ds_esd(
    net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    martingale: GlobalToyMartingale,
    form: str | None = None,
) -> Tensor:
    """Eulerian RODE residual via a JVP through the Z-map.

    Default (2D doss_esd.py convention): learned silenced Z-diagonal
    tangent, plain MSE. Optional forms:

    - ``'ayf_emd'``: transport with the *known* RODE drift ``b(s, X_s)``
      (locally anchored, no driver in Z-space) and the AYF-EMD weighted
      residual ``-(t - s) Xi D_s Xi``.
    - ``'imf'`` (2D ds_imf.py): fixed-coefficient conditioning with the
      learned diagonal tangent read off the interval's features; the
      compound ``u - h sg(D_s u)`` regresses onto the known ``b(s, X_s)``.
    """
    z_s = x_s  # local anchor: Z_s^s = X_s.

    def _step(z: Tensor, start: Tensor) -> Tensor:
        interval = martingale.interval(start, t)
        coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
        return net(z, start, t, coefficients)

    if form == 'ayf_emd':
        tangent = sde.drift(x_s, s).detach()
        xi, d_xi = jvp(_step, (z_s, s), (tangent, torch.ones_like(s)))
        weight = expand_t(t - s, xi)
        return -(weight * xi * d_xi.detach()).flatten(1).sum(1)

    if form == 'imf':
        interval = martingale.interval(s, t)
        coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
        diagonal_tangent = net.drift(z_s, s, s, coefficients).detach()
        beta_s = sde.drift(x_s, s).detach()

        def _interval_average(z: Tensor, start: Tensor) -> Tensor:
            return net.drift(z, start, t, coefficients)

        average, material_derivative = jvp(
            _interval_average, (z_s, s), (diagonal_tangent, torch.ones_like(s))
        )
        compound = average - expand_t(t - s, x_s) * material_derivative.detach()
        return _per_sample_mse(compound - beta_s)

    learned_drift = net.drift(z_s, s, s, _silenced(net, z_s)).detach()
    _, total_deriv = jvp(_step, (z_s, s), (learned_drift, torch.ones_like(s)))
    return _per_sample_mse(total_deriv)


def _ds_lsd(
    net: SSFMNet,
    ema_net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    martingale: GlobalToyMartingale,
    form: str | None = None,
) -> Tensor:
    """Lagrangian RODE residual: ``partial_t Xi = b(t, Xi + M_t - M_s)``.

    Default (2D doss_lsd.py convention): plain MSE against the EMA
    teacher's drift read at the reattached physical point. Optional
    ``'ayf_lmd'`` form: the AYF-LMD weighted residual
    ``(t - s) Xi (sg(d Xi/dt) - b_t)`` with the gradient flowing through
    the student's map only.
    """
    del sde
    z_s = x_s

    def _step(end: Tensor) -> Tensor:
        interval = martingale.interval(s, end)
        coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
        return net(z_s, s, end, coefficients)

    xi, d_xi_dt = jvp(_step, (t,), (torch.ones_like(t),))
    with torch.inference_mode():
        interval = martingale.interval(s, t)
        coefficients = interval.coefficient_features(normalize=ema_net.normalize_coefficients)
        xi_ema = ema_net(z_s, s, t, coefficients)
        x_t = xi_ema + interval.raw_increment()
        drift = ema_net.drift(x_t, t, t, _silenced(ema_net, x_t))
    if form == 'ayf_lmd':
        residual = d_xi_dt.detach() - drift.clone().detach()
        weight = expand_t(t - s, xi)
        return (weight * xi * residual).flatten(1).sum(1)
    return _per_sample_mse(d_xi_dt - drift.clone().detach())


# ---------------------------------------------------------------------------
# plain arm: closed-form increments, no derivatives, no learned noise head
# ---------------------------------------------------------------------------


def _none_discrete_esd(
    net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
    dt: float,
) -> Tensor:
    """Full map vs stopped local drift step + current-net map over the rest.

    Discrete-AYF ESD on the closed-form-increment map: the stopped leading
    substep rides the net's diagonal drift with the *exact* increment
    ``M_split - M_s``; the remainder is the current net's map with
    ``M_t - M_split``. The student's one full step regresses onto the
    stop-gradient composition, ``h^{-2}``-weighted.
    """
    split = s + sample_delta(s, t, dt)
    leading, trailing, parent = _toy_clock_split(sde, x_s, s, split, t, degree)
    with deterministic_evaluation(net):
        with torch.inference_mode():
            u_diag = net.drift(x_s, s, s, _silenced(net, x_s))
            x_split = x_s + expand_t(split - s, x_s) * u_diag + leading.raw_increment()
            trailing_coeffs = trailing.coefficient_features(normalize=net.normalize_coefficients)
            target = (
                x_split
                + expand_t(t - split, x_split)
                * net.drift(x_split, split, t, trailing_coeffs)
                + trailing.raw_increment()
            ).clone()
        parent_coeffs = parent.coefficient_features(normalize=net.normalize_coefficients)
        prediction = (
            x_s
            + expand_t(t - s, x_s) * net.drift(x_s, s, t, parent_coeffs)
            + parent.raw_increment()
        )
    h = (t - s).clamp_min(torch.finfo(t.dtype).eps)
    return _ayf_per_sample_mse(prediction - target.detach()) / (h**2)


def _none_discrete_lsd(
    net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
    dt: float,
) -> Tensor:
    """Full map vs current-net shorter map + stopped trailing drift step."""
    split = t - sample_delta(s, t, dt)
    leading, trailing, parent = _toy_clock_split(sde, x_s, s, split, t, degree)
    with deterministic_evaluation(net):
        with torch.inference_mode():
            leading_coeffs = leading.coefficient_features(normalize=net.normalize_coefficients)
            x_split = (
                x_s
                + expand_t(split - s, x_s) * net.drift(x_s, s, split, leading_coeffs)
                + leading.raw_increment()
            )
            u_diag = net.drift(x_split, split, split, _silenced(net, x_split))
            target = (
                x_split
                + expand_t(t - split, x_split)
                * u_diag
                + trailing.raw_increment()
            ).clone()
        parent_coeffs = parent.coefficient_features(normalize=net.normalize_coefficients)
        prediction = (
            x_s
            + expand_t(t - s, x_s) * net.drift(x_s, s, t, parent_coeffs)
            + parent.raw_increment()
        )
    h = (t - s).clamp_min(torch.finfo(t.dtype).eps)
    return _ayf_per_sample_mse(prediction - target.detach()) / (h**2)


# ---------------------------------------------------------------------------
# PSD per arm: EMA teacher two half-steps vs student one full step
# ---------------------------------------------------------------------------


def _toy_clock_split(
    sde: NonlinearSDE, x_s: Tensor, s: Tensor, split: Tensor, t: Tensor, degree: int
) -> tuple[BrownianPolynomial, BrownianPolynomial, BrownianPolynomial]:
    """Chen-consistent clock split-paths and parent (arbitrary split)."""
    shape = tuple(x_s.shape)
    field_ones = (1,) * (x_s.ndim - 1)
    clock_s = toy_clock(sde, s.view(-1, *field_ones))
    clock_split = toy_clock(sde, split.view(-1, *field_ones))
    clock_t = toy_clock(sde, t.view(-1, *field_ones))
    leading = BrownianPolynomial.sample(
        shape, clock_s, clock_split, degree, device=x_s.device, dtype=x_s.dtype
    )
    trailing = BrownianPolynomial.sample(
        shape, clock_split, clock_t, degree, device=x_s.device, dtype=x_s.dtype
    )
    return leading, trailing, BrownianPolynomial.concat_arbitrary(leading, trailing)


def _toy_clock_halves(
    sde: NonlinearSDE, x_s: Tensor, s: Tensor, t: Tensor, degree: int
) -> tuple[BrownianPolynomial, BrownianPolynomial, BrownianPolynomial]:
    """Chen-consistent clock half-paths and parent (unequal clock spans)."""
    m = 0.5 * (s + t)
    return _toy_clock_split(sde, x_s, s, m, t, degree)


def _increment_psd(
    net: SSFMNet,
    ema_net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
) -> Tensor:
    """EMA two half-steps vs one student full step on the increment map.

    PSD never touches the driver ``Mdot`` — it is arm-agnostic, so the
    WZ and plain arms share this single implementation.
    """
    left, right, parent = _toy_clock_halves(sde, x_s, s, t, degree)
    with torch.inference_mode():
        left_coeffs = left.coefficient_features(normalize=ema_net.normalize_coefficients)
        right_coeffs = right.coefficient_features(normalize=ema_net.normalize_coefficients)
        m = 0.5 * (s + t)
        x_mid = (
            x_s
            + expand_t(m - s, x_s) * ema_net.drift(x_s, s, m, left_coeffs)
            + left.raw_increment()
        )
        x_teacher = (
            x_mid
            + expand_t(t - m, x_mid) * ema_net.drift(x_mid, m, t, right_coeffs)
            + right.raw_increment()
        )
    target = x_teacher.clone().detach()
    parent_coeffs = parent.coefficient_features(normalize=net.normalize_coefficients)
    x_student = (
        x_s + expand_t(t - s, x_s) * net.drift(x_s, s, t, parent_coeffs) + parent.raw_increment()
    )
    residual = _per_sample_mse(x_student - target)
    return residual


def _ds_psd(
    net: SSFMNet,
    ema_net: SSFMNet,
    sde: NonlinearSDE,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
) -> Tensor:
    """EMA two Z-half-steps (with reattachment) vs one student full step."""
    left, right, parent = _toy_clock_halves(sde, x_s, s, t, degree)
    m = 0.5 * (s + t)
    with torch.inference_mode():
        left_coeffs = left.coefficient_features(normalize=ema_net.normalize_coefficients)
        right_coeffs = right.coefficient_features(normalize=ema_net.normalize_coefficients)
        z_mid = ema_net(x_s, s, m, left_coeffs)
        x_mid = z_mid + left.raw_increment()
        z_teacher = ema_net(x_mid, m, t, right_coeffs)
        x_teacher = z_teacher + right.raw_increment()
    target = x_teacher.clone().detach()
    parent_coeffs = parent.coefficient_features(normalize=net.normalize_coefficients)
    z_student = net(x_s, s, t, parent_coeffs)
    x_student = z_student + parent.raw_increment()
    return _per_sample_mse(x_student - target)


# ---------------------------------------------------------------------------
# mixer
# ---------------------------------------------------------------------------


class Toy1DSSFMLoss(nn.Module):
    """Drift matching plus a selected (PSD|ESD|LSD) distillation on (WZ|DS|None).

    :param arm: Training-objective arm — every arm rides the same
        closed-form-increment map ``Psi = x + (t-s) u + (M_t - M_s)``
        (an ``euler`` net; no diffusion head is ever learned).
        ``'wz'``: the derivative formalism (driver ``Mdot`` in the fields);
        ``'ds'``: the Z-space RODE map + reattachment;
        ``'none'``: the plain non-derivative family (``psd`` and the
        discrete-AYF stopped-substep variants — increments only).
    :param distillation: ``'psd'`` | ``'esd'`` | ``'lsd'`` |
        ``'discrete_esd'`` | ``'discrete_lsd'``. The discrete AYF variants
        are plain-arm only: the full map regresses onto a stopped local
        drift substep (with the exact increment) composed with the
        current-net map over the remainder.
    :param form: Optional gradient structure for the derivative arms:
        WZ ESD ``'ayf_emd'`` (default) | ``'imf'``; DS ESD ``None`` (2D
        Doss convention) | ``'ayf_emd'`` | ``'imf'``; LSD ``'ayf_lmd'``
        (or ``None`` on DS for the 2D convention). ``psd`` and the
        discrete variants take no form.
    :param p: Distillation fraction of each batch (remainder: drift matching).
    :param degree: Polynomial degree ``N`` of the closed-form martingale.
    :param delta_t: Minimum distillation step (also caps the simulated
        start times so ``s <= 1 - delta_t``); for the discrete variants it
        additionally scales the local EM substep.
    :param sim_steps: Fine tamed-EM intervals for ``X_s ~ p_s``.
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        init_std: float = 1.0,
        arm: ToyArmName = 'wz',
        distillation: ToyDistillationName = 'esd',
        form: ToyFormName | None = None,
        p: float = 0.75,
        degree: int = 3,
        delta_t: float = 0.05,
        sim_steps: int = 128,
    ) -> None:
        super().__init__()
        if arm not in ('none', 'wz', 'ds'):
            raise ValueError(f'Unknown toy arm {arm!r}; expected one of (none, wz, ds).')
        if distillation not in ('psd', 'esd', 'lsd', 'discrete_esd', 'discrete_lsd'):
            raise ValueError(
                f'Unknown toy distillation {distillation!r}; expected one of '
                '(psd, esd, lsd, discrete_esd, discrete_lsd).'
            )
        if distillation.startswith('discrete') and arm != 'none':
            raise ValueError(
                f'{distillation!r} is a plain-arm (non-derivative) variant; '
                f'got arm={arm!r}.'
            )
        if distillation.startswith('discrete') and form is not None:
            raise ValueError('the discrete variants have no forms; leave form unset.')
        if arm == 'none' and distillation in ('esd', 'lsd'):
            raise ValueError(
                'the plain arm carries only the non-derivative family '
                '(psd, discrete_esd, discrete_lsd); the derivative modes '
                'live on the wz/ds arms.'
            )
        if (
            form is None
            and arm == 'wz'
            and distillation in ('esd', 'lsd')
        ):
            form = _TOY_DEFAULT_FORM[distillation]
        if form is not None and form not in _Toy_FORMS:
            raise ValueError(f'Unknown toy form {form!r}; expected one of {list(_Toy_FORMS)}.')
        if arm == 'none' and form is not None:
            raise ValueError(f'Toy arm {arm!r} has no form taxonomy; leave form unset.')
        if arm == 'wz' and distillation == 'esd' and form == 'ayf_lmd':
            raise ValueError('ayf_lmd is Lagrangian only; use distillation="lsd".')
        if arm == 'wz' and distillation == 'lsd' and form in ('imf', 'ayf_emd'):
            raise ValueError(f'{form!r} is Eulerian only; use distillation="esd".')
        if arm == 'ds' and distillation == 'esd' and form not in (None, 'ayf_emd', 'imf'):
            raise ValueError(
                f'{arm} ESD accepts no form or one of (ayf_emd, imf); '
                f'got {form!r}.'
            )
        if arm == 'ds' and distillation == 'lsd' and form not in (None, 'ayf_lmd'):
            raise ValueError(f'{arm} LSD accepts no form or ayf_lmd; got {form!r}.')
        if distillation == 'psd' and form is not None:
            raise ValueError('psd has no gradient-structure forms; leave form unset.')
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1]; got {p}.')
        if degree < 1:
            raise ValueError(f'degree must be >= 1; got {degree}.')
        if not 0.0 < delta_t < 1.0:
            raise ValueError(f'delta_t must lie in (0, 1); got {delta_t}.')
        if sim_steps < 1:
            raise ValueError(f'sim_steps must be >= 1; got {sim_steps}.')
        self.sde = NonlinearSDE(beta_min=beta_min, beta_max=beta_max, init_std=init_std)
        self.arm = arm
        self.distillation_name = distillation
        self.form: ToyFormName | None = form
        self.p = p
        self.degree = degree
        self.delta_t = delta_t
        self.sim_steps = sim_steps

    def _require_net(self, net: SSFMNet) -> None:
        _require_parameterization(net, 'euler', self.arm)

    def _distill(
        self, net: SSFMNet, ema_net: SSFMNet, x_s: Tensor, s: Tensor, t: Tensor
    ) -> Tensor:
        sde = self.sde
        if self.arm == 'none':
            # Non-derivative family: closed-form increments only.
            if self.distillation_name == 'discrete_esd':
                return _none_discrete_esd(net, sde, x_s, s, t, self.degree, self.delta_t)
            if self.distillation_name == 'discrete_lsd':
                return _none_discrete_lsd(net, sde, x_s, s, t, self.degree, self.delta_t)
            return _increment_psd(net, ema_net, sde, x_s, s, t, self.degree)

        martingale = sample_global_toy_martingale(
            sde, tuple(x_s.shape), self.degree, device=x_s.device, dtype=x_s.dtype
        )
        if self.arm == 'wz':
            if self.distillation_name == 'esd':
                return _wz_esd(net, sde, x_s, s, t, martingale, self.form or 'ayf_emd')
            if self.distillation_name == 'lsd':
                return _wz_lsd(net, ema_net, sde, x_s, s, t, martingale, self.form or 'ayf_lmd')
            return _increment_psd(net, ema_net, sde, x_s, s, t, self.degree)

        if self.distillation_name == 'esd':
            return _ds_esd(net, sde, x_s, s, t, martingale, self.form)
        if self.distillation_name == 'lsd':
            return _ds_lsd(net, ema_net, sde, x_s, s, t, martingale, self.form)
        return _ds_psd(net, ema_net, sde, x_s, s, t, self.degree)

    def _diagonal(self, net: SSFMNet, x_s: Tensor, s: Tensor) -> Tensor:
        """Drift matching: the silenced diagonal head regresses ``f(s, x)``."""
        target = self.sde.drift(x_s, s).detach()
        prediction = net.drift(x_s, s, s, _silenced(net, x_s))
        return _per_sample_mse(prediction - target)

    def _sample_times(self, batch: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Sample ``t ~ U[delta_t, 1]`` then ``h ~ U[delta_t, min(1, t)]``."""
        t = self.delta_t + (1.0 - self.delta_t) * torch.rand(
            batch, device=reference.device, dtype=torch.float32
        )
        h = self.delta_t + (t - self.delta_t) * torch.rand_like(t)
        return t - h, t

    def forward(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the mixed loss and scalar diagnostics."""
        del global_step
        self._require_net(net)
        batch = x_0.shape[0]
        n_distill = int(batch * self.p)
        max_index = max(1, int((1.0 - self.delta_t) * self.sim_steps))
        x_s, s_sim, _ = self.sde.sample_marginals(x_0, self.sim_steps, max_index=max_index)

        if n_distill:
            s, t = self._sample_times(n_distill, x_1)
            distill = self._distill(net, ema_net, x_s[:n_distill], s, t).mean()
        else:
            distill = x_1.new_zeros(())

        if n_distill < batch:
            diagonal = self._diagonal(net, x_s[n_distill:], s_sim[n_distill:]).mean()
        else:
            diagonal = x_1.new_zeros(())

        total = self.p * distill + (1.0 - self.p) * diagonal
        return total, {
            'diagonal_loss': diagonal.detach(),
            'distillation_loss': distill.detach(),
            'effective_p': total.new_tensor(self.p),
            'n_distill': total.new_tensor(n_distill),
        }
