"""Pathwise strong-error evaluation against a fine Euler--Maruyama reference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .brownian import BrownianPolynomial
from .ei_doss import ei_doss_map, ei_doss_reconstruct
from .ei_euler import EIEulerPath, ei_euler_map
from .schedule import AffineSchedule, expand_t
from .ssfm import SSFMNet


@dataclass(frozen=True)
class PiecewiseBrownianPath:
    """One physical Brownian realization represented on a fine uniform grid."""

    nodes: Tensor
    leaves: list[BrownianPolynomial]

    def interval(self, start: Tensor, end: Tensor) -> BrownianPolynomial:
        """Project the piecewise realization onto one arbitrary interval."""
        pieces = []
        start_value, end_value = float(start), float(end)
        for index, leaf in enumerate(self.leaves):
            left, right = float(self.nodes[index]), float(self.nodes[index + 1])
            lo, hi = max(start_value, left), min(end_value, right)
            if hi > lo:
                pieces.append(leaf.restrict(lo, hi))
        if not pieces:
            raise ValueError('Requested Brownian interval has no positive overlap.')
        result = pieces[0]
        for piece in pieces[1:]:
            result = BrownianPolynomial.concat_arbitrary(result, piece)
        return result


def sample_piecewise_brownian(
    shape: tuple[int, ...],
    degree: int,
    steps: int,
    t0: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> PiecewiseBrownianPath:
    """Sample independent polynomial Brownian leaves on a fine physical grid."""
    nodes = torch.linspace(t0, 1.0, steps + 1, device=device, dtype=dtype)
    leaves = [
        BrownianPolynomial.sample(
            shape,
            nodes[index],
            nodes[index + 1],
            degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for index in range(steps)
    ]
    return PiecewiseBrownianPath(nodes, leaves)


def sample_piecewise_ei_clock(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    degree: int,
    steps: int,
    t0: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    zeta: float = 1.0,
) -> PiecewiseBrownianPath:
    """Sample independent martingale leaves on the (annealed) EI weighted clock."""
    nodes = torch.linspace(t0, 1.0, steps + 1, device=device, dtype=dtype)
    leaves = [
        BrownianPolynomial.sample(
            shape,
            schedule.annealed_ei_weighted_clock(nodes[index], t0, zeta),
            schedule.annealed_ei_weighted_clock(nodes[index + 1], t0, zeta),
            degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for index in range(steps)
    ]
    return PiecewiseBrownianPath(nodes, leaves)


def _scale_ei_leaf(
    leaf: BrownianPolynomial,
    schedule: AffineSchedule,
    start: Tensor,
    end: Tensor,
    field: Tensor,
    t0: float,
) -> EIEulerPath:
    """Scale one unscaled clock leaf into a local EI-Euler path."""
    batch = field.shape[0]
    field_ones = (1,) * (field.ndim - 1)
    s = start.expand(batch).view(-1, *field_ones)
    t = end.expand(batch).view(-1, *field_ones)
    return EIEulerPath(
        coeffs=schedule.ei_martingale_scale(t).unsqueeze(0) * leaf.coeffs,
        normalized_coeffs=leaf.coefficient_features(normalize=True).movedim(1, 0),
        variance=schedule.ei_driver_variance(s, t, t0=t0),
        degrees=leaf.degrees,
    )


def coupled_ei_euler_path(
    realization: PiecewiseBrownianPath,
    schedule: AffineSchedule,
    start: Tensor,
    end: Tensor,
    field: Tensor,
    t0: float,
) -> EIEulerPath:
    """Compose overlapping EI-clock leaves into one scaled local ``G`` path."""
    pieces = []
    start_value, end_value = float(start), float(end)
    for index, leaf in enumerate(realization.leaves):
        left, right = float(realization.nodes[index]), float(realization.nodes[index + 1])
        lo, hi = max(start_value, left), min(end_value, right)
        if hi <= lo:
            continue
        clock_lo = schedule.ei_weighted_clock(start.new_tensor(lo), t0)
        clock_hi = schedule.ei_weighted_clock(start.new_tensor(hi), t0)
        # The clock evaluation is not guaranteed monotone under float
        # rounding (nearly-flat regions); a non-positive span is a zero
        # increment, so drop the piece instead of failing restrict.
        if float(clock_hi - clock_lo) <= 0.0:
            continue
        pieces.append(leaf.restrict(clock_lo, clock_hi))
    if not pieces:
        raise ValueError('Requested EI interval has no positive overlap.')
    parent = pieces[0]
    for piece in pieces[1:]:
        parent = BrownianPolynomial.concat_arbitrary(parent, piece)
    return _scale_ei_leaf(parent, schedule, start, end, field, t0)


def learned_physical_drift(
    net: SSFMNet,
    schedule: AffineSchedule,
    state: Tensor,
    time: Tensor,
    *,
    kind: str,
    zeta: float = 1.0,
) -> Tensor:
    """Physical SDE drift ``b_θ(t, X)`` from the learned diagonal representation.

    EI-Euler and EI-Doss both predict flow-matching velocity ``u`` on the
    diagonal (their maps own the interval weighting), so both convert to the
    annealed Holderrieth drift ``b^ζ = (1+ζ²)u - ζ²(α̇/α) X`` (``ζ = 1`` is the
    un-annealed ``2u - (α̇/α) X``). Plain x-space SSFM already predicts ``b``
    directly and is never annealed here. Brownian
    features are silenced (``I = 0``), matching training.
    """
    zeros = state.new_zeros(state.shape[0], 1, *state.shape[1:])
    if kind in ('ei_euler', 'ei_doss'):
        diagonal = getattr(net, 'diagonal_prediction', None)
        velocity = (
            diagonal(state, time) if diagonal is not None else net.drift(state, time, time, zeros)
        )
        ratio = expand_t(schedule.alpha_dot(time) / schedule.alpha(time), state)
        return (1.0 + zeta * zeta) * velocity - (zeta * zeta) * ratio * state
    if kind == 'ssfm':
        return net.drift(state, time, time, zeros)
    raise ValueError(f'unknown diagonal kind {kind!r}')


def learned_diffusion(
    net: SSFMNet,
    schedule: AffineSchedule,
    state: Tensor,
    time: Tensor,
) -> Tensor:
    """Diffusion coefficient for the EM reference at ``(t, X)``.

    Under ``euler_maruyama`` the backbone owns a diffusion head trained against
    ``nu_t`` (see :func:`~...losses.flow_matching.instantaneous_loss`), so the
    reference must use the *learned* field rather than the analytic one --
    otherwise the reference is not a solve of the model the run actually
    trained. Under ``euler`` the sampler owns the noise and the closed-form
    ``nu = sqrt(nu_squared(t))`` is exact.
    """
    nu = expand_t(torch.sqrt(schedule.nu_squared(time).clamp_min(0.0)), state)
    if net.parameterization != 'euler_maruyama':
        return nu
    zeros = state.new_zeros(state.shape[0], 1, *state.shape[1:])
    _, diffusion = net.fields(state, time, time, zeros)
    return nu if diffusion is None else diffusion


def physical_em_rollout(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    realization: PiecewiseBrownianPath,
    *,
    kind: str,
) -> Tensor:
    """Euler--Maruyama the learned diagonal SDE on a piecewise physical Brownian path.

    Each leaf takes ``X ← X + b_θ(t, X) Δt + g(t, X) ΔW`` at the left endpoint,
    where ``g`` is the closed-form ``ν = sqrt(nu_squared(t))`` for ``euler``
    backbones and the learned diffusion head for ``euler_maruyama`` ones
    (see :func:`learned_diffusion`).
    """
    state = latents.clone()
    batch = latents.shape[0]
    for index, leaf in enumerate(realization.leaves):
        start, end = realization.nodes[index], realization.nodes[index + 1]
        time = start.expand(batch)
        drift = learned_physical_drift(net, schedule, state, time, kind=kind)
        diffusion = learned_diffusion(net, schedule, state, time)
        state = state + (end - start) * drift + diffusion * leaf.raw_increment()
    return state


def interval_martingale_increment(
    realization: PiecewiseBrownianPath,
    schedule: AffineSchedule,
    start: Tensor,
    end: Tensor,
    field: Tensor,
    *,
    ei_weighted: bool,
) -> Tensor:
    """Return a Doss-clock increment over ``[start, end]``.

    The realization's leaves must already live on the selected martingale
    clock, so their full coefficient vectors can be restricted and composed
    without dropping higher path modes.
    """
    return martingale_interval_path(
        realization, schedule, start, end, ei_weighted=ei_weighted
    ).raw_increment()


def martingale_interval_path(
    realization: PiecewiseBrownianPath,
    schedule: AffineSchedule,
    start: Tensor,
    end: Tensor,
    *,
    ei_weighted: bool,
    zeta: float = 1.0,
) -> BrownianPolynomial:
    """Compose a physical-time interval from leaves sampled on a Doss clock."""
    pieces = []
    start_value, end_value = float(start), float(end)
    t0 = float(realization.nodes[0])

    def clock(value: Tensor) -> Tensor:
        if ei_weighted:
            return schedule.annealed_ei_weighted_clock(value, t0, zeta)
        return schedule.martingale_clock(value, t0)

    start_t = realization.leaves[0].coeffs.new_tensor(start_value)
    end_t = realization.leaves[0].coeffs.new_tensor(end_value)
    if float(clock(end_t) - clock(start_t)) <= 0.0:
        # Degenerate clock span (zeta = 0 under annealing): the martingale is
        # identically zero, so every interval view is the zero polynomial.
        zero_s = clock(start_t)
        return BrownianPolynomial(
            coeffs=realization.leaves[0].coeffs.new_zeros(realization.leaves[0].coeffs.shape),
            s=zero_s,
            t=zero_s + zero_s.new_zeros(()),
        )

    for index, leaf in enumerate(realization.leaves):
        left, right = float(realization.nodes[index]), float(realization.nodes[index + 1])
        lo, hi = max(start_value, left), min(end_value, right)
        if hi <= lo:
            continue
        if lo == left and hi == right:
            piece = leaf
        else:
            lo_t = leaf.coeffs.new_tensor(lo)
            hi_t = leaf.coeffs.new_tensor(hi)
            # Clamp the freshly evaluated clock values into the leaf's own
            # (monotone-clamped) span: adjacent pieces then share endpoints
            # bit-for-bit, and a nearly-flat annealed clock (small zeta) can
            # only round a span to zero — a zero increment (the exact
            # zeta -> 0 limit), dropped below.
            lo_c = clock(lo_t).clamp(min=leaf.s, max=leaf.t)
            hi_c = clock(hi_t).clamp(min=leaf.s, max=leaf.t)
            # Guard before restrict: the clock evaluation need not be
            # monotone under float rounding, and restrict demands a strictly
            # positive child span. A non-positive span is a zero increment.
            if float(hi_c - lo_c) <= 0.0:
                continue
            piece = leaf.restrict(lo_c, hi_c)
        if float(piece.t - piece.s) > 0.0:
            pieces.append(piece)
    if not pieces:
        # Every overlapping piece rounded to a zero span: the interval
        # increment is numerically zero, same as the degenerate-clock branch.
        zero_s = clock(realization.leaves[0].coeffs.new_tensor(start_value))
        return BrownianPolynomial(
            coeffs=realization.leaves[0].coeffs.new_zeros(realization.leaves[0].coeffs.shape),
            s=zero_s,
            t=zero_s + zero_s.new_zeros(()),
        )
    result = pieces[0]
    for piece in pieces[1:]:
        result = BrownianPolynomial.concat_arbitrary(result, piece)
    return result


def _ei_path_from_g(
    g: Tensor,
    schedule: AffineSchedule,
    start: Tensor,
    end: Tensor,
    *,
    t0: float,
    degree: int,
) -> EIEulerPath:
    """Wrap a coupled scaled increment ``G = ΔM / α_t`` as an EI-Euler path."""
    batch = g.shape[0]
    field_ones = (1,) * (g.ndim - 1)
    time_s = start.reshape(()).expand(batch).view(-1, *field_ones)
    time_t = end.reshape(()).expand(batch).view(-1, *field_ones)
    variance = schedule.ei_driver_variance(time_s, time_t, t0=t0)
    coeffs = g.new_zeros(degree, *g.shape)
    normalized = g.new_zeros(degree, *g.shape)
    coeffs[0] = g
    std = variance.clamp_min(0.0).sqrt()
    normalized[0] = g / std.clamp_min(torch.finfo(g.dtype).tiny)
    return EIEulerPath(coeffs=coeffs, normalized_coeffs=normalized, variance=variance)


@torch.no_grad()
def ei_doss_rollout_batch(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    realization: PiecewiseBrownianPath,
    step_counts: list[int],
    *,
    fine_steps: int,
    shift: float = 1.0,
    zeta: float = 1.0,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Return EM reference and coarse (annealed) EI-Doss endpoints on one path.

    Fine: the annealed lifted reference
    ``Y ← Y + (1+ζ²) α^{ζ²} u_θ(X) Δt + ΔM~^ζ`` (``X = Y / α^{ζ²}``) on the
    annealed EI-clock leaves. Coarse: the annealed EI-Euler skeleton on
    ``Z_s = X_s`` then ``X_t = Z_t + ΔM~^ζ / α_t^{ζ²}``. ``zeta = 0`` rolls
    out the probability-flow ODE; ``zeta = 1`` (default) is the un-annealed
    EI-Doss pathwise diagnostic.
    """
    del fine_steps, shift
    t0 = float(realization.nodes[0])
    dtype, device = latents.dtype, latents.device
    normalize = getattr(net, 'normalize_coefficients', False)
    ones = torch.ones(latents.shape[0], device=device, dtype=dtype)
    power = zeta * zeta

    def lift(time: Tensor) -> Tensor:
        return schedule.alpha(time) ** power

    def _rollout(num_steps: int) -> Tensor:
        grid = torch.linspace(t0, 1.0, num_steps + 1, device=device, dtype=dtype)
        state = latents
        for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
            s_b = s_cur.expand_as(ones)
            t_b = t_nxt.expand_as(ones)
            path = martingale_interval_path(
                realization, schedule, s_cur, t_nxt, ei_weighted=True, zeta=zeta
            )
            delta_m = path.raw_increment()
            features = path.coefficient_features(normalize=normalize)
            z_t = ei_doss_map(net, schedule, state, s_b, t_b, features, zeta)
            state = ei_doss_reconstruct(schedule, z_t, t_b, delta_m, zeta)
        return state.to(dtype)

    reference_y = expand_t(lift(realization.nodes[0]), latents) * latents
    for index, leaf in enumerate(realization.leaves):
        start, end = realization.nodes[index], realization.nodes[index + 1]
        time = start.expand_as(ones)
        reference = reference_y / expand_t(lift(start), reference_y)
        drift = learned_physical_drift(net, schedule, reference, time, kind='ei_doss', zeta=zeta)
        y_drift = (zeta * zeta) * expand_t(
            schedule.alpha_dot(start) / schedule.alpha(start), reference_y
        ) * reference_y + expand_t(lift(start), reference) * drift
        reference_y = reference_y + (end - start) * y_drift + leaf.raw_increment()
    reference = reference_y / expand_t(lift(realization.nodes[-1]), reference_y)
    return reference, {steps: _rollout(steps) for steps in step_counts}


@torch.no_grad()
def strong_rollout_batch(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    realization: PiecewiseBrownianPath,
    step_counts: list[int],
    *,
    ei_euler: bool = True,
    t0: float = 0.01,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Return fine and coarse EI-Euler endpoints on one EI-clock realization.

    Every particle has its own path, reused across all step counts. Restriction
    and composition preserve the complete configured polynomial degree.
    """
    if not ei_euler:
        raise NotImplementedError(
            'strong_rollout_batch currently supports EI-Euler maps only; physical-EM '
            'coupling for other parameterizations is not defined here.'
        )
    dtype, device = latents.dtype, latents.device
    batch = latents.shape[0]
    reference = latents.clone()
    for index, leaf in enumerate(realization.leaves):
        start, end = realization.nodes[index], realization.nodes[index + 1]
        path = _scale_ei_leaf(leaf, schedule, start, end, reference, t0)
        reference = ei_euler_map(
            net, schedule, reference, start.expand(batch), end.expand(batch), path
        )
    endpoints: dict[int, Tensor] = {}
    for steps in step_counts:
        state = latents.clone()
        boundaries = torch.linspace(
            float(realization.nodes[0]), 1.0, steps + 1, device=device, dtype=dtype
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            path = coupled_ei_euler_path(realization, schedule, start, end, state, t0)
            state = ei_euler_map(
                net, schedule, state, start.expand(batch), end.expand(batch), path
            )
        endpoints[steps] = state
    return reference, endpoints


@torch.no_grad()
def strong_error_batch(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    realization: PiecewiseBrownianPath,
    step_counts: list[int],
    *,
    ei_euler: bool = False,
    t0: float = 0.01,
) -> Tensor:
    """Return one pathwise terminal MSE for every requested flow-map step count."""
    dtype, device = latents.dtype, latents.device
    batch, degree = latents.shape[0], realization.leaves[0].degree
    if ei_euler:
        reference = physical_em_rollout(net, schedule, latents, realization, kind='ei_euler')
    else:
        reference = latents.clone()
        for index, leaf in enumerate(realization.leaves):
            start, end = realization.nodes[index], realization.nodes[index + 1]
            time = start.expand(batch)
            features = reference.new_zeros(batch, degree, *reference.shape[1:])
            drift = net.drift(reference, time, time, features)
            dt = end - start
            diffusion = expand_t(torch.sqrt(schedule.nu_squared(time).clamp_min(0.0)), reference)
            reference = reference + dt * drift + diffusion * leaf.raw_increment()

    errors = []
    for steps in step_counts:
        state = latents.clone()
        boundaries = torch.linspace(
            float(realization.nodes[0]), 1.0, steps + 1, device=device, dtype=dtype
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            time_s, time_t = start.expand(batch), end.expand(batch)
            if ei_euler:
                delta_m = interval_martingale_increment(
                    realization, schedule, start, end, state, ei_weighted=True
                )
                scale = expand_t(schedule.ei_martingale_scale(end.expand(batch)), state)
                path = _ei_path_from_g(scale * delta_m, schedule, start, end, t0=t0, degree=degree)
                state = ei_euler_map(net, schedule, state, time_s, time_t, path)
            else:
                path = realization.interval(start, end)
                features = path.coefficient_features(normalize=net.normalize_coefficients)
                net.set_brownian(path)
                try:
                    state = net(state, time_s, time_t, features)
                finally:
                    net.set_brownian(None)
        errors.append(((state - reference) ** 2).mean())
    return torch.stack(errors)
