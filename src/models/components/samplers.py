"""Inference-time samplers for flow-matching and stochastic-flow-map models.

These follow the project convention ``t = 0`` is **noise** and ``t = 1`` is
**data** (opposite to the DMF reference), so integration runs forward from
``t_start = 0.0`` to ``t_end = 1.0``. The models are **unconditional**: no class
labels or classifier-free guidance are involved.

- :func:`fm_euler_sampler` integrates a plain flow-matching velocity field with
  an optional Heun (2nd-order) correction.
- :func:`flow_map_sampler` chains the stochastic flow map ``Phi_{s,t}`` over a
  schedule of times, sampling fresh polynomial Brownian coefficients per step.
- :func:`ei_euler_sampler` chains the exponential-integrator Euler flow map,
  whose scaled martingale increment splits through :func:`split_ei_euler_path`.
- :func:`ei_doss_sussmann_sampler` chains the EI-Euler skeleton on the
  physical Doss state and reconstructs ``X = Z + ΔM / α``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .ablation_map import ablation_map, has_learned_noise, is_ablation_net
from .brownian import BrownianPolynomial
from .ei_doss import ei_doss_map, ei_doss_reconstruct, sample_ei_martingale_path
from .ei_euler import EIEulerPath, ei_euler_map, sample_ei_euler_path
from .gaussian import randn
from .schedule import AffineSchedule, Parameterization
from .ssfm import SSFMNet, analytic_euler_map


def expand_t_like_x(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a per-sample time vector ``t`` to broadcast against ``x``.

    :param t: Times of shape ``(B,)``.
    :param x: Reference tensor of shape ``(B, ...)``.
    :return: ``t`` reshaped to ``(B, 1, 1, ...)`` matching ``x.ndim``.
    """
    return t.view(t.size(0), *([1] * (x.ndim - 1)))


def _noise_end_state(schedule: AffineSchedule, latents: Tensor, t_start: float | Tensor) -> Tensor:
    """Map standardized prior noise onto the schedule's marginal at ``t_start``.

    Callers draw unit ``X_0 ~ N(0, I)``, but an affine path's noise-end
    marginal is ``sigma(t) X_0`` with (near-)zero data weight — Sana-Sprint's
    TrigFlow prior, for instance, scales noise by ``sigma_data`` — and the
    first integration node sits at the clock origin rather than ``t = 0``.
    Evaluating the marginal through ``sample_x_t`` keeps every schedule on
    its own path; unit-sigma schedules are unchanged up to the usual
    ``X_{t0} ~ X_0`` buffer.
    """
    return schedule.sample_x_t(latents, torch.zeros_like(latents), t_start)


def get_tsteps(
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    shift: float = 1.0,
) -> Tensor:
    """Build the integration time grid (noise ``->`` data by default).

    :param num_steps: Number of integration steps (returns ``num_steps + 1`` nodes).
    :param t_start: Start time (default ``0.0`` = noise).
    :param t_end: End time (default ``1.0`` = data).
    :param shift: Optional time-warp factor; ``1.0`` leaves the grid uniform.
    :return: Monotone time grid of shape ``(num_steps + 1,)``, dtype float64.
    """
    t_steps = torch.linspace(t_start, t_end, num_steps + 1, dtype=torch.float64)
    if shift == 1.0:
        return t_steps
    return shift * t_steps / (1 + (shift - 1.0) * t_steps)


@torch.no_grad()
def fm_euler_sampler(
    model: Callable,
    latents: Tensor,
    num_steps: int = 50,
    shift: float = 1.0,
    heun: bool = False,
) -> Tensor:
    """Integrate a flow-matching velocity field from noise to data.

    :param model: Callable ``model(x, t) -> velocity``.
    :param latents: Initial noise ``x_0`` of shape ``(B, C, H, W)``.
    :param num_steps: Number of Euler steps.
    :param shift: Time-grid warp factor.
    :param heun: If ``True``, apply a 2nd-order Heun correction per step.
    :return: The generated data sample ``(B, C, H, W)`` in the input dtype.
    """
    dtype, device = latents.dtype, latents.device
    t_steps = get_tsteps(num_steps, t_start=0.0, t_end=1.0, shift=shift).to(device)
    x = latents.to(torch.float64)

    for t_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True):
        dt = t_nxt - t_cur
        t_b = t_cur * torch.ones(x.shape[0], device=device, dtype=dtype)
        d_cur = model(x.to(dtype), t_b).to(torch.float64)
        x_euler = x + dt * d_cur

        if heun and t_nxt < 1.0:
            t_b_nxt = t_nxt * torch.ones(x.shape[0], device=device, dtype=dtype)
            d_nxt = model(x_euler.to(dtype), t_b_nxt).to(torch.float64)
            x = x + dt * 0.5 * (d_cur + d_nxt)
        else:
            x = x_euler

    return x.to(dtype)


@torch.no_grad()
def fm_em_sampler(
    model: Callable,
    schedule: AffineSchedule,
    latents: Tensor,
    prediction: Parameterization = 'velocity',
    num_steps: int = 50,
    eps: float = 1e-4,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate the flow-as-SDE from noise to data via Euler-Maruyama.

    The flow-matching probability path admits an SDE (Holderrieth "flow-as-SDE"
    form) with drift ``b_t(x) = 2 u_t(x) - (alphadot_t / alpha_t) x`` and
    diffusion ``nu_t = sqrt(2 (alphadot/alpha) sigma^2 - 2 sigma sigmadot)``.
    This sampler integrates that SDE forward from ``t = eps`` (noise) to
    ``t = 1`` (data) with the Euler-Maruyama scheme:

        x_{n+1} = x_n + b_t(x_n) dt + nu_t sqrt(dt) z,   z ~ N(0, I).

    Starting at ``t = eps`` instead of ``t = 0`` avoids the
    ``alphadot/alpha -> 1/t`` singularity of the linear schedule; for small
    ``eps``, ``X_eps ≈ X_0`` so the latents are a valid initial condition.

    :param model: Callable ``model(x, t) -> raw`` returning the network output
        in the ``prediction`` parameterization.
    :param schedule: The affine schedule providing ``alpha``, ``alpha_dot``,
        ``sigma``, ``sigma_dot`` (and thus ``nu_squared``).
    :param latents: Initial noise ``x_0`` of shape ``(B, C, H, W)``.
    :param prediction: Parameterization of the network output.
    :param num_steps: Number of Euler-Maruyama steps.
    :param eps: Start time (avoids the ``t = 0`` singularity).
    :param generator: Optional RNG for reproducible Brownian increments.
    :return: The generated data sample ``(B, C, H, W)`` in the input dtype.
    """
    dtype, device = latents.dtype, latents.device
    t_steps = get_tsteps(num_steps, t_start=eps, t_end=1.0).to(device)
    x = _noise_end_state(schedule, latents, t_steps[0]).to(torch.float64)

    for t_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True):
        dt = (t_nxt - t_cur).to(torch.float64)
        t_b = t_cur * torch.ones(x.shape[0], device=device, dtype=dtype)
        # Network output -> velocity u_t.
        raw = model(x.to(dtype), t_b).to(torch.float64)
        u_t = schedule.to_velocity(raw, prediction, x, t_b)
        # SDE drift: b_t = 2 u_t - (alphadot/alpha) x.
        ratio = expand_t_like_x(schedule.alpha_dot(t_b) / schedule.alpha(t_b), x)
        drift = 2.0 * u_t - ratio * x
        # Diffusion coefficient nu_t = sqrt(nu_squared), clamped >= 0.
        nu = expand_t_like_x(schedule.nu_squared(t_b).clamp_min(0.0).sqrt(), x)
        # Brownian increment dW = sqrt(dt) * z.
        z = randn(tuple(x.shape), device=device, dtype=torch.float64, generator=generator)
        x = x + drift * dt + nu * z * dt.sqrt()

    return x.to(dtype)


@torch.no_grad()
def flow_map_sampler(
    net: SSFMNet,
    latents: Tensor,
    num_steps: int = 4,
    degree: int = 1,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
    paths: list[BrownianPolynomial] | None = None,
    schedule: AffineSchedule | None = None,
    analytic_diffusion: bool = False,
    t_start: float = 0.0,
) -> Tensor:
    """Chain the stochastic flow map ``Phi_{s,t}`` from noise to data.

    For each step ``[s, t]`` a polynomial Brownian path is assigned to the network
    via :meth:`SSFMNet.set_brownian` and its normalized coefficient features are
    supplied to the network. The network assembles the full flow-map step ``Psi``
    internally (including the Euler-Maruyama ``W_{s,t} g`` term when enabled), so
    the caller simply chains its output.

    By default a fresh path is sampled per step. Passing ``paths`` instead rides a
    set of *precomputed* per-step paths (one per step, in time order) — used by
    :func:`multistep_flow_map_sampler` to share a single realized Brownian motion
    across several step counts via the Chen-relation dyadic tree.

    :param net: A concrete :class:`SSFMNet` flow-map network.
    :param latents: Initial noise ``x_0`` of shape ``(B, C, H, W)``.
    :param num_steps: Number of flow-map steps (ignored when ``paths`` is given).
    :param degree: Polynomial Brownian degree ``N`` per step.
    :param shift: Time-grid warp factor (must be ``1.0`` when ``paths`` is given).
    :param generator: Optional RNG for reproducible increments.
    :param paths: Optional precomputed per-step paths; each path's ``[s, t]``
        endpoints define that step's time interval (overrides ``num_steps``).
    :return: The generated data sample ``(B, C, H, W)`` in the input dtype.
    """
    dtype, device = latents.dtype, latents.device
    x = latents
    steps = _resolve_steps(
        x, num_steps, degree, shift, device, dtype, generator, paths, t_start=t_start
    )
    for s_cur, t_nxt, brownian in steps:
        s_b = s_cur.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        t_b = t_nxt.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        coefficients = brownian.coefficient_features(normalize=net.normalize_coefficients)
        if analytic_diffusion:
            if schedule is None:
                raise ValueError('analytic_diffusion requires a schedule.')
            x = analytic_euler_map(net, schedule, x, s_b, t_b, brownian, coefficients)
        else:
            net.set_brownian(brownian)
            x = net(x, s_b, t_b, coefficients)

    net.set_brownian(None)
    return x.to(dtype)


def _resolve_steps(
    x: Tensor,
    num_steps: int,
    degree: int,
    shift: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    paths: list[BrownianPolynomial] | None,
    *,
    t_start: float = 0.0,
) -> list[tuple[Tensor, Tensor, BrownianPolynomial]]:
    """Build the ``(s, t, path)`` step schedule, sampling fresh paths if needed."""
    if paths is not None:
        return [(p.s.to(device), p.t.to(device), p) for p in paths]

    t_steps = get_tsteps(num_steps, t_start=t_start, t_end=1.0, shift=shift).to(device)
    schedule = []
    for s_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True):
        brownian = BrownianPolynomial.sample(
            tuple(x.shape),
            s=float(s_cur),
            t=float(t_nxt),
            degree=degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        schedule.append((s_cur, t_nxt, brownian))
    return schedule


@torch.no_grad()
def ei_doss_sussmann_sampler(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    num_steps: int = 4,
    degree: int = 3,
    t0: float = 0.01,
    shift: float = 1.0,
    zeta: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Chain locally anchored (annealed) EI-Doss maps in physical X.

    Each step samples ``I^M_{s,t}`` on the (annealed) EI-weighted clock,
    applies the annealed deterministic EI-Euler skeleton
    ``Z_t = (α_s/α_t)^{ζ²} X_s + ((1+ζ²) A^{(ζ)}_{s,t}/α_t^{ζ²}) u``, and
    reconstructs ``X_t = Z_t + (M~^ζ_t - M~^ζ_s) / α_t^{ζ²}``.
    ``zeta = 0`` samples the probability-flow ODE (zero martingale, Euler
    skeleton); ``zeta = 1`` (default) is the un-annealed EI-Doss sampler.
    Adjacent steps draw independent local paths (weak sampling); pathwise
    coupling uses
    :func:`~src.models.components.strong_convergence.ei_doss_rollout_batch`.

    :param net: Euler-parameterized flow map predicting velocity ``u``.
    :param schedule: Affine schedule providing ``alpha`` and the annealed EI clock.
    :param latents: Initial noise ``X_{t0}`` of shape ``(B, ...)``.
    :param num_steps: Number of flow-map steps over ``[t0, 1]``.
    :param degree: Polynomial Brownian degree of each local path.
    :param t0: Clock origin (``M_{t0} = 0``); must be ``> 0`` so ``α_{t0} > 0``.
    :param shift: Time-grid warp factor over ``[t0, 1]``.
    :param zeta: Annealing amplitude in ``[0, 1]`` (0 = ODE, 1 = flow-as-SDE).
    :param generator: Optional RNG for reproducible local paths.
    :return: Generated data sample ``(B, ...)`` in the input dtype.
    """
    dtype, device = latents.dtype, latents.device
    t_steps = get_tsteps(num_steps, t_start=t0, t_end=1.0, shift=shift).to(device)
    x = _noise_end_state(schedule, latents, t_steps[0])
    for s_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True):
        s_b = s_cur.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        t_b = t_nxt.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        field_ones = (1,) * (x.ndim - 1)
        martingale = sample_ei_martingale_path(
            schedule,
            tuple(x.shape),
            s=s_b.view(-1, *field_ones),
            t=t_b.view(-1, *field_ones),
            degree=degree,
            t0=t0,
            zeta=zeta,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        coefficients = martingale.coefficient_features(normalize=net.normalize_coefficients)
        z_t = ei_doss_map(net, schedule, x, s_b, t_b, coefficients, zeta)
        x = ei_doss_reconstruct(schedule, z_t, t_b, martingale.terminal_increment(), zeta)

    return x.to(dtype)


@torch.no_grad()
def chen_matched_ei_doss_rows(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    step_counts: list[int],
    degree: int = 3,
    t0: float = 0.01,
    zeta: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Integrate one shared ``X_0`` under leaf-composed paths at every step count.

    The Chen-coupled (matched-Brownian-path) NFE comparison grid: fresh
    degree-``degree`` leaf polynomials are sampled on the finest
    (``max(step_counts)``) clock grid — the same per-step path distribution as
    the stock sampler — and every coarser step count composes those leaves with
    ``BrownianPolynomial.concat_arbitrary`` (exact Chen relations), then
    integrates them with the EI-Doss map. Within a row, moving between step
    counts changes *only* the time discretization; the Brownian realization is
    shared. This is unlike restricting a single global polynomial, whose leaf
    features are far off-distribution and sample badly.

    :param net: Euler-parameterized flow map predicting velocity ``u``.
    :param schedule: Affine schedule providing ``alpha`` and the annealed EI clock.
    :param latents: Initial noise of shape ``(B, ...)`` shared by every row.
    :param step_counts: NFEs to evaluate; the largest must be a power of two
        and every entry must divide it (rows are returned in the given order).
    :param degree: Polynomial Brownian degree of each leaf.
    :param t0: Clock origin (``M_{t0} = 0``); must be ``> 0`` so ``α_{t0} > 0``.
    :param zeta: Annealing amplitude in ``[0, 1]`` (0 = ODE, 1 = flow-as-SDE).
    :param generator: Optional RNG for the leaf paths (one realization shared
        by every row).
    :return: Latent samples of shape ``(len(step_counts), B, ...)``, indexed
        ``(row, batch)``.
    """
    if net.parameterization != 'euler':
        raise ValueError(
            f'chen_matched_ei_doss_rows requires an Euler network; got {net.parameterization!r}.'
        )
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    if not step_counts:
        raise ValueError('step_counts must be non-empty.')
    max_steps = max(step_counts)
    if max_steps <= 0 or (max_steps & (max_steps - 1)) != 0:
        raise ValueError(f'The largest step count must be a power of two; got {max_steps}.')
    if any(max_steps % steps for steps in step_counts):
        raise ValueError(
            f'Every step count must divide the largest {max_steps}; got {step_counts}.'
        )

    device = latents.device
    num_paths = latents.shape[0]
    nodes = torch.linspace(t0, 1.0, max_steps + 1, device=device, dtype=torch.float64)
    latents = _noise_end_state(schedule, latents, nodes[0])
    # Monotone clamp: under a nearly-flat annealed clock (small zeta) rounding
    # can make the raw sequence non-monotone; clamping keeps every leaf span
    # >= 0 exactly (a zero-span leaf samples zero coefficients — the exact
    # zeta -> 0 limit) and adjacent leaves share endpoints bit-for-bit.
    clock_nodes = []
    running_max = None
    for index in range(max_steps + 1):
        value = schedule.annealed_ei_weighted_clock(nodes[index], t0, zeta)
        if running_max is not None and float(value) < float(running_max):
            value = running_max
        else:
            running_max = value
        clock_nodes.append(value)
    # Degenerate annealed clock (zeta = 0 exactly, or spans rounding away at
    # tiny zeta): the martingale is identically zero, so every row reduces to
    # the deterministic EI-Doss ODE skeleton. Composing zero-span leaves would
    # trip concat_arbitrary's strict span validation, so build the zero
    # polynomial directly -- the exact zeta -> 0 limit, not an approximation.
    zero_clock = all(
        not bool((clock_nodes[index + 1] - clock_nodes[index]) > 0) for index in range(max_steps)
    )
    # Zero polynomial over the full clock: the exact zeta -> 0 limit used for
    # degenerate all-zero leaves and for rows whose leaves all round to zero.
    zero_leaf = BrownianPolynomial(
        coeffs=latents.new_zeros((degree, *latents.shape)),
        s=clock_nodes[0],
        t=clock_nodes[-1],
    )
    if zero_clock:
        leaf_paths = [zero_leaf] * max_steps
    else:
        leaf_paths = [
            BrownianPolynomial.sample(
                tuple(latents.shape),
                clock_nodes[index],
                clock_nodes[index + 1],
                degree,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            for index in range(max_steps)
        ]

    samples = []
    for steps in step_counts:
        stride = max_steps // steps
        x = latents
        for start in range(0, max_steps, stride):
            end = start + stride
            composed = None
            for leaf in leaf_paths[start:end]:
                # A nearly-flat annealed clock (small zeta) can round a leaf's
                # span to zero; a zero-span leaf is a zero increment (the
                # exact zeta -> 0 limit), so skip it — and seed the row from
                # the first positive-span leaf so the composition stays valid.
                if float(leaf.t - leaf.s) <= 0.0:
                    continue
                composed = (
                    leaf
                    if composed is None
                    else BrownianPolynomial.concat_arbitrary(composed, leaf)
                )
            if composed is None:
                # Every leaf in this row rounded to zero span: the row is the
                # deterministic ODE skeleton (zero martingale).
                composed = zero_leaf
            s_b = nodes[start].to(x.dtype) * torch.ones(num_paths, device=device, dtype=x.dtype)
            t_b = nodes[end].to(x.dtype) * torch.ones(num_paths, device=device, dtype=x.dtype)
            coefficients = composed.coefficient_features(normalize=net.normalize_coefficients)
            z_t = ei_doss_map(net, schedule, x, s_b, t_b, coefficients, zeta)
            x = ei_doss_reconstruct(schedule, z_t, t_b, composed.coeffs[0], zeta)
        samples.append(x)
    return torch.stack(samples)


def split_ei_euler_path(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    s: Tensor,
    m: Tensor,
    t: Tensor,
    degree: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    t0: float = 0.0,
    degrees: Tensor | None = None,
) -> tuple[EIEulerPath, EIEulerPath, EIEulerPath]:
    """Split one EI-Euler step ``[s, t]`` into Chen-consistent children.

    The EI-Euler map carries the *scaled* stochastic term
    ``M_{s,t} / alpha_t`` with ``M_{s,t} = int_s^t nu_r dW_r``. Only the
    **unscaled** martingale increments are additive,

    ``M_{s,t} = M_{s,m} + M_{m,t}``,

    so this helper does its Chen composition in unscaled coordinates: two
    independent :class:`BrownianPolynomial` leaves are drawn on the adjacent
    martingale-clock intervals ``[clock(s), clock(m)]`` and
    ``[clock(m), clock(t)]``, composed with
    :meth:`BrownianPolynomial.concat_arbitrary` (the existing Chen precedent for
    unequal spans), and only then divided by the terminal ``alpha`` of the
    interval each child actually terminates at.

    Consequently the returned children are *not* additive in their own
    coordinates. Composing them reproduces the parent increment as

    ``left.terminal_increment() * (alpha_m / alpha_t)
      + right.terminal_increment() == parent.terminal_increment()``,

    i.e. the left child must be re-expressed in parent coordinates by the
    state coefficient ``A_mt = alpha_m / alpha_t`` before being added, exactly as
    ``docs/ei_euler_objective.md`` prescribes. Adding scaled increments directly
    is the failure mode this helper exists to prevent.

    :param schedule: Affine schedule owning ``alpha`` and the martingale clock.
    :param shape: Per-coefficient sample shape, e.g. ``(B, C, H, W)``.
    :param s: Interval start times, pre-shaped to broadcast against ``shape``.
    :param m: Split times, strictly inside ``(s, t)``.
    :param t: Interval terminal times.
    :param degree: Polynomial degree ``N`` shared by both children and the parent.
    :param device: Optional torch device.
    :param dtype: Optional torch dtype.
    :param generator: Optional RNG for reproducible leaves.
    :param t0: Shared martingale-clock anchor; it cancels in every difference.
    :param degrees: Optional per-sample active degree in ``[1, degree]``. When
        set, padded modes stay zero and the mixed-degree metadata is preserved
        through Chen composition.
    :return: ``(left, right, parent)`` scaled paths on ``[s, m]``, ``[m, t]``,
        and ``[s, t]``, all built from one realized martingale.
    :raises ValueError: If ``degree < 1`` or any split violates ``s < m < t``.
    """
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    s_t = torch.as_tensor(s, device=device, dtype=dtype)
    m_t = torch.as_tensor(m, device=device, dtype=dtype)
    t_t = torch.as_tensor(t, device=device, dtype=dtype)
    if torch.any(s_t <= 0):
        raise ValueError('EI-Euler intervals require s > 0.')
    if torch.any((m_t <= s_t) | (m_t >= t_t)):
        raise ValueError('EI-Euler split times must satisfy s < m < t.')
    if torch.any(t_t > 1):
        raise ValueError('EI-Euler intervals require t <= 1.')

    # Unscaled martingale leaves live on the clock, where quadratic variation is
    # the clock difference, so Chen composition is the plain Brownian relation.
    clock_s = schedule.ei_weighted_clock(s_t, t0)
    clock_m = schedule.ei_weighted_clock(m_t, t0)
    clock_t = schedule.ei_weighted_clock(t_t, t0)
    left_leaf = BrownianPolynomial.sample(
        shape,
        s=clock_s,
        t=clock_m,
        degree=degree,
        degrees=degrees,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    right_leaf = BrownianPolynomial.sample(
        shape,
        s=clock_m,
        t=clock_t,
        degree=degree,
        degrees=degrees,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    parent_leaf = BrownianPolynomial.concat_arbitrary(left_leaf, right_leaf)

    def _scaled(leaf: BrownianPolynomial, start: Tensor, terminal: Tensor) -> EIEulerPath:
        """Divide one unscaled leaf by ``alpha`` at the interval it terminates at."""
        scale = schedule.ei_martingale_scale(terminal)
        variance = schedule.ei_driver_variance(start, terminal, t0=t0)
        return EIEulerPath(
            coeffs=scale.unsqueeze(0) * leaf.coeffs,
            normalized_coeffs=leaf.coefficient_features(normalize=True).movedim(1, 0),
            variance=variance,
            degrees=leaf.degrees,
        )

    return (
        _scaled(left_leaf, s_t, m_t),
        _scaled(right_leaf, m_t, t_t),
        _scaled(parent_leaf, s_t, t_t),
    )


@torch.no_grad()
def ei_euler_sampler(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    num_steps: int = 4,
    degree: int = 1,
    t0: float = 0.01,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Chain exponential-integrator Euler flow maps from noise to data.

    Each step ``[s, t]`` samples a fresh :class:`EIEulerPath` and advances the
    state through :func:`~src.models.components.ei_euler.ei_euler_map`, which
    evaluates

    ``X_t = (alpha_s / alpha_t) X_s + (A_st / alpha_t) u_st^theta(X_s; I)
            + (1 / alpha_t) int_s^t nu_r dW_r``.

    The map already adds the scaled martingale term, so this sampler performs
    **no** martingale reconstruction of its own. Adding one would double-count
    the noise.

    The network's raw ``drift`` head is interpreted as the two-time
    interval-averaged velocity ``u_st``, conditioned on the martingale
    polynomial features. The construction is Euler-only by design: the
    stochastic term comes from the sampled path, never from a learned
    Euler--Maruyama diffusion head.

    :param net: A concrete :class:`SSFMNet` velocity flow-map network (``euler``).
    :param schedule: Affine schedule owning the EI-Euler coefficients, the
        martingale scaling, and the clock.
    :param latents: Initial noise ``x_0`` of shape ``(B, C, H, W)``.
    :param num_steps: Number of flow-map steps over ``[t0, 1]``.
    :param degree: Polynomial degree ``N`` of the per-step martingale path.
    :param t0: Start time; must be ``> 0`` because ``alpha_s / alpha_t`` and the
        linear-schedule clock are singular at the origin.
    :param shift: Time-grid warp factor over ``[t0, 1]``.
    :param generator: Optional RNG making the sample deterministic.
    :return: The generated data sample ``(B, C, H, W)`` in the input dtype.
    :raises ValueError: If ``net`` is not an Euler network or ``degree < 1``.
    """
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    if is_ablation_net(net):
        return ablation_sampler(
            net, schedule, latents, num_steps, degree, t0, shift=shift, generator=generator
        )
    if net.parameterization != 'euler':
        raise ValueError(
            f'ei_euler_sampler requires an Euler network; got {net.parameterization!r}.'
        )

    dtype, device = latents.dtype, latents.device
    t_steps = get_tsteps(num_steps, t_start=t0, t_end=1.0, shift=shift).to(device)
    x = _noise_end_state(schedule, latents, t_steps[0])
    for s_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True):
        s_b = s_cur.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        t_b = t_nxt.to(dtype) * torch.ones(x.shape[0], device=device, dtype=dtype)
        field_ones = (1,) * (x.ndim - 1)
        path = sample_ei_euler_path(
            schedule,
            tuple(x.shape),
            s=s_b.view(-1, *field_ones),
            t=t_b.view(-1, *field_ones),
            degree=degree,
            device=device,
            dtype=dtype,
            generator=generator,
            t0=t0,
        )
        x = ei_euler_map(net, schedule, x, s_b, t_b, path)

    return x.to(dtype)


def _ablation_leaves(
    net: SSFMNet,
    schedule: AffineSchedule,
    nodes: Tensor,
    shape: tuple[int, ...],
    degree: int,
    t0: float,
    generator: torch.Generator | None,
) -> list[BrownianPolynomial]:
    """Sample one independent polynomial path per grid interval.

    Learned-noise nets ride real time (``W``); unlifted closed-form nets ride
    the martingale clock so each leaf's increment is ``int nu dW``.
    """
    coords = nodes if has_learned_noise(net) else schedule.martingale_clock(nodes, t0)
    return [
        BrownianPolynomial.sample(
            shape, coords[i], coords[i + 1], degree,
            device=nodes.device, dtype=nodes.dtype, generator=generator,
        )
        for i in range(nodes.numel() - 1)
    ]


def _chain_ablation(
    net: SSFMNet,
    schedule: AffineSchedule,
    x: Tensor,
    nodes: Tensor,
    leaves: list[BrownianPolynomial],
    stride: int,
) -> Tensor:
    """Chain ablation maps over ``nodes``, each hop composing ``stride`` leaves."""
    batch = x.shape[0]
    for start in range(0, len(leaves), stride):
        path = leaves[start]
        for child in leaves[start + 1 : start + stride]:
            path = BrownianPolynomial.concat_arbitrary(path, child)
        s = nodes[start].expand(batch)
        t = nodes[start + stride].expand(batch)
        x = ablation_map(net, schedule, x, s, t, path)
    return x


@torch.no_grad()
def ablation_sampler(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    num_steps: int = 4,
    degree: int = 1,
    t0: float = 0.01,
    *,
    shift: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Chain an ablation-ladder map (learned noise and/or no lift) to data.

    Each step rides a fresh polynomial path on ``[s, t]`` (see
    :func:`~.ablation_map.ablation_map`).
    """
    nodes = get_tsteps(num_steps, t_start=t0, t_end=1.0, shift=shift).to(
        device=latents.device, dtype=latents.dtype
    )
    x = _noise_end_state(schedule, latents, nodes[0])
    leaves = _ablation_leaves(net, schedule, nodes, tuple(x.shape), degree, t0, generator)
    return _chain_ablation(net, schedule, x, nodes, leaves, 1).to(latents.dtype)


@torch.no_grad()
def multistep_ablation_sampler(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    max_steps: int = 16,
    degree: int = 1,
    t0: float = 0.01,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[int]]:
    """Sample every dyadic step count on one Chen-coupled ablation path."""
    nodes = torch.linspace(t0, 1.0, max_steps + 1, device=latents.device, dtype=latents.dtype)
    x = _noise_end_state(schedule, latents, nodes[0])
    leaves = _ablation_leaves(net, schedule, nodes, tuple(x.shape), degree, t0, generator)
    step_counts = [max_steps // 2**level for level in range(max_steps.bit_length())]
    samples = [
        _chain_ablation(net, schedule, x, nodes, leaves, max_steps // steps)
        for steps in step_counts
    ]
    return torch.stack(samples, dim=0).to(latents.dtype), step_counts


@torch.no_grad()
def multistep_ei_euler_sampler(
    net: SSFMNet,
    schedule: AffineSchedule,
    latents: Tensor,
    max_steps: int = 16,
    degree: int = 1,
    t0: float = 0.01,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[int]]:
    """Sample every dyadic step count from one Chen-coupled EI-clock path.

    Finest leaves live on the weighted clock
    ``int alpha^2 nu^2``. Coarse hops compose those leaves, then scale by
    ``1 / alpha_t`` of the interval they terminate at, so
    ``G_{s,t} = (alpha_m / alpha_t) G_{s,m} + G_{m,t}``.
    """
    if max_steps <= 0 or (max_steps & (max_steps - 1)) != 0:
        raise ValueError(f'max_steps must be a positive power of two; got {max_steps}.')
    if degree < 1:
        raise ValueError(f'degree must be at least 1; got {degree}.')
    if is_ablation_net(net):
        return multistep_ablation_sampler(
            net, schedule, latents, max_steps, degree, t0, generator=generator
        )
    if net.parameterization != 'euler':
        raise ValueError(
            f'multistep_ei_euler_sampler requires an Euler network; got {net.parameterization!r}.'
        )

    dtype, device = latents.dtype, latents.device
    nodes = torch.linspace(t0, 1.0, max_steps + 1, device=device, dtype=dtype)
    latents = _noise_end_state(schedule, latents, nodes[0])
    batch = latents.shape[0]
    field_ones = (1,) * (latents.ndim - 1)
    clock_nodes = [schedule.ei_weighted_clock(nodes[index], t0) for index in range(max_steps + 1)]
    leaf_brownian = [
        BrownianPolynomial.sample(
            tuple(latents.shape),
            clock_nodes[index],
            clock_nodes[index + 1],
            degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for index in range(max_steps)
    ]

    def _shared_path(start_index: int, end_index: int) -> EIEulerPath:
        start = nodes[start_index].expand(batch)
        end = nodes[end_index].expand(batch)
        s_field = start.view(-1, *field_ones)
        t_field = end.view(-1, *field_ones)
        composed = leaf_brownian[start_index]
        for child in leaf_brownian[start_index + 1 : end_index]:
            composed = BrownianPolynomial.concat_arbitrary(composed, child)
        return EIEulerPath(
            coeffs=schedule.ei_martingale_scale(t_field).unsqueeze(0) * composed.coeffs,
            normalized_coeffs=composed.coefficient_features(normalize=True).movedim(1, 0),
            variance=schedule.ei_driver_variance(s_field, t_field, t0=t0),
            degrees=composed.degrees,
        )

    samples = []
    step_counts = []
    for num_steps in (max_steps // (2**level) for level in range(max_steps.bit_length())):
        stride = max_steps // num_steps
        x = latents
        for start_index in range(0, max_steps, stride):
            end_index = start_index + stride
            s = nodes[start_index].expand(batch)
            t = nodes[end_index].expand(batch)
            x = ei_euler_map(net, schedule, x, s, t, _shared_path(start_index, end_index))
        samples.append(x.to(dtype))
        step_counts.append(num_steps)

    return torch.stack(samples, dim=0), step_counts


@torch.no_grad()
def multistep_flow_map_sampler(
    net: SSFMNet,
    latents: Tensor,
    max_steps: int = 16,
    degree: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[int]]:
    """Generate one sample per dyadic step count from a *shared* Brownian motion.

    Samples a single realized polynomial Brownian motion at the finest resolution
    (``max_steps`` uniform leaf intervals on ``[0, 1]``) and folds it down through
    the Chen relations (:meth:`BrownianPolynomial.build_dyadic_levels`) into every
    coarser dyadic resolution ``max_steps, max_steps/2, ..., 1``. The same initial
    ``latents`` and the same underlying Brownian path drive every resolution, so
    the rows form a clean step-count sweep where the *only* difference is the
    number of network evaluations (discretization).

    :param net: A concrete :class:`SSFMNet` flow-map network.
    :param latents: Initial noise ``x_0`` of shape ``(B, C, H, W)``, shared by all
        resolutions.
    :param max_steps: Finest step count; must be a positive power of two.
    :param degree: Polynomial Brownian degree ``N`` per step.
    :param generator: Optional RNG for reproducible increments.
    :return: ``(samples, step_counts)`` where ``samples`` has shape
        ``(R, B, C, H, W)`` (one row per resolution, finest first) and
        ``step_counts`` lists the corresponding step count of each row.
    :raises ValueError: If ``max_steps`` is not a positive power of two.
    """
    if max_steps <= 0 or (max_steps & (max_steps - 1)) != 0:
        raise ValueError(f'max_steps must be a positive power of two; got {max_steps}.')

    dtype, device = latents.dtype, latents.device
    # Uniform finest grid: midpoints of coarse intervals are exact fine-grid nodes,
    # so dyadic Chen folding is exact (no time-warp shift allowed here).
    t_steps = get_tsteps(max_steps, t_start=0.0, t_end=1.0, shift=1.0).to(device)
    leaves = [
        BrownianPolynomial.sample(
            tuple(latents.shape),
            s=float(s_cur),
            t=float(t_nxt),
            degree=degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for s_cur, t_nxt in zip(t_steps[:-1], t_steps[1:], strict=True)
    ]

    levels = BrownianPolynomial.build_dyadic_levels(leaves)
    samples = []
    step_counts = []
    for level in levels:
        samples.append(flow_map_sampler(net, latents, degree=degree, paths=level))
        step_counts.append(len(level))

    return torch.stack(samples, dim=0), step_counts
