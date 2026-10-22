"""Weighted particle samplers for FKC toy problems.

Two references and one learned sampler:

* :func:`numerical_fkc` integrates the coupled FKC system
  ``dX = b_beta(X, t) dt + nu_t dW``, ``dw = g_beta(X, t) dt`` with a fine
  exponential-integrator Euler scheme driven by independent Gaussian
  martingale increments. This is the high-resolution Ito baseline.
* :func:`wz_reference` runs the same EI-Euler scheme but drives it with the
  increments of ONE fixed polynomial (Wong-Zakai) martingale realization, so
  it solves the same truncated WZ path used by the learned map.
* :func:`learned_fkc` applies the augmented network map ``steps`` times,
  returning samples and accumulated log weights, with optional systematic
  resampling at step boundaries.

The two references must not be conflated: only the first is an Ito SDE solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.models.components.brownian import BrownianPolynomial
from src.models.components.ei_doss import (
    GlobalEIMartingalePath,
    sample_ei_martingale_path,
)
from src.models.components.schedule import AffineSchedule, LinearSchedule

from .network import augmented_fkc_map, potential_scale, velocity_from_prediction
from .weights import effective_sample_size, systematic_resample


class LearnedFKCFields:
    """Expose a trained augmented network as an FKC ``problem``.

    Returns the *diagonal* learned fields ``(velocity, rate) = net(x, t, t, 0)``
    so :func:`numerical_fkc` integrates the learned drift
    ``b = 2 velocity - (alpha_dot / alpha) x`` with the learned weight rate. This
    isolates the learned-field error from the few-step flow-map error: a correct
    drift/rate pair should reproduce the target under a fine rollout even when
    the few-step map is inaccurate.
    """

    def __init__(self, net: object, schedule: AffineSchedule) -> None:
        self.net = net
        self.schedule = schedule

    def _times(self, x: Tensor, t: Tensor | float) -> Tensor:
        """Coerce a scalar or ``(B,)`` time to ``(B,)`` on ``x``'s device/dtype."""
        t_b = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t_b.ndim == 0:
            t_b = t_b.expand(x.shape[0])
        return t_b

    def fields(self, x: Tensor, t: Tensor | float) -> tuple[Tensor, Tensor]:
        """Return the learned ``(velocity, potential g)`` at the diagonal ``s = t``."""
        t_b = self._times(x, t)
        coefficients = x.new_zeros(x.shape[0], 1, 2)
        prediction, rate = self.net(x, t_b, t_b, coefficients)
        velocity = velocity_from_prediction(self.net, prediction, self.schedule, t_b, t_b)
        # The head predicts the potential g directly for bounded-nu schedules and
        # the clock integrand psi = g / nu^2 otherwise; convert back to g so the
        # shared numerical_fkc weight update recovers the head's rate.
        if getattr(self.net, 'output_parameterization', 'velocity') == 'score':
            potential = rate * potential_scale(self.net, self.schedule, t_b)
        elif self.schedule.weight_rate_is_potential:
            potential = rate
        else:
            potential = rate * self.schedule.nu_squared(t_b)
        return velocity, potential

    def drift(self, x: Tensor, t: Tensor | float) -> Tensor:
        """Return the learned SDE drift ``b = 2 velocity - (alpha_dot / alpha) x``."""
        velocity, _ = self.fields(x, t)
        t_b = self._times(x, t)
        a = (self.schedule.alpha_dot(t_b) / self.schedule.alpha(t_b))[:, None] * x
        return 2.0 * velocity - a


@dataclass(frozen=True)
class FKCSamples:
    """Weighted terminal particles from an FKC sampler.

    :param x: Spatial states ``(n, 2)``.
    :param log_weights: Accumulated log weights ``(n,)``.
    :param resampling_count: Number of resampling events performed.
    """

    x: Tensor
    log_weights: Tensor
    resampling_count: int = 0


def _time_grid(
    t0: float, t1: float, steps: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Validate the integration window and return ``steps + 1`` increasing times."""
    if steps < 1:
        raise ValueError(f'steps must be at least 1; got {steps}.')
    if not 0.0 <= t0 < t1 <= 1.0:
        raise ValueError(f'need 0 <= t0 < t1 <= 1; got t0={t0}, t1={t1}.')
    return torch.linspace(t0, t1, steps + 1, device=device, dtype=dtype)


def _initial_weights(x: Tensor, w: Tensor | None) -> Tensor:
    """Return the supplied log weights or zeros matching ``x``'s batch."""
    if w is None:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if w.ndim != 1 or w.shape[0] != x.shape[0]:
        raise ValueError(f'w must be of shape ({x.shape[0]},); got {tuple(w.shape)}.')
    return w


def _clip_log_weights(log_w: Tensor, clip: float | None) -> Tensor:
    """Clip the extreme ``clip`` fraction of each tail of the log-weight cloud.

    ``clip`` is a tail *rate* (e.g. ``0.002`` = 0.2% per tail): log weights are
    clamped to the ``[clip, 1 - clip]`` empirical quantiles, the usual
    particle-filter safeguard against a handful of particles dominating.
    Applied identically to the learned and reference samplers.
    """
    if clip is None:
        return log_w
    if not 0.0 < clip < 0.5:
        raise ValueError(f'log_weight_clip must lie in (0, 0.5); got {clip}.')
    lower = log_w.quantile(clip)
    upper = log_w.quantile(1.0 - clip)
    return log_w.clamp(lower, upper)


def _resample_if_needed(
    x: Tensor,
    w: Tensor,
    ess_threshold: float | None,
    resample_every_step: bool,
    generator: torch.Generator | None,
    *,
    step: int | None = None,
    resampling_interval: int | None = None,
) -> tuple[Tensor, Tensor, bool]:
    """Systematically resample every step, every ``K`` steps, on ESS, or never.

    ``resampling_interval=K`` follows the FKC reference
    (``gmm_temp_annealed_birth_death.ipynb``): after 1-indexed step ``k`` the
    cloud is systematically resampled iff ``k % K == 0`` and log weights reset
    to zero. ``K=1`` is the paper's protocol; ``K > steps`` never resamples.
    """
    if resampling_interval is not None:
        if resampling_interval < 1:
            raise ValueError(f'resampling_interval must be >= 1; got {resampling_interval}.')
        if step is None:
            raise ValueError('resampling_interval requires the 1-indexed step.')
        if step % resampling_interval == 0:
            indices = systematic_resample(w, generator=generator)
            return x[indices], torch.zeros_like(w), True
        return x, w, False
    if resample_every_step:
        indices = systematic_resample(w, generator=generator)
        return x[indices], torch.zeros_like(w), True
    if ess_threshold is None:
        return x, w, False
    if not 0.0 < ess_threshold <= 1.0:
        raise ValueError(f'ess_threshold must lie in (0, 1]; got {ess_threshold}.')
    if float(effective_sample_size(w)) < ess_threshold * x.shape[0]:
        indices = systematic_resample(w, generator=generator)
        return x[indices], torch.zeros_like(w), True
    return x, w, False


def numerical_fkc(
    problem: object,
    x: Tensor,
    w: Tensor | None = None,
    *,
    t0: float = 0.05,
    t1: float = 1.0,
    steps: int = 64,
    generator: torch.Generator | None = None,
    ess_threshold: float | None = None,
    resample_every_step: bool = False,
    resampling_interval: int | None = None,
    log_weight_clip: float | None = None,
    schedule: LinearSchedule | None = None,
) -> FKCSamples:
    """Integrate the coupled FKC SDE with independent Ito increments.

    :param problem: Object exposing ``fields(x, t) -> (velocity, g)`` (e.g.
        :class:`~src.models.components.fkc.problem.FKCProblem`).
    :param x: Initial spatial states ``(n, 2)``.
    :param w: Optional initial log weights ``(n,)`` (defaults to zeros).
    :param t0: Positive start time (the linear-schedule SDE is singular at 0).
    :param t1: Terminal time.
    :param steps: Number of EI-Euler steps.
    :param generator: Optional RNG.
    :param ess_threshold: If set, resample when ESS drops below this fraction.
    :param resample_every_step: If ``True``, systematically resample after every
        step (the paper's SMC protocol); fresh increments are drawn afterwards.
    :return: Weighted terminal :class:`FKCSamples`.
    """
    schedule = schedule or LinearSchedule()
    times = _time_grid(t0, t1, steps, device=x.device, dtype=x.dtype)
    log_w = _initial_weights(x, w)
    clock = schedule.ei_weighted_clock(times, t0)
    delta_clock = clock[1:] - clock[:-1]
    resampling_count = 0
    for step in range(steps):
        s, t = times[step], times[step + 1]
        delta_m = delta_clock[step].sqrt() * torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )
        x, log_w = _ei_step(problem, x, log_w, s, t, delta_m, schedule, t0)
        log_w = _clip_log_weights(log_w, log_weight_clip)
        x, log_w, resampled = _resample_if_needed(
            x,
            log_w,
            ess_threshold,
            resample_every_step,
            generator,
            step=step + 1,
            resampling_interval=resampling_interval,
        )
        resampling_count += int(resampled)
    return FKCSamples(x, log_w, resampling_count)


def wz_reference(
    problem: object,
    x: Tensor,
    w: Tensor | None,
    martingale: GlobalEIMartingalePath,
    *,
    t0: float = 0.05,
    t1: float = 1.0,
    steps: int = 64,
    schedule: LinearSchedule | None = None,
) -> FKCSamples:
    """Solve the same truncated Wong-Zakai path with the fine EI-Euler scheme.

    Unlike :func:`numerical_fkc`, the increments are read from one fixed
    polynomial martingale realization, so this is a reference for the learned
    map's *smoothed* dynamics, not for the Ito SDE.
    """
    schedule = schedule or LinearSchedule()
    times = _time_grid(t0, t1, steps, device=x.device, dtype=x.dtype)
    log_w = _initial_weights(x, w)
    for step in range(steps):
        s, t = times[step], times[step + 1]
        delta_m = martingale.interval(s, t).terminal_increment()
        x, log_w = _ei_step(problem, x, log_w, s, t, delta_m, schedule, t0)
    return FKCSamples(x, log_w, 0)


class ChenBrownianGrid:
    """Degree-N Brownian coefficients on the FINEST step grid, Chen-merged upward.

    Mirrors training: each of the ``fine_steps`` intervals gets its own
    independently sampled degree-``N`` polynomial on the EI clock (exactly the
    law the loss samples per interval). A coarse step spanning several fine
    intervals uses their exact Chen composition (:meth:`BrownianPolynomial.
    concat_arbitrary`), so every step count rides ONE shared realization and
    every coarse increment is the exact sum of the fine increments.
    """

    def __init__(
        self,
        schedule: object,
        shape: tuple[int, ...],
        fine_steps: int,
        degree: int,
        t0: float = 0.0,
        t1: float = 1.0,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.schedule = schedule
        self.fine_steps = fine_steps
        self.times = torch.linspace(t0, t1, fine_steps + 1, device=device, dtype=dtype)
        clock = schedule.ei_weighted_clock(self.times.double(), t0)
        self.paths = []
        for i in range(fine_steps):
            span = (clock[i + 1] - clock[i]).to(dtype)
            self.paths.append(
                BrownianPolynomial.sample(
                    shape,
                    torch.zeros((), device=device, dtype=dtype),
                    span,
                    degree,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )

    def steps(self, steps: int) -> list:
        """Return ``steps`` Chen-merged paths covering the grid (one per coarse step)."""
        if self.fine_steps % steps != 0:
            raise ValueError(f'{steps} must divide the finest grid {self.fine_steps}.')
        block = self.fine_steps // steps
        merged = []
        for j in range(steps):
            path = self.paths[j * block]
            for k in range(1, block):
                right = self.paths[j * block + k]
                # Shift the right child to start where the running parent ends.
                right = BrownianPolynomial(
                    coeffs=right.coeffs, s=path.t, t=path.t + (right.t - right.s)
                )
                path = BrownianPolynomial.concat_arbitrary(path, right)
            merged.append(path)
        return merged


def learned_fkc(
    net: object,
    x: Tensor,
    w: Tensor | None = None,
    *,
    t0: float = 0.05,
    t1: float = 1.0,
    steps: int = 1,
    degree: int = 3,
    generator: torch.Generator | None = None,
    ess_threshold: float | None = None,
    resample_every_step: bool = False,
    resampling_interval: int | None = None,
    log_weight_clip: float | None = None,
    martingale: GlobalEIMartingalePath | None = None,
    schedule: LinearSchedule | None = None,
    chen_grid: ChenBrownianGrid | None = None,
) -> FKCSamples:
    """Apply the augmented network map ``steps`` times.

    ``chen_grid`` (preferred for coupled strong-error evaluation) supplies
    per-step paths Chen-merged from one finest-grid realization, matching the
    per-interval degree-N law the loss trains on.

    :param net: Augmented backbone with ``forward(x, s, t, coefficients)``.
    :param x: Initial spatial states ``(n, 2)``.
    :param w: Optional initial log weights ``(n,)``.
    :param t0: Positive start time.
    :param t1: Terminal time.
    :param steps: Number of learned steps (the NFE budget).
    :param degree: Brownian coefficient degree for the locally sampled paths.
    :param generator: Optional RNG.
    :param ess_threshold: If set, resample when the ESS drops below this
        fraction. Must be ``None`` when a fixed ``martingale`` is supplied,
        because resampling would require branching the shared realization.
    :param resample_every_step: If ``True``, systematically resample after every
        step (the paper's SMC protocol); fresh future noise is drawn for the
        resampled particles. Also incompatible with a fixed ``martingale``.
    :param martingale: Optional fixed global EI martingale; when given, every
        step rides the same realization (used for coupled WZ comparisons).
    :return: Weighted terminal :class:`FKCSamples`.
    """
    if martingale is not None and (
        ess_threshold is not None or resample_every_step or resampling_interval is not None
    ):
        raise ValueError(
            'resampling is incompatible with a supplied fixed global martingale; '
            'drop ess_threshold/resample_every_step or the martingale argument.'
        )
    schedule = schedule or LinearSchedule()
    times = _time_grid(t0, t1, steps, device=x.device, dtype=x.dtype)
    log_w = _initial_weights(x, w)
    resampling_count = 0
    chen_steps = chen_grid.steps(steps) if chen_grid is not None else None
    for step in range(steps):
        s, t = times[step], times[step + 1]
        if chen_steps is not None:
            path = chen_steps[step]
            coefficients = path.coefficient_features(normalize=net.normalize_coefficients)
            delta_m = path.raw_increment()
        elif martingale is not None:
            interval = martingale.interval(s, t)
            coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
            delta_m = interval.terminal_increment()
        else:
            path = sample_ei_martingale_path(
                schedule,
                tuple(x.shape),
                s,
                t,
                degree,
                t0,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            coefficients = path.coefficient_features(normalize=net.normalize_coefficients)
            delta_m = path.terminal_increment()
        # Weight span multiplying the head's rate (schedule-owned: the nu**2
        # clock for the psi parameterization, the interval length for g).
        weight_span = schedule.weight_span(s, t, t0)
        x, log_w = augmented_fkc_map(
            net, x, log_w, s, t, coefficients, delta_m, weight_span, schedule
        )
        log_w = _clip_log_weights(log_w, log_weight_clip)
        x, log_w, resampled = _resample_if_needed(
            x,
            log_w,
            ess_threshold,
            resample_every_step,
            generator,
            step=step + 1,
            resampling_interval=resampling_interval,
        )
        resampling_count += int(resampled)
    return FKCSamples(x, log_w, resampling_count)


def _ei_step(
    problem: object,
    x: Tensor,
    log_w: Tensor,
    s: Tensor,
    t: Tensor,
    delta_m: Tensor,
    schedule: LinearSchedule,
    t0: float,
) -> tuple[Tensor, Tensor]:
    """One EI-Euler step of the coupled system in lifted coordinates.

    ``alpha_t X_t = alpha_s X_s + 2 A_st u_beta(X_s, s) + DeltaM`` and the
    weight advances by the exact quadrature in the clock ``int nu^2``.
    """
    velocity, g = problem.fields(x, s)
    alpha_s, alpha_t = schedule.alpha(s), schedule.alpha(t)
    lifted = alpha_s * x + 2.0 * schedule._ei_alpha_integral(s, t) * velocity + delta_m
    weight_span = schedule.weight_span(s, t, t0)
    return lifted / alpha_t, log_w + weight_span * schedule.weight_rate(g, s)


