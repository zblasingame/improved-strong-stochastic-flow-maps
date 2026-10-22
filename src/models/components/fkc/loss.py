"""Teacher-driven Wong-Zakai EI / improved-MeanFlow loss for FKC toys.

The augmented map in lifted coordinates ``y = alpha_s x`` (``alpha_t = t``) is

    ``Y_t = Y_s + (t**2 - s**2) u_theta(x_s, s, t; I_{s,t}) + DeltaM_{s,t}``,
    ``w_t = w_s + (t - s) g_theta(x_s, s, t; I_{s,t})``,

where ``DeltaM`` is the exponential-integrator martingale increment and
``I_{s,t}`` the coefficients of ONE global Brownian realization restricted to
``[s, t]``. Writing ``D_s = partial_s + b_Y . grad_y`` for the material
derivative along the smoothed field ``b_Y = 2 s u_beta + alpha_s nu_s Mdot_s``
(terminal ``t`` and the global path held fixed), the Eulerian identities are

    ``u_beta = u_theta - (A_{s,t} / alpha_s) D_s u_theta``,
    ``g_beta = g_theta - (t - s) D_s g_theta``,

with ``A_{s,t} = (t**2 - s**2) / 2``. The loss uses the improved-MeanFlow
graph: the direct heads stay live while the numerical JVP correction is
stop-gradient'd, and a diagonal batch fraction regresses the silenced
diagonal directly onto the analytic teacher fields.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import jvp

from src.models.components.ei_doss import sample_global_ei_martingale_path
from src.models.components.losses.wz import wz_driver
from src.models.components.schedule import AffineSchedule, LinearSchedule


def _huber(r: Tensor, delta: float | None) -> Tensor:
    """Per-sample Huber loss; ``delta=None`` is plain squared error.

    Caps the influence of the heavy-tailed psi/rate residuals near the narrow
    modes, which otherwise blow the update up before gradient clipping helps.
    """
    if delta is None:
        return r.square()
    return torch.where(r.abs() <= delta, 0.5 * r.square(), delta * (r.abs() - 0.5 * delta))


def _adaptive(per_row: Tensor, power: float | None, c: float = 1e-3) -> Tensor:
    """MeanFlow adaptive weighting ``sg(1 / (L + c)**p) * L`` per row.

    ``power=None`` returns ``L`` unchanged. ``power=1`` turns each row's
    squared residual into ~``log(L)`` in gradient, so heavy-tailed rows (early,
    long intervals where the Wong-Zakai driver dominates the target) cannot
    swamp the batch or trigger global gradient clipping every step.
    """
    if power is None:
        return per_row
    return per_row / (per_row.detach() + c) ** power


def _reference_device_dtype(
    net: nn.Module, problem: nn.Module
) -> tuple[torch.device, torch.dtype]:
    """Pick the device/dtype from the net, falling back to the problem's buffers."""
    for module in (net, problem):
        for tensor in list(module.parameters()) + list(module.buffers()):
            return tensor.device, tensor.dtype
    return torch.device('cpu'), torch.get_default_dtype()


class FKCWZIMFLoss(nn.Module):
    """Improved-MeanFlow Wong-Zakai objective for the augmented FKC map.

    The three terms (spatial, log-weight, diagonal) are combined with learned
    homoscedastic uncertainty weights (Kendall et al.), one log-variance per
    term: ``L = sum_i exp(-s_i) L_i + s_i``. This replaces a hand-tuned scale
    with a learned balance, and the ``s_i`` are logged so the rebalancing is
    visible.
    """

    def __init__(
        self,
        t0: float = 0.05,
        degree: int = 3,
        diagonal_fraction: float = 0.25,
        weight_loss_scale: float = 1.0,
        dt: float = 0.05,
        h_max: float = 1.0,
        schedule: AffineSchedule | None = None,
        state_source: str = 'target',
        time_source: str = 'per_sample',
        huber_delta: float | None = None,
        pool_num_s: int = 64,
        pool_per_s: int = 2048,
        pool_s_range: tuple[float, float] | None = None,
        late_fraction: float = 0.0,
        late_start: float = 0.75,
        adaptive_power: float | None = None,
    ) -> None:
        """Configure the loss.

        :param t0: Positive clock origin and smallest start time.
        :param degree: Brownian polynomial degree of the global path.
        :param diagonal_fraction: Fraction of the batch used for the diagonal
            anchor instead of the Eulerian identity.
        :param weight_loss_scale: Fixed base scale of the log-weight residual
            (multiplied by the learned uncertainty weight).
        """
        super().__init__()
        if not 0.0 <= t0 < 1.0:
            raise ValueError(f't0 must lie in [0, 1); got {t0}.')
        if state_source not in ('target', 'expert', 'target_cheap', 'pool'):
            raise ValueError(
                f"state_source must be 'target', 'expert', 'target_cheap' or 'pool'; "
                f'got {state_source!r}.'
            )
        if time_source not in ('per_sample', 'scalar'):
            raise ValueError(f"time_source must be 'per_sample' or 'scalar'; got {time_source!r}.")
        if degree < 1:
            raise ValueError(f'degree must be at least 1; got {degree}.')
        if not 0.0 <= diagonal_fraction <= 1.0:
            raise ValueError(f'diagonal_fraction must lie in [0, 1]; got {diagonal_fraction}.')
        if weight_loss_scale <= 0.0:
            raise ValueError(f'weight_loss_scale must be positive; got {weight_loss_scale}.')
        if dt <= 0.0:
            raise ValueError(f'dt must be positive; got {dt}.')
        if h_max < dt:
            raise ValueError(f'h_max must be >= dt; got h_max={h_max}, dt={dt}.')
        self.t0 = t0
        self.degree = degree
        self.diagonal_fraction = diagonal_fraction
        self.weight_loss_scale = weight_loss_scale
        self.dt = dt
        self.h_max = h_max
        self.schedule: AffineSchedule = schedule or LinearSchedule()
        self.state_source = state_source
        self.time_source = time_source
        self.huber_delta = huber_delta
        self.pool_num_s = pool_num_s
        self.pool_per_s = pool_per_s
        self.pool_s_range = (t0, 1.0) if pool_s_range is None else tuple(pool_s_range)
        # Oversample the late interval (s >= late_start) where the learned map
        # compounds error under resampling; see FKC particle-sweep diagnostics.
        self.late_fraction = late_fraction
        self.late_start = late_start
        self.adaptive_power = adaptive_power

    def _sample_times(
        self, batch: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Draw per-sample ``(s, t)`` and a diagonal mask (WZ-style sampling).

        Follows ``WZSSFMLoss._sample_times``: draw ``t`` first, then the step
        length ``h = t - s`` in ``[dt, min(h_max, t - t0)]``, so ``s >= t0`` and
        the interval never collapses or grows unbounded.
        """
        t = (
            self.t0
            + self.dt
            + (1.0 - self.t0 - self.dt)
            * torch.rand(batch, device=device, dtype=dtype, generator=generator)
        )
        upper = (t - self.t0).clamp(min=self.dt, max=self.h_max)
        h = self.dt + (upper - self.dt) * torch.rand(
            batch, device=device, dtype=dtype, generator=generator
        )
        s = t - h
        diagonal = torch.rand(batch, device=device, dtype=dtype, generator=generator) < (
            self.diagonal_fraction
        )
        # Keep diagonal entries strictly inside (s, 1] so the interval machinery
        # stays valid; their loss is selected from the separate diagonal term.
        t = torch.where(diagonal, s + (1.0 - s) * 1e-3, t)
        return s, t, diagonal

    def _score_forward(
        self,
        net: nn.Module,
        problem: nn.Module,
        x_s: Tensor,
        s: Tensor,
        t: Tensor,
        diagonal: Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, Tensor]:
        """Score-parameterized iMF matching the inference map exactly.

        The map (:func:`augmented_fkc_map` via ``velocity_from_prediction``) is

            ``x_t = x_s + C_st * S(eps_bar),  S(e) = -e / k_s``,
            ``C_st = sigma_s**2 - sigma_t**2``  (the VE clock ``int nu**2``),
            ``w_t = w_s + Lam_st * q_bar,  Lam_st = log(k_s**2 / k_t**2)``,

        with ``k**2 = sigma**2 + m**2``, so the network output is an interval
        average of the scaled score over the clock ``C`` (spatial) and of the
        scaled potential ``q = g k**2 / nu**2`` over the clock ``Lam`` (weight).
        Both clocks satisfy ``d C/ds = -nu_s**2`` and ``d Lam/ds = -nu_s**2/k_s**2``.

        Differentiating ``x_t = x_s + C_st S`` along the random-ODE velocity
        ``beta`` (with ``D_s x_t = 0``) gives the Eulerian identity

            ``beta = nu_s**2 S - C_st D_s S``,

        where ``D_s S`` includes the explicit ``k_s`` dependence. Expressed in
        noise units (multiply by ``k_s / nu_s**2``) the residual is O(1):

            ``k_s beta / nu_s**2 = -eps_bar' + (C_st / nu_s**2) k_s D_s S``.

        We regress the live prediction ``S`` against this with the JVP term
        stop-gradient'd (improved MeanFlow). The weight block is analogous with
        clock ``Lam``: ``q = q_bar - (Lam_st k_s**2 / nu_s**2) D_s q_bar``.
        """
        from src.models.components.brownian import BrownianPolynomial

        from .network import noise_from_score, potential_scale, potential_span, score_noise_scale

        schedule = self.schedule
        dtype = x_s.dtype
        floor2 = net.score_scale_floor**2
        zero = x_s.new_zeros(())
        imf_x = imf_w = fm_x = fm_w = zero
        imf_x_opt = imf_w_opt = fm_x_opt = fm_w_opt = zero

        def composed_score(x: Tensor, r: Tensor) -> Tensor:
            return sum(
                power * value
                for power, value in zip(problem.powers, problem.scores(x, r), strict=True)
            )

        # Pool s may reach 1, making t == s; such rows carry no interval.
        off = (~diagonal) & (t - s > 1e-6)
        if bool(off.any()):
            xo, so, to = x_s[off], s[off], t[off]
            n = xo.shape[0]
            with torch.no_grad():
                score = composed_score(xo, so)
                eps_teacher = noise_from_score(net, score, schedule, so)
                _, g_teacher = problem.fields(xo, so)
                q_teacher = g_teacher / potential_scale(net, schedule, so)

            # Local EI-clock coordinates, float64-offset like the PSD loss: the
            # VE clock is ~100 near t=1 where valid spans fall below float32 ULP.
            def clock64(r: Tensor) -> Tensor:
                return schedule.ei_weighted_clock(r.double(), self.t0)

            anchor = clock64(so).detach()
            span = (clock64(to) - anchor).to(dtype).reshape(-1, 1)
            parent = BrownianPolynomial.sample(
                (n, 2), torch.zeros_like(span), span, self.degree,
                device=xo.device, dtype=dtype, generator=generator,
            )

            def child(start: Tensor) -> BrownianPolynomial:
                # Same realization restricted to [clock(start), clock(t)] in
                # local coordinates: value 0 at start=s, tangent dclock/ds.
                offset = (clock64(start) - anchor).to(dtype).reshape(-1, 1)
                return parent.restrict(offset, span)

            start = so.detach().contiguous()
            # Wong-Zakai driver Mdot_s = (dclock/ds) Wdot at the interval start.
            _, clock_rate = jvp(lambda r: clock64(r).to(dtype), (start,), (torch.ones_like(start),))
            driver = clock_rate[:, None] * parent.derivative(torch.zeros_like(span))
            nu2_s = schedule.nu_squared(so)
            beta = (nu2_s[:, None] * score + driver).detach()

            def heads(x: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
                features = child(r).coefficient_features(normalize=net.normalize_coefficients)
                eps_bar, q_bar = net(x, r, to, features)
                # Physical interval-average score S = -eps_bar / k_r.
                return -eps_bar / score_noise_scale(net, schedule, r)[:, None], q_bar

            (s_bar, q_bar), (d_s_bar, d_q) = jvp(
                heads, (xo, start), (beta, torch.ones_like(start))
            )
            k_s = score_noise_scale(net, schedule, so)
            clock_st = schedule.sigma(so).square() - schedule.sigma(to).square()
            lam_st = potential_span(net, schedule, so, to)
            # nu_s**2 S - C D_s S = nu_s**2 score  -> noise units (x -k_s / nu_s**2).
            residual_x = (
                -k_s[:, None] * s_bar
                + (k_s * clock_st / nu2_s)[:, None] * d_s_bar.detach()
                - eps_teacher
            )
            # q_bar - (Lam k_s**2 / nu_s**2) D_s q_bar = q.
            lam_coeff = lam_st * (schedule.sigma(so).square() + floor2) / nu2_s
            residual_w = q_bar - lam_coeff * d_q.detach() - q_teacher
            raw_x = _huber(residual_x, self.huber_delta).sum(-1)
            raw_w = _huber(residual_w, self.huber_delta)
            # Logged values stay the raw means; the optimized loss is weighted.
            imf_x_opt = _adaptive(raw_x, self.adaptive_power).mean()
            imf_w_opt = _adaptive(raw_w, self.adaptive_power).mean()
            imf_x, imf_w = raw_x.mean(), raw_w.mean()

        if bool(diagonal.any()):
            xd, sd = x_s[diagonal], s[diagonal]
            with torch.no_grad():
                eps_teacher_d = noise_from_score(net, composed_score(xd, sd), schedule, sd)
                _, g_d = problem.fields(xd, sd)
                q_teacher_d = g_d / potential_scale(net, schedule, sd)
            eps_diag, q_diag = net(xd, sd, sd, xd.new_zeros(xd.shape[0], 1, 2))
            raw_fx = _huber(eps_diag - eps_teacher_d, self.huber_delta).sum(-1)
            raw_fw = _huber(q_diag - q_teacher_d, self.huber_delta)
            fm_x_opt = _adaptive(raw_fx, self.adaptive_power).mean()
            fm_w_opt = _adaptive(raw_fw, self.adaptive_power).mean()
            fm_x, fm_w = raw_fx.mean(), raw_fw.mean()

        fm_loss = fm_x + self.weight_loss_scale * fm_w
        imf_loss = imf_x + self.weight_loss_scale * imf_w
        total = (
            imf_x_opt + self.weight_loss_scale * imf_w_opt
            + fm_x_opt + self.weight_loss_scale * fm_w_opt
        )
        return {
            'loss': total,
            'imf_x': imf_x.detach(),
            'imf_w': imf_w.detach(),
            'imf_loss': imf_loss.detach(),
            'fm_x': fm_x.detach(),
            'fm_w': fm_w.detach(),
            'fm_loss': fm_loss.detach(),
            'spatial_loss': imf_x.detach(),
            'weight_loss': imf_w.detach(),
            'diagonal_loss': fm_loss.detach(),
        }

    def forward(
        self,
        net: nn.Module,
        problem: nn.Module,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Return the total loss and detached diagnostics for one step."""
        if batch_size < 1:
            raise ValueError(f'batch_size must be positive; got {batch_size}.')
        device, dtype = _reference_device_dtype(net, problem)
        # None deliberately uses PyTorch's advancing global RNG. Constructing a
        # fresh Generator here resets to the same default seed on EVERY batch.

        if self.state_source == 'pool':
            # Draw (x_s, s) jointly from the cached exact-target pool: the
            # 40**3 = 64k-component target is built once, not per step.
            if getattr(problem, '_state_pool', None) is None:
                problem.build_state_pool(
                    num_s=self.pool_num_s,
                    per_s=self.pool_per_s,
                    s_range=self.pool_s_range,
                    generator=generator,
                )
            x_s, s = problem.sample_pool(
                batch_size,
                generator=generator,
                late_fraction=self.late_fraction,
                late_start=self.late_start,
            )
            upper = (1.0 - s).clamp(max=self.h_max)
            lower = upper.clamp(max=self.dt)
            h = lower + (upper - lower) * torch.rand(
                batch_size, device=device, dtype=dtype, generator=generator
            )
            t = s + h
            diagonal = (
                torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
                < self.diagonal_fraction
            )
            t = torch.where(diagonal, s + (1.0 - s) * 1e-3, t)
            x_s = x_s.to(device=device, dtype=dtype)
        else:
            s, t, diagonal = self._sample_times(
                batch_size, device=device, dtype=dtype, generator=generator
            )
            if self.state_source == 'expert':
                x_s = problem.sample_expert_batch(s, generator=generator)
            elif self.state_source == 'target_cheap':
                x_s = problem.sample_target_cheap(s, generator=generator)
            elif self.time_source == 'scalar':
                x_s = problem.target(float(s[0])).sample(batch_size, 1.0, generator=generator)
            else:
                x_s = problem.sample_target_batch(s, generator=generator)
            x_s = x_s.to(device=device, dtype=dtype)
        if getattr(net, 'output_parameterization', 'velocity') == 'score':
            return self._score_forward(net, problem, x_s, s, t, diagonal, generator)
        y_s = self.schedule.alpha(s)[:, None] * x_s
        martingale = sample_global_ei_martingale_path(
            self.schedule,
            (batch_size, 2),
            self.t0,
            self.degree,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        u_teacher, g_teacher = (value.detach() for value in problem.fields(x_s, s))
        # `wz_driver` differentiates through the time tensor, so the per-sample
        # times must be a contiguous (non-broadcast) tensor for forward AD.
        start = s.detach().contiguous()
        driver = wz_driver(martingale, start, y_s).detach()
        direction = (2.0 * self.schedule.alpha(s)[:, None] * u_teacher + driver).detach()

        def heads(y: Tensor, start_t: Tensor) -> tuple[Tensor, Tensor]:
            interval = martingale.interval(start_t, t)
            coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
            return net(y / self.schedule.alpha(start_t)[:, None], start_t, t, coefficients)

        (velocity, rate), (d_velocity, d_rate) = jvp(
            heads, (y_s, start), (direction, torch.ones_like(start))
        )
        span = 2.0 * self.schedule._ei_alpha_integral(start, t)
        off_diagonal = (~diagonal).to(dtype)
        # Lifted Y units (matching WZ-EI iMF): `lift * u - span * D_s u` keeps the
        # residual O(1) instead of amplifying it like the physical `span / (2 s)`.
        lift = 2.0 * self.schedule.alpha(start)
        residual_velocity = (
            lift[:, None] * velocity
            - span[:, None] * d_velocity.detach()
            - lift[:, None] * u_teacher
        )
        # The scalar head predicts either the bounded potential g (when the
        # schedule's nu_t**2 stays finite, e.g. variance-exploding) or the clock
        # integrand psi = g / nu_t**2; the schedule owns which span multiplies
        # the rate and its iMF JVP correction.
        psi_teacher = self.schedule.weight_rate(g_teacher, start)
        rate_jvp_span = self.schedule.weight_rate_jvp_span(start, t, self.t0)
        residual_rate = rate - rate_jvp_span * d_rate.detach() - psi_teacher
        n_off = off_diagonal.sum().clamp_min(1.0)
        spatial_loss = (
            _huber(residual_velocity, self.huber_delta).sum(-1) * off_diagonal
        ).sum() / n_off
        weight_loss = (_huber(residual_rate, self.huber_delta) * off_diagonal).sum() / n_off

        diagonal_mask = diagonal.to(dtype)
        diagonal_coefficients = x_s.new_zeros(batch_size, 1, 2)
        velocity_diag, rate_diag = net(x_s, s, s, diagonal_coefficients)
        n_diag = diagonal_mask.sum().clamp_min(1.0)
        # Lift the diagonal anchor too: physical u ~ x / t diverges as t -> 0,
        # while lift * u stays O(1) (same reason as the off-diagonal residual).
        diagonal_spatial = (
            _huber(lift[:, None] * (velocity_diag - u_teacher), self.huber_delta).sum(-1)
            * diagonal_mask
        ).sum() / n_diag
        diagonal_rate = (
            _huber(rate_diag - psi_teacher, self.huber_delta) * diagonal_mask
        ).sum() / n_diag
        diagonal_loss = diagonal_spatial + self.weight_loss_scale * diagonal_rate

        # Fixed per-term weights (no learned uncertainty scaling): every term is
        # actually optimized, including the spatial iMF residual that the
        # learned log-variances had down-weighted by ~200x.
        total = spatial_loss + self.weight_loss_scale * weight_loss + diagonal_loss
        return {
            'loss': total,
            # flow-map (iMF, 75% off-diagonal) and flow-matching (diagonal,
            # 25%) components, each split by the augmented block x / w.
            'imf_x': spatial_loss.detach(),
            'imf_w': weight_loss.detach(),
            'imf_loss': (spatial_loss + weight_loss).detach(),
            'fm_x': diagonal_spatial.detach(),
            'fm_w': diagonal_rate.detach(),
            'fm_loss': diagonal_loss.detach(),
            # backward-compatible aliases used by the tests
            'spatial_loss': spatial_loss.detach(),
            'weight_loss': weight_loss.detach(),
            'diagonal_loss': diagonal_loss.detach(),
        }
