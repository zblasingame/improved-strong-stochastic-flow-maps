"""Direct/MG DMF anchor plus the Wong-Zakai EI iMF compound objective.

This module keeps the Wong-Zakai (WZ) treatment of
:mod:`src.models.components.losses.wz_ei`: the EI SDE

    ``dX = [(1+z^2) u - z^2 (alphadot/alpha) X] dt + z nu dW``

is lifted to the annealed state ``Y = alpha^{z^2} X``, where it is the
additive-noise RODE

    ``dY/dr = (1+z^2) alpha_r^{z^2} u_r + Mdot~^z_r``,

and the martingale ``M~^z = z int alpha^{z^2} nu dW`` is realized as a
degree-``N`` polynomial Brownian motion on the annealed EI clock. The
polynomial is of bounded variation, so the SDE becomes a smooth random ODE
and the flow map owns its increment explicitly (no reconstruction):

    ``Y_t = Y_s + (1+z^2) A^(z)_{s,t} u_theta(X_s, s, t; I^M) + DM~^z``,

the exact EI-Euler SDE step in Y units (``wz_ei_y_map``). Differentiating
the map in ``(Y_s, s)`` along the smoothed lifted field
``b_Y^z = (1+z^2) alpha_s^{z^2} u_s + Mdot~^z_s`` and setting the Eulerian
condition ``D_s Psi = 0`` gives the iMF identity (module docstring of
:mod:`src.models.components.losses.wz_ei`)

    ``lift_s * u_s = lift_s * u - a_st * D_s u``,
    ``lift_s = (1+z^2) alpha_s^{z^2}``,
    ``a_st = (1+z^2) A^(z)_{s,t}``.

Following improved MeanFlow, only the JVP output ``D_s u`` is stopped; the
direct interval prediction ``u`` stays live. For DMF distillation the
instantaneous start field ``u_s`` is the stopped *teacher velocity*
``v_tgt`` — the same field the DMF anchor distills on the diagonal
(:func:`dmf_guided_velocity` for ``guidance='mg'``, or one stopped forward
of the configured source for ``guidance='direct'``) — so the compound
regresses ``lift_s * v_tgt`` while the JVP transports along
``b_Y^z = lift_s * v_tgt + Mdot~^z_s`` (driver included, exactly as the
standalone WZ objectives).

The remaining ``(1 - p)`` batch fraction runs the standard DMF diagonal
anchor (:func:`dmf_model_guidance_anchor_loss` /
:func:`dmf_direct_anchor_loss`), numerically identical to the anchor of
:class:`~src.models.components.losses.dmf_ei_euler.DMFEIEulerAnchorPSDLoss`.
There is no teacher dependence in the iMF term beyond the stopped target
and no diagonal branch in the distill split. Optionally, both branch losses
are reweighted by the learned Kendall et al. log-variance
:class:`~src.models.components.losses.uncertainty.UncertaintyMLP` exactly as
in :class:`~src.models.components.losses.dmf_ei_euler.DMFEIEulerAnchorPSDLoss`
(``use_uncertainty_weighting``; the iMF branch reads ``l_{s,t}`` at its
interval endpoints, the anchor at ``(t, t)``).

Direct guidance
---------------
``guidance='direct'`` exploits that the upstream DMF checkpoint was already
trained with model guidance (``g_type='mg'``): its conditional forward *is*
the guided field — ``v_data + omega (v_cond - v_uncond)`` in ``[g_min,
g_max]`` for kept labels, the plain velocity elsewhere, no nudge for the
null label. The target therefore collapses to a single frozen teacher
forward ``anchor_prediction(x, t, y_student)`` (1 teacher NFE per sample
instead of the cond/uncond pair), evaluated with the student's currently
assigned labels so dropped samples read the unconditional teacher.
``omega``, ``g_min``, and ``g_max`` are inherited from the checkpoint and
unused in this mode.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.func import jvp

from src.models.components.ei_doss import (
    GlobalEIMartingalePath,
    ei_x_state,
    ei_y_state,
    sample_global_ei_martingale_path,
)
from src.models.components.fixed_coefficient_ssfm import FixedCoefficientDMFEIEulerSSFM
from src.models.components.losses.wz import _per_sample_mse, _require_euler, wz_driver
from src.models.components.schedule import AffineSchedule, expand_t
from src.models.components.ssfm import SSFMNet

from .dmf_ei_euler import dmf_direct_field, dmf_guided_velocity, dmf_model_guidance_anchor_loss
from .uncertainty import UncertaintyMLP, apply_uncertainty
from .wz_ei import wz_ei_y_map

_DMF_ADAPTERS = (FixedCoefficientDMFEIEulerSSFM,)


class DMFWZEIIMFAnchorLoss(nn.Module):
    """DMF anchor mixed with the Wong-Zakai EI improved-MeanFlow compound.

    A fixed ``p`` batch split: the leading fraction trains the degree-``N``
    polynomial-Brownian WZ random-ODE map with the iMF compound
    ``lift_s * u - a_st * sg(D_s u)`` regressed against the DMF teacher
    field; the remainder distills that same field on the diagonal through
    the standard DMF anchor.

    :param schedule: Affine noise schedule providing the EI coefficients
        and the (annealed) EI martingale clock.
    :param p: Fraction of each batch routed to the WZ-iMF compound.
    :param degree: Polynomial Brownian degree of the WZ martingale (the
        fixed-coefficient adapter consumes exactly this many features).
    :param epsilon: Clock origin ``t0`` (``M_{t0} = 0``); interval endpoints
        stay in ``[epsilon, 1]``.
    :param dt: Minimum sampled interval length ``h = t - s``.
    :param h_max: Maximum sampled interval length.
    :param cfg_prob: Label-drop probability for the student (classifier-free).
    :param guidance: ``'mg'`` (guidance-mixed teacher velocity) or ``'direct'``
        (one stopped forward of the configured source).
    :param guidance_source: ``'ema'`` or ``'frozen'`` field source; ``None``
        keeps the per-form default (``mg`` -> ``'ema'``, else ``'frozen'``).
    :param omega: Model-guidance strength ``v + omega (v_cond - v_uncond)``
        (unused for ``'direct'``).
    :param g_min: Guidance interval lower endpoint (project time; unused
        for ``'direct'``).
    :param g_max: Guidance interval upper endpoint (project time; unused
        for ``'direct'``).
    :param zeta_anneal_steps: Ramp ``zeta`` from 0 (probability-flow ODE) to
        1 (flow-as-SDE) over this many optimizer steps; 0 disables annealing.
    :param use_uncertainty_weighting: Learn a per-sample log-variance
        ``l_{s,t}`` (Kendall et al.) and reweight both branch losses with it.
    :param uncertainty_form: ``'log_eps'`` (numerically stable, default) or
        ``'reference'`` (plain Boffi et al. form).
    :param fourier_dim: Random Fourier features per endpoint of the
        uncertainty MLP.
    """

    def __init__(
        self,
        schedule: AffineSchedule,
        p: float = 0.75,
        degree: int = 3,
        epsilon: float = 0.01,
        dt: float = 0.05,
        h_max: float = 1.0,
        cfg_prob: float = 0.1,
        guidance: str = 'mg',
        guidance_source: str | None = None,
        omega: float = 0.6,
        g_min: float = 0.3,
        g_max: float = 1.0,
        zeta_anneal_steps: int = 0,
        use_uncertainty_weighting: bool = False,
        uncertainty_form: str = 'log_eps',
        fourier_dim: int = 256,
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1]; got {p}.')
        if degree < 1:
            raise ValueError(f'degree must be >= 1; got {degree}.')
        if epsilon <= 0.0:
            raise ValueError(f'WZ-EI needs epsilon > 0 (clock origin); got {epsilon}.')
        if dt < 0.0:
            raise ValueError(f'dt must be >= 0; got {dt}.')
        if h_max < dt:
            raise ValueError(f'h_max must be >= dt; got h_max={h_max}, dt={dt}.')
        if guidance not in ('direct', 'mg'):
            raise ValueError(f"guidance must be 'direct' or 'mg'; got {guidance!r}.")
        if guidance_source not in (None, 'ema', 'frozen'):
            raise ValueError(
                f"guidance_source must be 'frozen', 'ema', or None; got {guidance_source!r}."
            )
        if guidance_source is None:
            guidance_source = 'ema' if guidance == 'mg' else 'frozen'
        if omega < 0.0:
            raise ValueError(f'omega must be >= 0; got {omega}.')
        if not 0.0 <= g_min <= g_max <= 1.0:
            raise ValueError(
                f'guidance interval must satisfy 0 <= g_min <= g_max <= 1; got [{g_min}, {g_max}].'
            )
        if zeta_anneal_steps < 0:
            raise ValueError(f'zeta_anneal_steps must be >= 0; got {zeta_anneal_steps}.')
        if uncertainty_form not in ('log_eps', 'reference'):
            raise ValueError(
                f"uncertainty_form must be 'log_eps' or 'reference'; got {uncertainty_form!r}."
            )
        self.schedule = schedule
        self.p = p
        self.degree = degree
        self.epsilon = epsilon
        self.dt = dt
        self.h_max = h_max
        self.t0 = epsilon
        self.cfg_prob = cfg_prob
        self.guidance = guidance
        self.guidance_source = guidance_source
        self.omega = omega
        self.g_min = g_min
        self.g_max = g_max
        self.zeta_anneal_steps = zeta_anneal_steps
        self.uncertainty_form = uncertainty_form
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty = (
            UncertaintyMLP(fourier_dim=fourier_dim) if use_uncertainty_weighting else None
        )

    def zeta_at(self, global_step: int) -> float:
        """Return the annealing amplitude ``zeta`` at a given optimizer step.

        ``zeta_anneal_steps = 0`` (default) disables annealing (``zeta = 1``,
        the un-annealed flow-as-SDE). Otherwise ``zeta`` ramps linearly from
        0 (the probability-flow ODE) to 1 (the flow-as-SDE) over the first
        ``zeta_anneal_steps`` optimizer steps.
        """
        if self.zeta_anneal_steps <= 0:
            return 1.0
        return min(1.0, global_step / self.zeta_anneal_steps)

    def _sample_times(self, batch: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Sample ``t ~ U[eps+dt, 1]`` then ``h ~ U[dt, min(h_max, t-eps)]``.

        Times are always sampled in float32 regardless of the reference
        dtype, mirroring the EI-Doss convention. The start time is clamped
        back onto the clock origin: ``s = t - h`` can round one float32 ulp
        below ``epsilon`` (observed ~1 batch in 2e4 at batch 64 on the
        cluster), and the martingale views/guards reject any ``s < t0``
        outright -- :meth:`GlobalMartingalePath.interval` /
        :meth:`GlobalMartingalePath.global_value` raise instead of nudging,
        so a single hit kills the run. The clamp moves ``s`` by at most one
        ulp (``~1e-10``), leaving the sampling law untouched.
        """
        t = (
            self.epsilon
            + self.dt
            + (1.0 - self.epsilon - self.dt)
            * torch.rand(batch, device=reference.device, dtype=torch.float32)
        )
        upper = (t - self.epsilon).clamp(min=self.dt, max=self.h_max)
        h = self.dt + (upper - self.dt) * torch.rand_like(t)
        s = (t - h).clamp_min(self.epsilon)
        return s, t

    def _sample_path(self, x: Tensor, zeta: float, degree: int) -> GlobalEIMartingalePath:
        """Sample one global (possibly annealed) EI martingale covering ``[t0, 1]``.

        The path is always sampled in float32 regardless of the batch dtype:
        every annealed clock *difference* (the sampling span, the interval
        restriction, and the :func:`wz_driver` chain rule through the clock)
        follows the path dtype, and those differences are exactly what
        underflow/round away in bf16 for small ``zeta`` (the repo's
        "keep annealed runs in float32" convention). The coefficient features
        are cast back down by the backbone's autocast embedder, so a bf16
        training run keeps its bf16 Triton dual-JVP attention while the
        martingale realization itself stays float32-exact.
        """
        return sample_global_ei_martingale_path(
            self.schedule,
            tuple(x.shape),
            t0=self.t0,
            degree=degree,
            zeta=zeta,
            device=x.device,
            dtype=torch.float32,
        )

    def _teacher_field(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_0: Tensor,
        x_1: Tensor,
        x: Tensor,
        t: Tensor,
        cond_labels: Tensor,
        guided: Tensor,
    ) -> Tensor:
        """Return the stopped DMF velocity field distilled at state ``(x, t)``.

        ``'direct'`` evaluates one stopped forward of the configured source
        (``frozen`` the pretrained teacher's conditional forward — the
        upstream checkpoint's model guidance is baked in, so dropped
        null-label samples read the unconditional teacher; ``ema`` the EMA
        shadow's diagonal head, the same shadow the PSD semigroup target
        uses). ``'mg'`` returns the guidance-mixed velocity of
        :func:`dmf_guided_velocity` (``v_data + omega (v_cond - v_uncond)``
        inside ``[g_min, g_max]`` for guided samples, the plain conditional
        velocity otherwise). All are computed under ``no_grad`` (stopped).
        """
        if self.guidance == 'direct':
            return dmf_direct_field(net, ema_net, x, t, self.guidance_source)
        return dmf_guided_velocity(
            net,
            ema_net,
            self.schedule,
            x_0,
            x_1,
            x,
            t,
            cond_labels,
            guided,
            self.omega,
            self.g_min,
            self.g_max,
            self.guidance_source,
        )

    def _imf_loss(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        s: Tensor,
        t: Tensor,
        cond_labels: Tensor,
        guided: Tensor,
        zeta: float,
        degree: int,
    ) -> Tensor:
        """Per-sample WZ-EI iMF compound MSE on one interval set."""
        x_s = self.schedule.sample_x_t(x_0, x_1, s)
        y_s = ei_y_state(self.schedule, x_s, s, zeta)
        lift_s = expand_t((1.0 + zeta * zeta) * self.schedule.alpha(s) ** (zeta * zeta), y_s)
        v_field = self._teacher_field(net, ema_net, x_0, x_1, x_s, s, cond_labels, guided).detach()
        # Full smoothed lifted field: the WZ driver enters the transport
        # tangent and cancels in the compound identity, so the regression
        # target stays the deterministic start field (module docstring).
        martingale = self._sample_path(x_s, zeta, degree)
        driver_s = wz_driver(martingale, s, y_s).detach()
        b_y = (lift_s * v_field + driver_s).detach()

        def _interval_velocity(y: Tensor, start: Tensor) -> Tensor:
            interval = martingale.interval(start, t)
            coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
            x = ei_x_state(self.schedule, y, start, zeta)
            return net.drift(x, start, t, coefficients)

        u, material_derivative = jvp(_interval_velocity, (y_s, s), (b_y, torch.ones_like(s)))
        a_st = expand_t(
            (1.0 + zeta * zeta) * self.schedule.annealed_ei_alpha_integral(s, t, zeta),
            u,
        )
        compound = lift_s * u - a_st * material_derivative.detach()
        target = (lift_s * v_field).detach()
        return _per_sample_mse(compound, target)

    def _anchor_loss(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        t: Tensor,
        cond_labels: Tensor,
        guided: Tensor,
    ) -> Tensor:
        """Dispatch the diagonal anchor to the configured guidance form.

        ``'direct'`` regresses the student's silenced diagonal against one
        stopped forward of the configured source at the same student labels
        (null labels read the unconditional source), matching the field the
        iMF compound regresses at the interval start.
        """
        if self.guidance == 'direct':
            x_t = self.schedule.sample_x_t(x_0, x_1, t)
            target = dmf_direct_field(net, ema_net, x_t, t, self.guidance_source)
            return _per_sample_mse(net.anchor_student_prediction(x_t, t), target)
        return dmf_model_guidance_anchor_loss(
            net,
            ema_net,
            self.schedule,
            x_1,
            x_0,
            t,
            cond_labels,
            guided,
            self.omega,
            self.g_min,
            self.g_max,
            self.guidance_source,
        )

    def _reweight(self, raw: Tensor, s: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(weighted, log_var)`` for one mixer branch.

        Without ``use_uncertainty_weighting`` the raw per-sample losses pass
        through untouched and the log-variance is reported as zeros.
        """
        if not self.use_uncertainty_weighting:
            return raw, raw.new_zeros(raw.shape)
        if self.uncertainty is None:
            raise RuntimeError('Uncertainty weighting is enabled without an uncertainty module.')
        log_var = self.uncertainty(s, t)
        return apply_uncertainty(raw, log_var, form=self.uncertainty_form), log_var

    def forward(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the fixed-ratio WZ-iMF/anchor loss and diagnostics.

        The annealing amplitude is ``zeta = zeta_at(global_step)``; both the
        iMF compound and its martingale realization ride the same ``zeta``.
        With ``use_uncertainty_weighting`` the per-sample branch losses are
        reweighted by the learned log-variance before the batch means; the
        diagnostics always report the raw (unweighted) means.
        """
        if not isinstance(net, _DMF_ADAPTERS):
            raise TypeError(
                'DMFWZEIIMFAnchorLoss requires a DMF EI adapter '
                '(the fixed-coefficient DMF EI-Euler student).'
            )
        _require_euler(net)
        batch = x_1.shape[0]
        n_imf = int(batch * self.p)
        zeta = self.zeta_at(global_step)
        s, t = self._sample_times(batch, x_1)
        labels = net._active_labels(x_1)
        drop_ids = torch.rand(batch, device=x_1.device) < self.cfg_prob
        student_labels = torch.where(drop_ids, net.num_classes, labels)
        degree = self.degree
        try:
            net.set_labels(student_labels[:n_imf])
            ema_net.set_labels(student_labels[:n_imf])
            raw_imf = (
                self._imf_loss(
                    net,
                    ema_net,
                    x_1[:n_imf],
                    x_0[:n_imf],
                    s[:n_imf],
                    t[:n_imf],
                    labels[:n_imf],
                    ~drop_ids[:n_imf],
                    zeta,
                    degree,
                )
                if n_imf
                else None
            )
            net.set_labels(student_labels[n_imf:])
            ema_net.set_labels(student_labels[n_imf:])
            raw_anchor = (
                self._anchor_loss(
                    net,
                    ema_net,
                    x_1[n_imf:],
                    x_0[n_imf:],
                    t[n_imf:],
                    labels[n_imf:],
                    ~drop_ids[n_imf:],
                )
                if n_imf < batch
                else None
            )
        finally:
            net.set_labels(labels)
            ema_net.set_labels(labels)
        if raw_imf is not None:
            weighted_imf, u_imf = self._reweight(raw_imf, s[:n_imf], t[:n_imf])
        else:
            weighted_imf, u_imf = None, x_1.new_zeros(0)
        if raw_anchor is not None:
            weighted_anchor, u_anchor = self._reweight(raw_anchor, t[n_imf:], t[n_imf:])
        else:
            weighted_anchor, u_anchor = None, x_1.new_zeros(0)
        imf = weighted_imf.mean() if weighted_imf is not None else x_1.new_zeros(())
        anchor = weighted_anchor.mean() if weighted_anchor is not None else x_1.new_zeros(())
        total = self.p * imf + (1.0 - self.p) * anchor
        diagnostics = {
            'anchor_loss': (
                raw_anchor.mean().detach() if raw_anchor is not None else anchor.detach()
            ),
            'distillation_loss': (
                raw_imf.mean().detach() if raw_imf is not None else imf.detach()
            ),
            'effective_p': total.new_tensor(self.p),
            'n_distill': total.new_tensor(n_imf),
            'path_degree': total.new_tensor(degree),
            'effective_cfg_prob': drop_ids.float().mean().detach(),
            'guidance_omega': total.new_tensor(self.omega),
            'zeta': total.new_tensor(zeta),
        }
        if self.use_uncertainty_weighting:
            diagnostics['uncertainty_mlp'] = torch.cat((u_imf, u_anchor)).mean().detach()
        return total, diagnostics


class DMFWZEILMDAnchorLoss(DMFWZEIIMFAnchorLoss):
    """AYF-LMD bilinear Lagrangian objective on the lifted WZ-EI map.

    The 75% branch replaces the improved-MeanFlow compound with the
    paper-faithful AYF-LMD bilinear tangent objective (Theorem 3.3 of
    Sabour et al. 2025, arXiv:2506.14603) applied to the lifted EI rODE.
    Differentiating the map ``Y_t = Y_s + (1+z^2) A^(z)_{s,t} u + DM~^z`` in
    ``t`` gives ``partial_t Psi = lift_t u + (1+z^2) A^(z) partial_t u +
    Mdot~^z_t``, and the annealed Lagrangian condition ``partial_t Psi =
    b_Y(Y_t) = lift_t v(X_t) + Mdot~^z_t`` cancels the drivers on both sides,
    leaving the residual

        ``r = sg(partial_t Psi) - (lift_t v + Mdot~^z_t)``

    whose bilinear surrogate ``(t - s) <Psi, r>`` is the repo-time spelling
    of the paper's ``w'(t, s) sign(s - t) f_theta^T (d f_{theta-} / ds -
    v_phi(f_{theta-}, s))`` with ``w' = |t - s|`` (the paper's ``t`` is the
    noise end; flipping to this repo's ``t``-is-data convention negates both
    the derivative and the velocity, so the sign carries through). The live
    map value ``Psi`` is the JVP primal, so one dual forward serves both the
    tangent and the bilinear term.

    The teacher field ``v`` comes from the same form x source dispatcher as
    the anchor (:meth:`_teacher_field`), evaluated at the EMA map endpoint
    ``X_t = Y_t^ema / alpha_t^{z^2}`` — the repo's documented stand-in for
    the paper's stopped student endpoint ``f_{theta-}`` (see
    :class:`~src.models.components.losses.ei_doss.EIDossLSDLoss`).

    Uncertainty weighting is rejected: the bilinear objective is signed, and
    the Kendall log-variance reweighting requires non-negative per-sample
    losses.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Build the LMD anchor loss; ``use_uncertainty_weighting`` is rejected."""
        if kwargs.get('use_uncertainty_weighting'):
            raise ValueError(
                'DMFWZEILMDAnchorLoss does not support uncertainty weighting: the '
                'AYF-LMD bilinear objective is signed, and the Kendall log-variance '
                'reweighting requires non-negative per-sample losses.'
            )
        super().__init__(*args, **kwargs)

    def _imf_loss(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        s: Tensor,
        t: Tensor,
        cond_labels: Tensor,
        guided: Tensor,
        zeta: float,
        degree: int,
    ) -> Tensor:
        """Per-sample AYF-LMD bilinear objective on one interval set.

        The iMF compound hook is reused so the inherited p-split forward,
        per-slice label discipline, zeta annealing, and anchor dispatch apply
        unchanged; this override swaps the Eulerian compound for the
        Lagrangian bilinear tangent objective (class docstring).
        """
        x_s = self.schedule.sample_x_t(x_0, x_1, s)
        y_s = ei_y_state(self.schedule, x_s, s, zeta)
        martingale = self._sample_path(x_s, zeta, degree)
        driver_t = wz_driver(martingale, t, x_s).detach()
        lift_t = expand_t((1.0 + zeta * zeta) * self.schedule.alpha(t) ** (zeta * zeta), y_s)

        def _map(end: Tensor) -> Tensor:
            interval = martingale.interval(s, end)
            coefficients = interval.coefficient_features(normalize=net.normalize_coefficients)
            return wz_ei_y_map(
                net, self.schedule, y_s, s, end, coefficients, interval.terminal_increment(), zeta
            )

        # Single JVP: the primal doubles as the live map value for the
        # bilinear term (two overlapping dual forwards OOM at batch 64; see
        # the EI-Doss LSD arm's note).
        psi, d_psi_dt = jvp(_map, (t,), (torch.ones_like(t),))
        with torch.inference_mode():
            interval = martingale.interval(s, t)
            coefficients = interval.coefficient_features(normalize=ema_net.normalize_coefficients)
            y_ema = wz_ei_y_map(
                ema_net,
                self.schedule,
                y_s,
                s,
                t,
                coefficients,
                interval.terminal_increment(),
                zeta,
            )
            x_t = ei_x_state(self.schedule, y_ema, t, zeta)
            v_field = self._teacher_field(net, ema_net, x_0, x_1, x_t, t, cond_labels, guided)
        residual = d_psi_dt.detach() - (lift_t * v_field + driver_t)
        weight = expand_t(t - s, psi)
        return (weight * psi * residual).flatten(1).sum(1)
