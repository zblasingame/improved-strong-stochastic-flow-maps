"""Direct/MG DMF anchors plus the EI-Euler PSD objective."""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.components.ei_euler import apply_ei_euler_prediction
from src.models.components.fixed_coefficient_ssfm import FixedCoefficientDMFEIEulerSSFM
from src.models.components.samplers import split_ei_euler_path
from src.models.components.ssfm import SSFMNet

from .ei_euler import EIEulerPSDLoss, _per_sample_mse, _prediction, ei_euler_psd_loss
from .uncertainty import UncertaintyMLP, apply_uncertainty


def normalize_guidance(guidance: str, source: str | None) -> tuple[str, str]:
    """Resolve the guidance ``form x source`` taxonomy.

    The target field of every DMF anchor is described by two axes: the
    ``form`` (``'direct'`` one stopped forward of the configured source,
    ``'mg'`` the ``v_data + omega (v_cond - v_uncond)`` mix) and the
    ``source`` (``'frozen'`` the pretrained teacher with LoRA disabled,
    ``'ema'`` the EMA shadow's diagonal head). An omitted source keeps the
    per-form defaults (``mg`` -> ``'ema'``, every other form -> ``'frozen'``).

    :param guidance: The guidance form.
    :param source: The field source, or ``None`` for the per-form default.
    :return: The normalized ``(guidance, source)`` pair.
    """
    if guidance not in ('direct', 'mg'):
        raise ValueError(f"guidance must be 'direct' or 'mg'; got {guidance!r}.")
    if source is not None and source not in ('ema', 'frozen'):
        raise ValueError(f"guidance_source must be 'frozen', 'ema', or None; got {source!r}.")
    if source is None:
        source = 'ema' if guidance == 'mg' else 'frozen'
    return guidance, source


def dmf_direct_field(
    net: FixedCoefficientDMFEIEulerSSFM,
    ema_net: SSFMNet,
    x: Tensor,
    t: Tensor,
    source: str,
) -> Tensor:
    """Return the direct-form target field at ``(x, t)`` from ``source``.

    One stopped forward of the configured source at the student's currently
    assigned labels (null labels read the unconditional source): ``'frozen'``
    the pretrained teacher's own conditional forward (LoRA disabled; the
    upstream checkpoint's model guidance is baked in), ``'ema'`` the EMA
    shadow's diagonal head -- the same shadow the PSD semigroup target uses.
    No ``v_data`` / ``omega`` mixing and no guidance window apply. The EMA
    diagonal reads its labels from the transient module state the loss
    forward assigns, so ``_active_labels`` fails loudly when unset.

    :param net: The student adapter (supplies the frozen path and the live
        diagonal).
    :param ema_net: The EMA shadow of the student.
    :param x: States to evaluate at, shape ``(B, C, H, W)``.
    :param t: Diagonal times, broadcastable to ``(B,)``.
    :param source: ``'frozen'`` or ``'ema'``.
    :return: The stopped target field of shape ``(B, C, H, W)``.
    """
    if source not in ('frozen', 'ema'):
        raise ValueError(f"direct source must be 'frozen' or 'ema'; got {source!r}.")
    with torch.no_grad():
        if source == 'frozen':
            return net.anchor_prediction(x, t, net._active_labels(x))
        return ema_net.anchor_student_prediction(x, t)


def dmf_direct_anchor_loss(
    net: FixedCoefficientDMFEIEulerSSFM,
    ema_net: SSFMNet,
    schedule,
    x_1: Tensor,
    x_0: Tensor,
    t: Tensor,
    source: str,
) -> Tensor:
    """Distill the direct-form field on the diagonal from ``source``.

    The student's live diagonal regresses against one stopped forward of the
    configured source at the same state, numerically matching the direct
    anchor of :class:`~src.models.components.losses.dmf_wz_imf.DMFWZEIIMFAnchorLoss`.
    """
    x_t = schedule.sample_x_t(x_0, x_1, t)
    target = dmf_direct_field(net, ema_net, x_t, t, source)
    return _per_sample_mse(net.anchor_student_prediction(x_t, t) - target)


def dmf_guided_velocity(
    net: FixedCoefficientDMFEIEulerSSFM,
    ema_net: SSFMNet,
    schedule,
    x_0: Tensor,
    x_1: Tensor,
    x: Tensor,
    t: Tensor,
    cond_labels: Tensor,
    guided: Tensor,
    omega: float,
    g_min: float,
    g_max: float,
    source: str = 'ema',
) -> Tensor:
    """Return the DMF guidance-mixed velocity target ``v_tgt`` at state ``(x, t)``.

    ``v_tgt = v_t + omega * (v_cond - v_uncond)`` inside ``[g_min, g_max]`` for
    non-dropped samples, and the plain flow-matching velocity ``v_t`` otherwise.
    This is the same teacher field the :func:`dmf_model_guidance_anchor_loss`
    anchor distills against, evaluated at an arbitrary state ``x`` (an ESD start
    point ``X_s``) and time ``t`` rather than the anchor diagonal ``X_t``.

    Detached: it is a stopped teacher field, never trainable.
    """
    if source not in ('ema', 'frozen'):
        raise ValueError(f"source must be 'frozen' or 'ema'; got {source!r}.")
    null_labels = torch.full_like(cond_labels, net.num_classes)
    with torch.no_grad():
        combined = torch.cat((x, x))
        combined_t = torch.cat((t, t))
        combined_labels = torch.cat((cond_labels, null_labels))
        if source == 'frozen':
            prediction = net.anchor_prediction(combined, combined_t, combined_labels)
        else:
            previous_labels = getattr(ema_net, '_labels', None)
            ema_net.set_labels(combined_labels)
            try:
                prediction = ema_net.anchor_student_prediction(combined, combined_t)
            finally:
                ema_net.set_labels(previous_labels)
        v_cond, v_uncond = prediction.chunk(2)
        v_data = schedule.velocity(x_0, x_1, t)
        in_interval = (t >= g_min) & (t <= g_max) & guided
        mask = in_interval.reshape(-1, *([1] * (x.ndim - 1)))
        target = v_data + omega * (v_cond - v_uncond)
        return torch.where(mask, target, v_data)


def dmf_model_guidance_anchor_loss(
    net: FixedCoefficientDMFEIEulerSSFM,
    ema_net: SSFMNet,
    schedule,
    x_1: Tensor,
    x_0: Tensor,
    t: Tensor,
    cond_labels: Tensor,
    guided: Tensor,
    omega: float,
    g_min: float,
    g_max: float,
    source: str = 'ema',
) -> Tensor:
    """Distill ``v_t + omega * (v_cond - v_uncond)`` from a chosen guidance source.

    ``v_t`` is the true flow-matching velocity. ``v_cond`` / ``v_uncond`` come
    from ``source``: the EMA diagonal (``'ema'``) or the frozen pretrained
    teacher (``'frozen'``). The nudge applies only inside ``[g_min, g_max]``
    for non-dropped labels.
    """
    if source not in ('ema', 'frozen'):
        raise ValueError(f"source must be 'frozen' or 'ema'; got {source!r}.")
    x_t = schedule.sample_x_t(x_0, x_1, t)
    target = dmf_guided_velocity(
        net, ema_net, schedule, x_0, x_1, x_t, t, cond_labels, guided, omega, g_min, g_max, source
    )
    return _per_sample_mse(net.anchor_student_prediction(x_t, t) - target)


class DMFEIEulerAnchorPSDLoss(EIEulerPSDLoss):
    """Use 25% DMF anchor and 75% direct EMA EI-Euler PSD.

    The anchor term distills a stopped teacher field on the diagonal,
    described by the ``form x source`` taxonomy of
    :func:`normalize_guidance`: the ``'direct'`` form evaluates one forward
    of the configured source (``'frozen'`` pretrained teacher or ``'ema'``
    shadow diagonal), and ``'mg'`` distills the guidance-mixed velocity of
    :func:`dmf_model_guidance_anchor_loss`. The 75% PSD branch always
    distills the EMA shadow's interval head over two Chen-coupled half-steps.
    """

    def __init__(
        self,
        *args,
        p: float = 0.75,
        cfg_prob: float = 0.1,
        guidance: str = 'mg',
        guidance_source: str | None = None,
        omega: float = 0.6,
        g_min: float = 0.0,
        g_max: float = 1.0,
        fourier_dim: int = 256,
        uncertainty_form: str = 'log_eps',
        use_uncertainty_weighting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, p=p, **kwargs)
        guidance, guidance_source = normalize_guidance(guidance, guidance_source)
        if omega < 0.0:
            raise ValueError(f'omega must be >= 0; got {omega}.')
        if not 0.0 <= g_min <= g_max <= 1.0:
            raise ValueError(
                f'guidance interval must satisfy 0 <= g_min <= g_max <= 1; got [{g_min}, {g_max}].'
            )
        if uncertainty_form not in ('log_eps', 'reference'):
            raise ValueError(
                f"uncertainty_form must be 'log_eps' or 'reference'; got {uncertainty_form!r}."
            )
        self.cfg_prob = cfg_prob
        self.guidance = guidance
        self.guidance_source = guidance_source
        self.omega = omega
        self.g_min = g_min
        self.g_max = g_max
        self.uncertainty_form = uncertainty_form
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty = (
            UncertaintyMLP(fourier_dim=fourier_dim) if use_uncertainty_weighting else None
        )

    def _reweight(self, raw: Tensor, s: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(weighted, log_var)`` for one mixer branch."""
        if not self.use_uncertainty_weighting:
            return raw, raw.new_zeros(raw.shape)
        if self.uncertainty is None:
            raise RuntimeError('Uncertainty weighting is enabled without an uncertainty module.')
        log_var = self.uncertainty(s, t)
        return apply_uncertainty(raw, log_var, form=self.uncertainty_form), log_var

    def _psd_loss(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        s: Tensor,
        t: Tensor,
        degree: int,
        degrees: Tensor | None,
    ) -> Tensor:
        """Return the per-sample semigroup term (closed-form-noise EI-Euler PSD)."""
        return ei_euler_psd_loss(
            net,
            ema_net,
            self.schedule,
            x_1,
            x_0,
            s,
            t,
            degree,
            self.t0,
            degrees,
            self.psd_space,
        )

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
        """Dispatch the anchor term to the configured guidance form.

        ``'direct'`` regresses the live student diagonal against one stopped
        forward of the configured source (frozen teacher / EMA shadow) at the
        currently assigned student labels (null included), matching the WZ-iMF
        direct anchor.
        """
        if self.guidance == 'direct':
            return dmf_direct_anchor_loss(
                net, ema_net, self.schedule, x_1, x_0, t, self.guidance_source
            )
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

    def forward(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return fixed-ratio anchor/PSD loss and diagnostics."""
        del global_step
        if not isinstance(net, FixedCoefficientDMFEIEulerSSFM):
            raise TypeError('DMFEIEulerAnchorPSDLoss requires a FixedCoefficientDMFEIEulerSSFM.')
        batch = x_1.shape[0]
        n_psd = int(batch * self.p)
        s, t = self._sample_times(batch, x_1)
        labels = net._active_labels(x_1)
        drop_ids = torch.rand(batch, device=x_1.device) < self.cfg_prob
        student_labels = torch.where(drop_ids, net.num_classes, labels)
        try:
            net.set_labels(student_labels[:n_psd])
            ema_net.set_labels(student_labels[:n_psd])
            path_degree = self._path_degree()
            degrees = self._sample_degrees(n_psd, x_1.device)
            raw_psd = self._psd_loss(
                net,
                ema_net,
                x_1[:n_psd],
                x_0[:n_psd],
                s[:n_psd],
                t[:n_psd],
                path_degree,
                degrees,
            )
            net.set_labels(student_labels[n_psd:])
            # The direct-form EMA anchor reads the shadow's ambient label
            # state, so it must mirror the student's slice assignment.
            ema_net.set_labels(student_labels[n_psd:])
            raw_anchor = self._anchor_loss(
                net,
                ema_net,
                x_1[n_psd:],
                x_0[n_psd:],
                t[n_psd:],
                labels[n_psd:],
                ~drop_ids[n_psd:],
            )
        finally:
            net.set_labels(labels)
            ema_net.set_labels(labels)
        weighted_psd, u_psd = self._reweight(raw_psd, s[:n_psd], t[:n_psd])
        weighted_anchor, u_anchor = self._reweight(raw_anchor, t[n_psd:], t[n_psd:])
        psd = weighted_psd.mean()
        anchor = weighted_anchor.mean()
        total = self.p * psd + (1.0 - self.p) * anchor
        diagnostics = {
            'anchor_loss': raw_anchor.mean().detach(),
            'psd_loss': raw_psd.mean().detach(),
            'effective_p': total.new_tensor(self.p),
            'n_psd': total.new_tensor(n_psd),
            'effective_cfg_prob': drop_ids.float().mean(),
            'guidance_omega': total.new_tensor(self.omega),
        }
        if self.use_uncertainty_weighting:
            diagnostics['uncertainty_mlp'] = torch.cat((u_psd, u_anchor)).mean().detach()
        return total, diagnostics


class DMFEIEulerAnchorDiscreteESDLoss(DMFEIEulerAnchorPSDLoss):
    """Mix the pretrained DMF anchor with discrete Eulerian self-distillation."""

    _discrete_kind = 'esd'

    def _discrete_middle(self, s: Tensor, t: Tensor) -> Tensor:
        """Sample a short split: leading for ESD, trailing for LSD."""
        delta = torch.minimum(
            0.5 * self.dt * (1.0 + torch.rand_like(s)),
            0.5 * (t - s),
        )
        if self._discrete_kind == 'lsd':
            return t - delta
        return s + delta

    def _detached_discrete_target(
        self,
        net: SSFMNet,
        x_s: Tensor,
        s: Tensor,
        middle: Tensor,
        t: Tensor,
        left: object,
        right: object,
    ) -> Tensor:
        """Build the stopped two-step teacher map used by discrete ESD/LSD."""
        diagonal = getattr(net, 'diagonal_prediction', None)
        if diagonal is None:
            raise TypeError('DMF discrete self-distillation requires diagonal_prediction(x, t).')
        if self._discrete_kind == 'lsd':
            leading = _prediction(net, x_s, s, middle, left)
            x_middle = apply_ei_euler_prediction(self.schedule, x_s, s, middle, leading, left)
            trailing = diagonal(x_middle, middle)
            return apply_ei_euler_prediction(
                self.schedule, x_middle, middle, t, trailing, right
            ).clone()
        left_velocity = diagonal(x_s, s)
        x_middle = apply_ei_euler_prediction(self.schedule, x_s, s, middle, left_velocity, left)
        target_velocity = _prediction(net, x_middle, middle, t, right)
        return apply_ei_euler_prediction(
            self.schedule, x_middle, middle, t, target_velocity, right
        ).clone()

    def forward(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the discrete ESD/anchor mixture and diagnostics."""
        del global_step
        if not isinstance(net, FixedCoefficientDMFEIEulerSSFM):
            raise TypeError('DMF discrete self-distillation requires a DMF EI-Euler adapter.')
        batch = x_1.shape[0]
        n_esd = int(batch * self.p)
        s, t = self._sample_times(batch, x_1)
        labels = net._active_labels(x_1)
        drop_ids = torch.rand(batch, device=x_1.device) < self.cfg_prob
        student_labels = torch.where(drop_ids, net.num_classes, labels)
        try:
            net.set_labels(student_labels[:n_esd])
            x_s = self.schedule.sample_x_t(x_0[:n_esd], x_1[:n_esd], s[:n_esd])
            middle = self._discrete_middle(s[:n_esd], t[:n_esd])
            path_degree = self._path_degree()
            degrees = self._sample_degrees(n_esd, x_1.device)
            left, right, parent = split_ei_euler_path(
                self.schedule,
                tuple(x_s.shape),
                s[:n_esd].view(-1, *([1] * (x_s.ndim - 1))),
                middle.view(-1, *([1] * (x_s.ndim - 1))),
                t[:n_esd].view(-1, *([1] * (x_s.ndim - 1))),
                path_degree,
                device=x_s.device,
                dtype=x_s.dtype,
                t0=self.t0,
                degrees=degrees,
            )
            with torch.no_grad():
                target = self._detached_discrete_target(
                    net, x_s, s[:n_esd], middle, t[:n_esd], left, right
                )
            student = _prediction(net, x_s, s[:n_esd], t[:n_esd], parent)
            prediction = apply_ei_euler_prediction(
                self.schedule, x_s, s[:n_esd], t[:n_esd], student, parent
            )
            h = (t[:n_esd] - s[:n_esd]).clamp_min(torch.finfo(s.dtype).eps)
            raw_esd = _per_sample_mse(prediction - target.detach()) / (h**2)
            net.set_labels(student_labels[n_esd:])
            # Mirror the student's slice assignment for the direct-form EMA
            # anchor (the discrete branch itself never reads ema).
            ema_net.set_labels(student_labels[n_esd:])
            raw_anchor = self._anchor_loss(
                net,
                ema_net,
                x_1[n_esd:],
                x_0[n_esd:],
                t[n_esd:],
                labels[n_esd:],
                ~drop_ids[n_esd:],
            )
        finally:
            net.set_labels(labels)
            ema_net.set_labels(labels)
        weighted_esd, u_esd = self._reweight(raw_esd, s[:n_esd], t[:n_esd])
        weighted_anchor, u_anchor = self._reweight(raw_anchor, t[n_esd:], t[n_esd:])
        esd_loss = weighted_esd.mean()
        anchor_loss = weighted_anchor.mean()
        total = self.p * esd_loss + (1.0 - self.p) * anchor_loss
        diagnostics = {
            'anchor_loss': raw_anchor.mean().detach(),
            'psd_loss': raw_esd.mean().detach(),
            'effective_p': total.new_tensor(self.p),
            'n_psd': total.new_tensor(n_esd),
            'effective_cfg_prob': drop_ids.float().mean().detach(),
            'guidance_omega': total.new_tensor(self.omega),
        }
        if self.use_uncertainty_weighting:
            diagnostics['uncertainty_mlp'] = torch.cat((u_esd, u_anchor)).mean().detach()
        return total, diagnostics


class DMFEIEulerAnchorDiscreteLSDLoss(DMFEIEulerAnchorDiscreteESDLoss):
    """Mix the pretrained DMF anchor with discrete Lagrangian self-distillation."""

    _discrete_kind = 'lsd'
