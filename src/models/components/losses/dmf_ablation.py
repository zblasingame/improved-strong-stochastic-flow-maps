"""Semigroup (PSD) objective for the SSFM -> iSSFM ablation ladder.

Ablation counterpart of :class:`~.dmf_ei_euler.DMFEIEulerAnchorPSDLoss` for
adapters with a learned diffusion head and/or no exponential lift (see
:mod:`src.models.components.ablation_map`). The batch split, time sampling,
label dropping, and uncertainty weighting are inherited unchanged; only the
two branches differ:

* **PSD branch.** The student's one full step regresses the EMA teacher's two
  Chen-coupled half-steps in state space (:func:`ablation_psd_loss`); learned
  noise and the unlifted drift cannot be supervised in velocity space. As in
  the original SSFM semigroup loss, the per-sample MSE is divided by ``h``.
* **Diagonal branch.** ``'anchor'`` matches the drift head to the frozen
  teacher on ``s = t`` (converted to the SDE drift ``2 u_t - x_t / t`` for
  unlifted nets) and, with learned noise, the diffusion head to ``nu_t``.
  ``'em_step'`` instead regresses one small step ``h ~ U[em_dt/2, em_dt]``
  on the Euler--Maruyama step of the teacher SDE (:func:`em_step_loss`), the
  original SSFM's diagonal term.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.components.ablation_map import (
    ablation_map,
    diagonal_drift_target,
    has_learned_noise,
    is_lifted,
    sample_closed_path,
)
from src.models.components.brownian import BrownianPolynomial
from src.models.components.ei_euler import sample_ei_euler_path
from src.models.components.samplers import split_ei_euler_path
from src.models.components.ssfm import SSFMNet

from .dmf_ei_euler import DMFEIEulerAnchorPSDLoss, dmf_direct_field
from .ei_euler import _per_sample_mse

DIAGONAL_TERMS = ('anchor', 'em_step')


def _split_paths(net: SSFMNet, schedule, shape, s, middle, t, degree, t0):
    """Sample Chen-coupled ``(left, right, parent)`` paths in the net's noise coordinates."""
    if has_learned_noise(net):
        left = BrownianPolynomial.sample(shape, s, middle, degree, device=s.device, dtype=s.dtype)
        right = BrownianPolynomial.sample(shape, middle, t, degree, device=s.device, dtype=s.dtype)
        return left, right, BrownianPolynomial.concat_arbitrary(left, right)
    if not is_lifted(net):
        left = sample_closed_path(schedule, shape, s, middle, degree, t0)
        right = sample_closed_path(schedule, shape, middle, t, degree, t0)
        return left, right, BrownianPolynomial.concat_arbitrary(left, right)
    return split_ei_euler_path(
        schedule, shape, s, middle, t, degree, device=s.device, dtype=s.dtype, t0=t0
    )


def ablation_psd_loss(
    net: SSFMNet,
    ema_net: SSFMNet,
    schedule,
    x_1: Tensor,
    x_0: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
    t0: float,
) -> Tensor:
    """Match one student step to two Chen-coupled EMA half-steps in state space.

    :return: Per-sample state-space MSE divided by ``h``, shape ``(B,)``.
    """
    x_s = schedule.sample_x_t(x_0, x_1, s)
    middle = 0.5 * (s + t)
    view = (-1, *([1] * (x_s.ndim - 1)))
    left, right, parent = _split_paths(
        net, schedule, tuple(x_s.shape), s.view(view), middle.view(view), t.view(view), degree, t0
    )
    with torch.no_grad():
        y_middle = ablation_map(ema_net, schedule, x_s, s, middle, left)
        target = ablation_map(ema_net, schedule, y_middle, middle, t, right)
    student = ablation_map(net, schedule, x_s, s, t, parent)
    return _per_sample_mse(student - target) / (t - s).clamp_min(1e-6)


def em_step_loss(
    net: SSFMNet,
    schedule,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    velocity: Tensor,
    degree: int,
    t0: float = 0.01,
) -> Tensor:
    """Match one small step to the Euler--Maruyama step of the teacher SDE.

    The original SSFM diagonal term: with ``h = t - s`` small, the student's
    map on ``[s, t]`` regresses ``x_s + h b_s + noise``, where
    ``b_s = 2 u_s - (alpha_dot_s / alpha_s) x_s`` is built from the given
    (stopped) velocity. With learned noise the noise is ``nu_s W_{s,t}`` on
    the student's real-time path; with closed-form noise it is the student's
    own exact ``int_s^t nu_r dW_r`` (unlifted martingale-clock path), so both
    sides share it and the term supervises the drift. Divided by ``h``.

    :param velocity: Stopped target velocity ``u_s`` at ``(x_s, s)``.
    :return: Per-sample loss, shape ``(B,)``.
    """
    view = (-1, *([1] * (x_s.ndim - 1)))
    h = (t - s).view(view)
    drift = 2.0 * velocity - (schedule.alpha_dot(s) / schedule.alpha(s)).view(view) * x_s
    if has_learned_noise(net):
        path = BrownianPolynomial.sample(
            tuple(x_s.shape), s.view(view), t.view(view), degree,
            device=x_s.device, dtype=x_s.dtype,
        )
        nu = schedule.nu_squared(s).clamp_min(0.0).sqrt().view(view)
        noise = nu * path.raw_increment()
    elif not is_lifted(net):
        path = sample_closed_path(schedule, tuple(x_s.shape), s.view(view), t.view(view), degree, t0)
        noise = path.raw_increment()
    else:
        path = sample_ei_euler_path(
            schedule, tuple(x_s.shape), s=s.view(view), t=t.view(view), degree=degree,
            device=x_s.device, dtype=x_s.dtype, t0=t0,
        )
        noise = path.terminal_increment()
    target = (x_s + h * drift + noise).detach()
    student = ablation_map(net, schedule, x_s, s, t, path)
    return _per_sample_mse(student - target) / h.flatten().clamp_min(1e-6)


class DMFAblationPSDLoss(DMFEIEulerAnchorPSDLoss):
    """DMF semigroup loss for learned-noise and/or unlifted ablation adapters.

    :param diagonal: ``'anchor'`` (teacher at ``s = t``) or ``'em_step'``
        (small Euler--Maruyama step on the teacher SDE).
    :param em_dt: Upper bound of the EM-step size, ``h ~ U[em_dt/2, em_dt]``
        (``0.01`` in the reference SSFM); independent of the semigroup ``dt``.
    """

    def __init__(
        self, *args, diagonal: str = 'anchor', em_dt: float = 0.01, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if diagonal not in DIAGONAL_TERMS:
            raise ValueError(f'diagonal must be one of {DIAGONAL_TERMS}; got {diagonal!r}.')
        if (self.guidance, self.guidance_source) != ('direct', 'frozen'):
            raise ValueError(
                "The ablation loss supports only guidance='direct' (or 'pretrained') with "
                "guidance_source='frozen'."
            )
        if self.degree_mode != 'fixed':
            raise ValueError("The ablation loss supports only degree_mode='fixed'.")
        if not 0.0 < em_dt <= self.dt:
            raise ValueError(f'em_dt must lie in (0, dt={self.dt}]; got {em_dt}.')
        self.diagonal = diagonal
        self.em_dt = em_dt

    def _psd_loss(self, net, ema_net, x_1, x_0, s, t, degree, degrees) -> Tensor:
        """Return the state-space semigroup term."""
        del degrees
        return ablation_psd_loss(net, ema_net, self.schedule, x_1, x_0, s, t, degree, self.t0)

    def _anchor_loss(self, net, ema_net, x_1, x_0, t, cond_labels, guided) -> Tensor:
        """Return the configured diagonal term (``s = t`` anchor or small EM step)."""
        del cond_labels, guided
        if self.diagonal == 'em_step':
            s = t - self.em_dt * (0.5 + 0.5 * torch.rand_like(t))
            x_s = self.schedule.sample_x_t(x_0, x_1, s)
            velocity = dmf_direct_field(net, ema_net, x_s, s, 'frozen')
            return em_step_loss(
                net, self.schedule, x_s, s, t, velocity, self.degree, self.t0
            )
        x_t = self.schedule.sample_x_t(x_0, x_1, t)
        velocity = dmf_direct_field(net, ema_net, x_t, t, 'frozen')
        drift, diffusion = net.diagonal_fields(x_t, t)
        loss = _per_sample_mse(drift - diagonal_drift_target(net, self.schedule, x_t, t, velocity))
        if diffusion is None:
            return loss
        nu = self.schedule.nu_squared(t).clamp_min(0.0).sqrt()
        return loss + _per_sample_mse(diffusion - nu.view(-1, *([1] * (x_t.ndim - 1))))
