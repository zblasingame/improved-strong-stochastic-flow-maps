"""Diagonal velocity matching and EI-Euler progressive self-distillation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.components.ei_euler import apply_ei_euler_prediction
from src.models.components.samplers import split_ei_euler_path
from src.models.components.schedule import AffineSchedule
from src.models.components.ssfm import SSFMNet


def _per_sample_mse(error: Tensor) -> Tensor:
    return error.square().flatten(1).mean(1)


def _field(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def _prediction(net: SSFMNet, x: Tensor, s: Tensor, t: Tensor, path: object) -> Tensor:
    features = path.coefficient_features(normalize=net.normalize_coefficients)
    net.set_brownian(path)
    try:
        return net.drift(x, s, t, features)
    finally:
        net.set_brownian(None)


def ei_euler_diagonal_loss(
    net: SSFMNet, schedule: AffineSchedule, x_1: Tensor, x_0: Tensor, t: Tensor
) -> Tensor:
    """Match the Brownian-free diagonal student to flow-matching velocity."""
    x_t = schedule.sample_x_t(x_0, x_1, t)
    diagonal = getattr(net, 'diagonal_prediction', None)
    if diagonal is None:
        raise TypeError('EI-Euler diagonal matching requires net.diagonal_prediction(x, t).')
    return _per_sample_mse(diagonal(x_t, t) - schedule.velocity(x_0, x_1, t))


def ei_euler_psd_loss(
    net: SSFMNet,
    ema_net: SSFMNet,
    schedule: AffineSchedule,
    x_1: Tensor,
    x_0: Tensor,
    s: Tensor,
    t: Tensor,
    degree: int,
    t0: float,
    degrees: Tensor | None = None,
    space: str = 'velocity',
) -> Tensor:
    """Match one student interval to two Chen-coupled EMA intervals.

    ``space='velocity'`` regresses heads after the increment identity cancels.
    ``space='map'`` regresses assembled maps, so both increments stay in the loss.
    """
    if space not in ('velocity', 'map'):
        raise ValueError(f"space must be 'velocity' or 'map'; got {space!r}.")
    x_s = schedule.sample_x_t(x_0, x_1, s)
    middle = 0.5 * (s + t)
    shape = tuple(x_s.shape)
    view = (-1, *([1] * (x_s.ndim - 1)))
    left, right, parent = split_ei_euler_path(
        schedule,
        shape,
        s.view(view),
        middle.view(view),
        t.view(view),
        degree,
        device=x_s.device,
        dtype=x_s.dtype,
        t0=t0,
        degrees=degrees,
    )
    with torch.no_grad():
        teacher_u_sm = _prediction(ema_net, x_s, s, middle, left)
        y_middle = apply_ei_euler_prediction(schedule, x_s, s, middle, teacher_u_sm, left)
        teacher_u_mt = _prediction(ema_net, y_middle, middle, t, right)
    student = _prediction(net, x_s, s, t, parent)
    if space == 'map':
        prediction = apply_ei_euler_prediction(schedule, x_s, s, t, student, parent)
        target = apply_ei_euler_prediction(schedule, y_middle, middle, t, teacher_u_mt, right)
        return _per_sample_mse(prediction - target.detach())
    left_weight, right_weight = schedule.ei_psd_target_weights(s, middle, t)
    target = _field(left_weight, x_s) * teacher_u_sm + _field(right_weight, x_s) * teacher_u_mt
    return _per_sample_mse(student - target.detach())


class EIEulerPSDLoss(nn.Module):
    """Mix diagonal velocity matching and EI-Euler PSD with a fixed batch split."""

    def __init__(
        self,
        schedule: AffineSchedule,
        p: float = 0.75,
        degree: int = 3,
        epsilon: float = 0.01,
        dt: float = 0.05,
        h_max: float = 1.0,
        degree_mode: str = 'fixed',
        degree_min: int = 1,
        degree_max: int | None = None,
        psd_space: str = 'velocity',
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1]; got {p}.')
        if degree_mode not in ('fixed', 'variable'):
            raise ValueError(f"degree_mode must be 'fixed' or 'variable'; got {degree_mode!r}.")
        if psd_space not in ('velocity', 'map'):
            raise ValueError(f"psd_space must be 'velocity' or 'map'; got {psd_space!r}.")
        if degree_mode == 'variable':
            if degree_max is None:
                raise ValueError("degree_mode='variable' requires an explicit degree_max.")
            if degree_min < 1 or degree_min > degree_max:
                raise ValueError(
                    f'degree_min must satisfy 1 <= degree_min <= degree_max; got {degree_min}, {degree_max}.'
                )
        self.schedule = schedule
        self.p = p
        self.degree = degree
        self.epsilon = epsilon
        self.dt = dt
        self.h_max = h_max
        self.t0 = epsilon
        self.degree_mode = degree_mode
        self.degree_min = degree_min
        self.degree_max = degree_max
        self.psd_space = psd_space

    def _path_degree(self) -> int:
        """Return the padded coefficient width used by sampled paths."""
        if self.degree_mode == 'variable':
            if self.degree_max is None:
                raise RuntimeError("degree_mode='variable' requires degree_max.")
            return self.degree_max
        return self.degree

    def _sample_degrees(self, batch: int, device: torch.device) -> Tensor | None:
        """Draw a per-sample active degree, or ``None`` for fixed-degree paths."""
        if self.degree_mode != 'variable':
            return None
        return torch.randint(self.degree_min, self._path_degree() + 1, (batch,), device=device)

    def _sample_times(self, batch: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        t = (
            self.epsilon
            + self.dt
            + (1.0 - self.epsilon - self.dt)
            * torch.rand(batch, device=reference.device, dtype=reference.dtype)
        )
        upper = (t - self.epsilon).clamp(min=self.dt, max=self.h_max)
        h = self.dt + (upper - self.dt) * torch.rand_like(t)
        return t - h, t

    def forward(
        self,
        net: SSFMNet,
        ema_net: SSFMNet,
        x_1: Tensor,
        x_0: Tensor,
        global_step: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return fixed-ratio diagonal/PSD loss and diagnostics."""
        del global_step
        batch = x_1.shape[0]
        n_psd = int(batch * self.p)
        s, t = self._sample_times(batch, x_1)
        degrees = self._sample_degrees(batch, x_1.device)
        psd = (
            ei_euler_psd_loss(
                net,
                ema_net,
                self.schedule,
                x_1[:n_psd],
                x_0[:n_psd],
                s[:n_psd],
                t[:n_psd],
                self._path_degree(),
                self.t0,
                degrees[:n_psd] if degrees is not None else None,
                self.psd_space,
            ).mean()
            if n_psd
            else x_1.new_zeros(())
        )
        diagonal = (
            ei_euler_diagonal_loss(net, self.schedule, x_1[n_psd:], x_0[n_psd:], t[n_psd:]).mean()
            if n_psd < batch
            else x_1.new_zeros(())
        )
        total = self.p * psd + (1.0 - self.p) * diagonal
        return total, {
            'diagonal_loss': diagonal.detach(),
            'psd_loss': psd.detach(),
            'effective_p': total.new_tensor(self.p),
            'n_psd': total.new_tensor(n_psd),
        }


