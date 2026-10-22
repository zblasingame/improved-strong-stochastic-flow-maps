"""Augmented FKC network and blockwise EI flow map for 2D toy experiments.

Feynman-Kac-corrector experiments (arXiv:2503.02819) evolve an augmented state
``(X, w)`` in ``R^2 oplus R``: ``X`` follows a Wong-Zakai / exponential-integrator
(EI) spatial dynamics and ``w`` accumulates log importance weights along the
same Brownian path. The project contract for these toy experiments is:

- the backbone :class:`AugmentedFKCMLP` reads the physical state ``x in R^2``,
  the interval endpoints ``s <= t``, and the coefficient features of ONE 2D
  Brownian path ``(B, degree, 2)``. A single trunk feeds a spatial velocity
  head ``u_theta in R^2`` and a scalar log-weight rate head ``gbar_theta``;
- the backbone never reads the incoming log-weight ``w``: weight dependence of
  the learned map is purely additive, so conditioning on ``w`` would only add
  variance. There is NO independent Brownian noise for ``w`` and NO
  exponential lift of ``w``;
- the blockwise map :func:`augmented_fkc_map` supports the linear schedule
  ``alpha_t = t`` only. The sampler owns the EI martingale increment
  ``DeltaM = int_s^t alpha_r nu_r dW_r`` and passes it in explicitly; the map
  itself samples nothing.

All differentiable computations are plain tensor ops on the live inputs (no
detached conversions of times, states, or path features), so ``torch.func.jvp``
and autograd propagate tangents through both heads exactly; detached reads
appear only in input validation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.components.schedule import AffineSchedule, GeometricSchedule, LinearSchedule

_STATE_DIM = 2


def _sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal features of width ``dim`` for times ``t`` of shape ``(B,)``.

    Dtype-preserving variant of the DiT timestep features: the frequency table
    is built in the caller's dtype instead of forcing ``float32``, so float64
    math tests keep full precision and forward-mode tangents w.r.t. times are
    never cast away.

    :param t: Times of shape ``(B,)``.
    :param dim: Feature width (must be at least 2).
    :param max_period: Longest wavelength of the table.
    :return: Features of shape ``(B, dim)``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    args = t[:, None] * freqs[None, :]
    embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class _TimeEmbed(nn.Module):
    """Two-layer MLP embedding of sinusoidal time features.

    Lightweight, torch-only analogue of :class:`~src.models.components.layers.
    TimestepEmbedder`, kept local so the FKC package does not import the
    flash-attention/einops-heavy DiT layer stack.
    """

    def __init__(self, hidden_size: int, emb_size: int) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed times of shape ``(B,)`` into ``(B, hidden_size)``."""
        features = _sinusoidal_embedding(t.reshape(-1), self.emb_size)
        return self.mlp(features)


def _trunk(in_dim: int, hidden_size: int, depth: int) -> nn.Sequential:
    """Build a SiLU MLP trunk ending in a hidden activation (mirrors ``mlp._trunk``)."""
    if depth < 1:
        raise ValueError(f'depth must be at least 1; got {depth}.')
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_size), nn.SiLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
    return nn.Sequential(*layers)


def _validate_state(x: Tensor) -> None:
    """Reject states that are not ``(..., 2)``-shaped ``(B, 2)`` batches."""
    if x.ndim != 2 or x.shape[-1] != _STATE_DIM:
        raise ValueError(f'expected x of shape (B, {_STATE_DIM}); got {tuple(x.shape)}.')


def _expand_time(value: Tensor, batch: int, name: str) -> Tensor:
    """Require a scalar or ``(B,)`` time tensor and broadcast scalars to ``(B,)``."""
    if value.ndim == 0:
        return value.expand(batch)
    if value.ndim != 1 or value.shape[0] != batch:
        raise ValueError(
            f'{name} must be a scalar or of shape ({batch},); got {tuple(value.shape)}.'
        )
    return value


def _broadcast_times(
    s: Tensor | float, t: Tensor | float, batch: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Coerce both endpoints to ``(B,)`` tensors on the state's device/dtype."""
    s_t = torch.as_tensor(s, device=device, dtype=dtype)
    t_t = torch.as_tensor(t, device=device, dtype=dtype)
    return _expand_time(s_t, batch, 's'), _expand_time(t_t, batch, 't')


def _validate_interval(s: Tensor, t: Tensor) -> None:
    """Require ``0 < s <= t <= 1`` elementwise.

    Value checks read detached copies only: no tangent needs to cross
    validation, and every map coefficient is computed from the live tensors
    afterwards, so ``torch.func.jvp`` tangents stay exact.
    """
    s_v, t_v = s.detach(), t.detach()
    if bool(torch.any(s_v < 0.0)):
        raise ValueError('start times must satisfy 0 <= s.')
    if bool(torch.any(t_v < s_v)):
        raise ValueError('interval times must satisfy s <= t.')
    if bool(torch.any(t_v > 1.0)):
        raise ValueError('terminal times must satisfy t <= 1 under the linear schedule.')


def _validate_coefficients(
    coefficients: Tensor, batch: int, max_degree: int | None = None
) -> None:
    """Require coefficient features ``(B, degree, 2)`` with ``degree <= max_degree``."""
    if coefficients.ndim != 3 or coefficients.shape[-1] != _STATE_DIM:
        raise ValueError(
            'coefficients must have shape (B, degree, '
            f'{_STATE_DIM}) matching x; got {tuple(coefficients.shape)}.'
        )
    if coefficients.shape[0] != batch:
        raise ValueError(
            'coefficients must share the batch of x; '
            f'got {tuple(coefficients.shape)} vs batch {batch}.'
        )
    if max_degree is not None and coefficients.shape[1] > max_degree:
        raise ValueError(
            f'coefficient degree {coefficients.shape[1]} exceeds max_degree {max_degree}.'
        )


class AugmentedFKCMLP(nn.Module):
    """Shared-trunk augmented FKC backbone: 2D velocity plus scalar weight rate.

    Conditions on the physical state ``x`` of shape ``(B, 2)``, the interval
    endpoints ``(s, t)`` with ``0 < s <= t <= 1``, and the coefficients of one
    2D Brownian path ``(B, degree, 2)`` (zero-padded up to :attr:`max_degree`,
    mirroring :class:`~src.models.components.mlp.SSFM_MLP`). One trunk feeds

    - ``velocity`` of shape ``(B, 2)``: the spatial EI velocity ``u_theta``;
    - ``rate`` of shape ``(B,)``: the scalar log-weight rate ``gbar_theta``.

    The incoming log-weight ``w`` is deliberately **not** an input: the
    augmented map's weight dependence is purely additive (see
    :func:`augmented_fkc_map`), so conditioning on ``w`` is unnecessary. As in
    :class:`~src.models.components.mlp.SSFM_MLP`, ``normalize_coefficients`` is
    stored for sampler/loss convenience: when ``True``, callers feed
    unit-variance coefficient features. Both heads are zero-initialized so the
    fresh network starts at the exact identity skeleton.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        depth: int = 3,
        max_degree: int = 3,
        *,
        time_emb_size: int = 128,
        normalize_coefficients: bool = True,
        spatial_input_scale: float = 1.0,
        output_parameterization: str = 'velocity',
        score_scale_floor: float = 0.0,
    ) -> None:
        """Initialize an ``AugmentedFKCMLP``.

        :param hidden_size: Width of every hidden layer and of the time embeddings.
        :param depth: Number of hidden SiLU layers in the shared trunk.
        :param max_degree: Maximum Brownian coefficient degree the net accepts.
        :param time_emb_size: Fourier width of the sinusoidal time features.
        :param normalize_coefficients: Whether callers feed unit-variance
            coefficient features (stored for sampler/loss convenience).
        :param spatial_input_scale: Multiply only the trunk's spatial input by
            this factor. Outputs, time/path features, and physical map units
            are unchanged. The default preserves existing checkpoints.
        :param output_parameterization: ``'velocity'`` (default) predicts the
            composed probability-flow velocity; ``'score'`` predicts the scaled
            denoising noise ``eps = -scale * score`` of the geometric VE path.
        :param score_scale_floor: Positive floor for ``scale=sqrt(sigma**2+floor**2)``.
            Required for ``'score'`` so the data endpoint remains well posed.
        """
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f'hidden_size must be at least 1; got {hidden_size}.')
        if depth < 1:
            raise ValueError(f'depth must be at least 1; got {depth}.')
        if max_degree < 1:
            raise ValueError(f'max_degree must be at least 1; got {max_degree}.')
        if time_emb_size < 2:
            raise ValueError(f'time_emb_size must be at least 2; got {time_emb_size}.')
        if not math.isfinite(spatial_input_scale) or spatial_input_scale <= 0.0:
            raise ValueError('spatial_input_scale must be finite and positive.')
        if output_parameterization not in ('velocity', 'score'):
            raise ValueError("output_parameterization must be 'velocity' or 'score'.")
        if not math.isfinite(score_scale_floor) or score_scale_floor < 0.0:
            raise ValueError('score_scale_floor must be finite and nonnegative.')
        if output_parameterization == 'score' and score_scale_floor <= 0.0:
            raise ValueError('score parameterization requires positive score_scale_floor.')
        self.hidden_size = hidden_size
        self.depth = depth
        self.max_degree = max_degree
        self.normalize_coefficients = normalize_coefficients
        self.spatial_input_scale = float(spatial_input_scale)
        self.output_parameterization = output_parameterization
        self.score_scale_floor = float(score_scale_floor)
        self.time_embed = _TimeEmbed(hidden_size, time_emb_size)
        in_dim = _STATE_DIM + 2 * hidden_size + max_degree * _STATE_DIM
        self.trunk = _trunk(in_dim, hidden_size, depth)
        self.velocity_head = nn.Linear(hidden_size, _STATE_DIM)
        self.rate_head = nn.Linear(hidden_size, 1)
        # Zero-init heads (SSFM/adaLN-Zero convention): a fresh network realizes
        # the exact identity skeleton, and diagonal anchors start calm.
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.zeros_(self.rate_head.bias)

    def forward(
        self, x: Tensor, s: Tensor | float, t: Tensor | float, coefficients: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict the augmented field over the interval ``[s, t]``.

        :param x: Spatial states of shape ``(B, 2)``.
        :param s: Start times, scalar or ``(B,)``.
        :param t: Terminal times, scalar or ``(B,)``.
        :param coefficients: 2D Brownian coefficient features ``(B, degree, 2)``.
        :return: ``(velocity, rate)`` with shapes ``(B, 2)`` and ``(B,)``.
        """
        _validate_state(x)
        s_b, t_b = _broadcast_times(s, t, x.shape[0], x.device, x.dtype)
        _validate_interval(s_b, t_b)
        _validate_coefficients(coefficients, x.shape[0], self.max_degree)
        hidden = self.trunk(self._condition(x, s_b, t_b, coefficients))
        velocity = self.velocity_head(hidden)
        rate = self.rate_head(hidden).squeeze(-1)
        return velocity, rate

    def _condition(self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor) -> Tensor:
        """Assemble the trunk input: state, both time embeddings, path features."""
        flat = coefficients.reshape(x.shape[0], -1)
        pad = self.max_degree * _STATE_DIM - flat.shape[1]
        if pad:
            flat = F.pad(flat, (0, pad))
        return torch.cat(
            (self.spatial_input_scale * x, self.time_embed(s), self.time_embed(t), flat), dim=-1
        )


def uses_score_parameterization(net: nn.Module) -> bool:
    """Return whether ``net`` emits the geometric-VE scaled score/noise target."""
    return getattr(net, 'output_parameterization', 'velocity') == 'score'


def score_noise_scale(net: nn.Module, schedule: AffineSchedule, t: Tensor) -> Tensor:
    """Return the positive scale converting predicted noise to score."""
    if not isinstance(schedule, GeometricSchedule):
        raise TypeError('score parameterization is implemented only for GeometricSchedule.')
    floor = getattr(net, 'score_scale_floor', 0.0)
    return (schedule.sigma(t).square() + floor**2).sqrt()


def score_from_prediction(
    net: nn.Module, prediction: Tensor, schedule: AffineSchedule, t: Tensor
) -> Tensor:
    """Convert predicted noise ``eps`` to score ``-eps / scale``."""
    return -prediction / score_noise_scale(net, schedule, t)[:, None]


def noise_from_score(net: nn.Module, score: Tensor, schedule: AffineSchedule, t: Tensor) -> Tensor:
    """Convert a score to the scaled denoising-noise training target."""
    return -score_noise_scale(net, schedule, t)[:, None] * score


def potential_scale(net: nn.Module, schedule: AffineSchedule, t: Tensor) -> Tensor:
    """Natural size of the FKC potential ``g`` at time ``t``: ``nu^2 / scale^2``.

    With the score head the rate head predicts ``g / potential_scale``, an O(1)
    quantity at every time (``g ~ nu^2 ||s||^2`` and ``||s|| ~ 1 / scale``).
    """
    return schedule.nu_squared(t) / score_noise_scale(net, schedule, t).square()


def potential_span(net: nn.Module, schedule: AffineSchedule, s: Tensor, t: Tensor) -> Tensor:
    """Closed form ``int_s^t potential_scale dr = log((sig_s^2+m^2) / (sig_t^2+m^2))``.

    Uses ``nu^2 = -d sigma^2 / dr`` on the geometric VE path; the spans compose
    additively, so the semigroup weight identity stays exact.
    """
    floor2 = getattr(net, 'score_scale_floor', 0.0) ** 2
    return torch.log((schedule.sigma(s).square() + floor2) / (schedule.sigma(t).square() + floor2))


def score_to_velocity(score: Tensor, schedule: AffineSchedule, t: Tensor) -> Tensor:
    """Convert a VE score to the equivalent probability-flow velocity."""
    return 0.5 * schedule.nu_squared(t)[:, None] * score


def velocity_from_prediction(
    net: nn.Module, prediction: Tensor, schedule: AffineSchedule, s: Tensor, t: Tensor
) -> Tensor:
    """Return physical velocity for either output parameterization."""
    if not uses_score_parameterization(net):
        return prediction
    if not isinstance(schedule, GeometricSchedule):
        raise TypeError('score parameterization is implemented only for GeometricSchedule.')
    score = score_from_prediction(net, prediction, schedule, s)
    # Non-diagonal VE map: x_t - x_s = int_s^t nu^2 score = (sigma_s^2 - sigma_t^2) score_bar.
    duration = (t - s).clamp_min(torch.finfo(prediction.dtype).tiny)
    clock = schedule.martingale_clock(t) - schedule.martingale_clock(s)
    diagonal_velocity = score_to_velocity(score, schedule, s)
    interval_velocity = (0.5 * clock / duration)[:, None] * score
    return torch.where((t > s)[:, None], interval_velocity, diagonal_velocity)


def augmented_fkc_map(
    net: nn.Module,
    x: Tensor,
    w: Tensor,
    s: Tensor | float,
    t: Tensor | float,
    coefficients: Tensor,
    delta_m: Tensor | float,
    weight_span: Tensor | float,
    schedule: AffineSchedule | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply one blockwise augmented EI flow-map step (linear schedule only).

    With ``(u_theta, psi_theta) = net(x, s, t, coefficients)`` the step is

        ``x_t = (s * x + (t**2 - s**2) * u_theta + delta_m) / t``,
        ``w_t = w + weight_span * psi_theta``,

    where ``weight_span`` is schedule-owned (:meth:`AffineSchedule.weight_span`):
    the analytic clock ``int nu_r**2 dr`` when the head predicts the clock
    integrand ``psi = g / nu_t**2``, or the interval length ``t - s`` when it
    predicts the physical-time potential ``g`` directly. The geometric VE
    schedule uses the latter as a regression parameterization; its diffusion
    remains positive at the data endpoint. ``delta_m``
    is the EI martingale increment; the map samples nothing itself. ``w`` enters
    only additively, so shifting ``w`` shifts ``w_t`` identically. On the
    diagonal ``s = t`` with ``delta_m = 0`` and ``weight_span = 0`` the map is the
    identity on both blocks.

    All coefficients are computed from the live ``s``/``t`` tensors (no
    detached conversions in the differentiable path), so ``torch.func.jvp``
    tangents w.r.t. times, states, and path features propagate exactly.

    :param net: Backbone with ``forward(x, s, t, coefficients) -> (velocity, rate)``,
        e.g. :class:`AugmentedFKCMLP`; ``rate`` is ``psi_theta``.
    :param x: Spatial states ``(B, 2)`` at time ``s``.
    :param w: Log weights ``(B,)`` at time ``s``.
    :param s: Start times, scalar or ``(B,)``.
    :param t: Terminal times, scalar or ``(B,)``.
    :param coefficients: 2D Brownian coefficient features ``(B, degree, 2)``.
    :param delta_m: EI martingale increment ``DeltaM``, ``(B, 2)`` or scalar.
    :param weight_span: Span multiplying the head's rate (schedule-owned via
        :meth:`AffineSchedule.weight_span`), scalar or ``(B,)``.
    :return: ``(x_t, w_t)`` with shapes ``(B, 2)`` and ``(B,)``.
    """
    _validate_state(x)
    batch = x.shape[0]
    if w.ndim != 1 or w.shape[0] != batch:
        raise ValueError(f'w must be of shape ({batch},); got {tuple(w.shape)}.')
    s_b, t_b = _broadcast_times(s, t, batch, x.device, x.dtype)
    _validate_interval(s_b, t_b)
    _validate_coefficients(coefficients, batch)
    delta = _expand_delta_m(delta_m, batch, x)
    clock = _expand_time(
        torch.as_tensor(weight_span, device=x.device, dtype=x.dtype), batch, 'weight_span'
    )
    schedule = schedule or LinearSchedule()
    velocity, rate = net(x, s_b, t_b, coefficients)
    velocity = velocity_from_prediction(net, velocity, schedule, s_b, t_b)
    if uses_score_parameterization(net):
        # The rate head predicts g / potential_scale; integrate it on the
        # matching closed-form clock (the caller's weight_span is replaced).
        clock = potential_span(net, schedule, s_b, t_b)
    alpha_s, alpha_t = schedule.alpha(s_b), schedule.alpha(t_b)
    alpha_integral = schedule._ei_alpha_integral(s_b, t_b)
    x_t = (
        (alpha_s / alpha_t)[:, None] * x
        + (2.0 * alpha_integral / alpha_t)[:, None] * velocity
        + delta / alpha_t[:, None]
    )
    w_t = w + clock * rate
    return x_t, w_t


def _expand_delta_m(delta_m: Tensor | float, batch: int, x: Tensor) -> Tensor:
    """Accept a scalar or ``(B, 2)`` martingale increment, broadcast to ``(B, 2)``."""
    if isinstance(delta_m, Tensor):
        delta = delta_m
    else:
        delta = torch.as_tensor(delta_m, device=x.device, dtype=x.dtype)
    if delta.ndim == 0:
        return delta.expand(batch, _STATE_DIM)
    if delta.shape != (batch, _STATE_DIM):
        raise ValueError(
            f'delta_m must be a scalar or of shape ({batch}, {_STATE_DIM}); '
            f'got {tuple(delta.shape)}.'
        )
    return delta
