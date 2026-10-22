"""Flow-map steps for the SSFM -> iSSFM ablation ladder.

Two independent switches on the DMF adapter select the step on ``[s, t]``:

* **lift** (``net.lift``): with the exponential lift the drift head is the
  interval velocity ``u_st`` and the deterministic part is
  ``(alpha_s / alpha_t) x_s + c_st u_st``; without it the drift head is the raw
  SDE drift ``b_st`` and the deterministic part is the plain Euler step
  ``x_s + (t - s) b_st``.
* **noise** (``net.parameterization``): a learned diffusion head
  (``'euler_maruyama'``) adds ``g_st (.) W_{s,t}`` on a real-time polynomial
  path, as in the original SSFM; closed-form noise (``'euler'``) adds the
  exact stochastic integral on a polynomial path built on the martingale
  clock -- ``int_s^t nu_r dW_r`` unlifted (:func:`sample_closed_path`), or
  ``(1 / alpha_t) int_s^t alpha_r nu_r dW_r`` lifted, which is the existing
  EI-Euler map (:mod:`src.models.components.ei_euler`).

Every path type exposes ``coefficient_features`` (network conditioning) and a
terminal noise increment (:func:`noise_increment`), so one map serves all four
combinations, and the semigroup loss and samplers stay rung-agnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .brownian import BrownianPolynomial
from .ei_euler import EIEulerPath, apply_ei_euler_prediction
from .schedule import AffineSchedule
from .ssfm import SSFMNet


def _field(value: Tensor, reference: Tensor) -> Tensor:
    """Reshape per-sample times ``(B,)`` to broadcast against ``reference``."""
    return value.reshape(-1, *([1] * (reference.ndim - 1)))


def is_lifted(net: SSFMNet) -> bool:
    """Return whether the net uses the exponential lift (the default)."""
    return getattr(net, 'lift', True)


def has_learned_noise(net: SSFMNet) -> bool:
    """Return whether the net owns a learned diffusion head."""
    return net.parameterization == 'euler_maruyama'



def is_ablation_net(net: SSFMNet) -> bool:
    """Return whether the net needs an ablation map (learned noise or no lift)."""
    return has_learned_noise(net) or not is_lifted(net)


def diagonal_drift_target(
    net: SSFMNet, schedule: AffineSchedule, x_t: Tensor, t: Tensor, velocity: Tensor
) -> Tensor:
    """Convert a target velocity ``u_t`` into the drift head's diagonal target.

    Lifted heads predict ``u_t`` itself; unlifted heads predict the SDE drift
    ``b_t = 2 u_t - (alpha_dot_t / alpha_t) x_t``.
    """
    if is_lifted(net):
        return velocity
    ratio = schedule.alpha_dot(t) / schedule.alpha(t)
    return 2.0 * velocity - _field(ratio, x_t) * x_t


def sample_closed_path(
    schedule: AffineSchedule,
    shape: tuple[int, ...],
    s: Tensor,
    t: Tensor,
    degree: int,
    t0: float,
    *,
    generator: torch.Generator | None = None,
) -> BrownianPolynomial:
    """Sample an unlifted martingale path on the clock ``[<M>_s, <M>_t]``.

    ``<M>`` is the quadratic variation of ``M = int nu dW``, so the path's
    terminal increment is exactly ``int_s^t nu_r dW_r`` and adjacent paths
    compose through :meth:`BrownianPolynomial.concat_arbitrary`.

    :param s: Start times pre-shaped to broadcast against ``shape``.
    :param t: Terminal times pre-shaped to broadcast against ``shape``.
    """
    return BrownianPolynomial.sample(
        shape,
        schedule.martingale_clock(s, t0),
        schedule.martingale_clock(t, t0),
        degree,
        device=s.device,
        dtype=s.dtype,
        generator=generator,
    )


def noise_increment(path: BrownianPolynomial | EIEulerPath) -> Tensor:
    """Return the path's terminal noise increment (``W``, ``int nu dW``, or EI)."""
    if isinstance(path, EIEulerPath):
        return path.terminal_increment()
    return path.raw_increment()


def ablation_map(
    net: SSFMNet,
    schedule: AffineSchedule,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
    path: BrownianPolynomial | EIEulerPath,
) -> Tensor:
    """Evaluate the rung's flow-map step on ``[s, t]`` (see module docstring).

    :param path: Real-time path (learned noise), unlifted martingale-clock
        path (closed form, no lift), or :class:`EIEulerPath` (closed form,
        lifted).
    :return: States at ``t``, shape ``(B, C, H, W)``.
    """
    velocity, diffusion = net.fields(
        x_s, s, t, path.coefficient_features(normalize=net.normalize_coefficients)
    )
    if is_lifted(net) and not has_learned_noise(net):
        return apply_ei_euler_prediction(schedule, x_s, s, t, velocity, path)
    if is_lifted(net):
        state_coeff, velocity_coeff = schedule.ei_euler_coefficients(s, t)
    else:
        state_coeff, velocity_coeff = torch.ones_like(s), t - s
    noise = noise_increment(path)
    if diffusion is not None:
        noise = diffusion * noise
    return _field(state_coeff, x_s) * x_s + _field(velocity_coeff, x_s) * velocity + noise


