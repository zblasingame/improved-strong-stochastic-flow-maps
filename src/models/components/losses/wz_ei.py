"""Wong-Zakai EI flow map on the EI-lifted (annealed) SDE.

Everything trains on the EI lifted state, exactly like the EI-Doss arm —
only the noise handling changes: instead of the Doss-Sussmann transform
(deterministic skeleton + endpoint reconstruction ``X = Z + DM / alpha^{z^2}``),
the annealed martingale is smoothed the Wong-Zakai way and its increment
rides *inside* the map. The lift of the physical state,
``Y = alpha_t^{z^2} X``, turns the annealed flow-as-SDE

    ``dX = [(1+z^2) u - z^2 (alphadot/alpha) X] dt + z nu dW``

into the additive-noise RODE (ordinary chain rule — the smoothed path is
of bounded variation)

    ``dY/dr = (1+z^2) alpha_r^{z^2} u_r(X_r) + Mdot~^z_r``,

whose driver is the annealed EI martingale ``M~^z = z int alpha^{z^2} nu dW``
realized as a degree-``N`` polynomial Brownian motion on the annealed clock
``z^2 int alpha^{2 z^2} nu^2``. The flow map is

    ``Y_t = Y_s + (1+z^2) A^{(z)}_{s,t} u_theta(X_s, s, t; I^M) + (M~^z_t - M~^z_s)``,

with ``A^{(z)} = int alpha^{z^2}`` and ``I^M`` the coefficient features of
the same realization on ``[s, t]`` — the user-facing ``x + (t-s) u + M_t -
M_s`` parameterization at ``z = 0``, and the exact EI-Euler SDE step in Y
units at every zeta. Differentiating the map:

``zeta`` in ``[0, 1]`` selects the annealed family member exactly as in
EI-Doss; ``zeta = 0`` degenerates to the probability-flow ODE (zero
clock, zero driver, plain Euler map ``Y_t = Y_s + (t-s) u``).
"""

from __future__ import annotations

from torch import Tensor

from src.models.components.ei_doss import ei_x_state
from src.models.components.schedule import AffineSchedule, expand_t
from src.models.components.ssfm import SSFMNet


def wz_ei_y_map(
    net: SSFMNet,
    schedule: AffineSchedule,
    y: Tensor,
    s: Tensor,
    t: Tensor,
    coefficients: Tensor,
    delta_m: Tensor,
    zeta: float = 1.0,
) -> Tensor:
    """Wong-Zakai EI flow map in the lifted state ``Y = alpha^{z^2} X``.

    ``Y_t = Y_s + (1+z^2) A^{(z)}_{s,t} u + (M~^z_t - M~^z_s)`` — the same
    affine map as :func:`src.models.components.ei_doss.ei_doss_y_map` plus
    the martingale increment, i.e. the exact EI-Euler SDE step in Y units.
    The backbone still reads physical ``X = Y / alpha_s^{z^2}`` and
    predicts the same velocity ``u``; the increment keeps every derivative
    ``O(1)`` (no ``1 / alpha_t^{z^2}`` reconstruction).
    """
    x = ei_x_state(schedule, y, s, zeta)
    velocity = net.drift(x, s, t, coefficients)
    weight = (1.0 + zeta * zeta) * schedule.annealed_ei_alpha_integral(s, t, zeta)
    return y + expand_t(weight, y) * velocity + delta_m
