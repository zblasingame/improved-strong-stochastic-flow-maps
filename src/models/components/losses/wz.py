"""Wong-Zakai map/driver building blocks for the WZ-family losses.

The generative SDE ``dX = b(r, X) dr + nu_r dW_r`` is smoothed the Wong-Zakai
way at the martingale level: ``M_r = int_{t0}^r nu dW`` is realized as a
degree-``N`` polynomial Brownian motion on the schedule clock
``Sigma = int nu^2``, so the SDE becomes the smooth random ODE

    ``dX/dr = b(r, X) + Mdot_r``,   ``Mdot_r = dM^N_r / dr``,

(the chain rule through the clock is exact, so ``Mdot`` is the smoothed
white-noise term). The flow map owns the martingale increment explicitly —
no Doss-Sussmann transform and no reconstruction:

    ``Psi_{s,t}(x) = x + (t - s) u_theta(x, s, t; I^M_{s,t}) + (M_t - M_s)``,

with ``I^M_{s,t}`` the coefficient features of the same realization on
``[s, t]``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.func import jvp

from src.models.components.doss_sussmann import GlobalMartingalePath
from src.models.components.ssfm import SSFMNet


def _per_sample_mse(pred: Tensor, target: Tensor | None = None) -> Tensor:
    """Mean squared error reduced over all but the batch axis -> ``(B,)``."""
    residual = pred if target is None else pred - target
    return residual.square().flatten(1).mean(1)


def _require_euler(net: SSFMNet) -> None:
    """Require an ``euler`` map: the WZ map owns the increment, never a head."""
    if net.parameterization != 'euler':
        raise ValueError(
            f'WZ requires net.parameterization="euler"; got {net.parameterization!r}.'
        )


def wz_driver(realization: GlobalMartingalePath, r: Tensor, reference: Tensor) -> Tensor:
    """Return the physical-time driver derivative ``Mdot_r`` of a realization.

    A forward-mode JVP through :meth:`GlobalMartingalePath.global_value`
    chains the schedule clock exactly (closed-form or quadrature), so the
    result is the smoothed white-noise term of the Wong-Zakai RODE
    ``dX/dr = b(r, X) + Mdot_r``. ``r`` may be per-sample ``(B,)`` times.
    """
    r_field = r.reshape(-1, *([1] * (reference.ndim - 1))).to(
        device=realization.path.coeffs.device, dtype=realization.path.coeffs.dtype
    )
    _, driver = jvp(
        lambda time: realization.global_value(time),
        (r_field,),
        (torch.ones_like(r_field),),
    )
    return driver


