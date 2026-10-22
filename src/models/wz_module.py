"""Lightning module for the Wong-Zakai EI-lifted flow-map arm.

:class:`WZEISussmannLitModule` reuses the martingale-increment sampling chain
of its EI-Doss sibling — the Wong-Zakai change is in the *loss* (the
martingale increment rides inside the differentiated map and the driver
``Mdot`` enters every field), not in the sampling recursion: each step is
``Y_t = Y_s + (1+z^2) A^{(z)} u + DM~^z`` in the lifted state, which in
physical X units is the identical recursion walked by
:func:`~src.models.components.samplers.ei_doss_sussmann_sampler`
(``X_t = (alpha_s/alpha_t)^{z^2} X_s + ((1+z^2) A^{(z)} / alpha_t^{z^2}) u
+ DM~^z / alpha_t^{z^2}``).

The subclass exists as a distinct type so Hydra configs select the
Wong-Zakai losses while the sampling callbacks (which route by module
type) keep their existing EI-Doss wiring.
"""

from __future__ import annotations

from src.models.ei_doss_module import EIDossSussmannLitModule


class WZEISussmannLitModule(EIDossSussmannLitModule):
    """Train the EI-lifted Wong-Zakai map; sample via the EI-Doss chain.

    Training uses the WZ-iMF anchor losses
    (e.g. :class:`~src.models.components.losses.dmf_wz_imf.DMFWZEIIMFAnchorLoss`)
    on the lifted map ``Y_t = Y_s + (1+z^2) A^{(z)} u + DM~^z``. In physical X
    units the map's per-step recursion is identical to the (annealed) EI-Doss
    sampler, so the inherited
    :meth:`~src.models.ei_doss_module.EIDossSussmannLitModule.sample`
    (including the ``zeta_at`` threading for annealed runs) is reused
    verbatim.
    """

    pass
