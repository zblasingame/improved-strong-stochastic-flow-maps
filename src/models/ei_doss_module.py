"""Lightning module for EI-then-Doss-Sussmann flow maps.

The student is the deterministic EI-Euler skeleton on the physical Doss
state ``Z_s = X_s``. Training anchors the diagonal to a stopped teacher
field (DMF anchor losses); sampling reconstructs ``X_t = Z_t + ΔM / α_t``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.models.components.samplers import ei_doss_sussmann_sampler
from src.models.ssfm_module import SSFMLitModule


class EIDossSussmannLitModule(SSFMLitModule):
    """Train and sample the EI-weighted Doss-Sussmann RODE flow map."""

    @torch.no_grad()
    def sample(
        self,
        latents: Tensor,
        num_steps: int = 4,
        shift: float = 1.0,
        generator: torch.Generator | None = None,
        use_ema: bool = True,
        labels: Tensor | None = None,
    ) -> Tensor:
        """Generate samples with one (annealed) EI-Doss evaluation per interval.

        The annealing amplitude is read from the loss at the current trainer
        step (``loss.zeta_at(self.global_step)``): mid-ramp validation panels
        sample the SDE the network is currently trained on, and after the
        ramp (or with annealing disabled) this is the full ``ζ = 1``
        flow-as-SDE sampler.
        """
        net = self._teacher() if use_ema else self.net
        label_setter = getattr(net, 'set_labels', None)
        if labels is not None and label_setter is None:
            raise TypeError('Labeled EI-Doss sampling requires net.set_labels().')
        if label_setter is not None:
            label_setter(labels)
        try:
            return ei_doss_sussmann_sampler(
                net,
                self.schedule,
                latents,
                num_steps=num_steps,
                degree=self.loss.degree,
                t0=self.loss.t0,
                shift=shift,
                zeta=self.loss.zeta_at(self.global_step),
                generator=generator,
            )
        finally:
            if label_setter is not None:
                label_setter(None)
