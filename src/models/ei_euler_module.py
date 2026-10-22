"""Lightning module for velocity-parameterized EI-Euler flow maps."""

import torch
from torch import Tensor

from src.models.components.samplers import ei_euler_sampler
from src.models.ssfm_module import SSFMLitModule


class EIEulerLitModule(SSFMLitModule):
    """Train and sample EI-Euler velocity flow maps."""

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
        """Generate samples with one EI-Euler evaluation per interval."""
        net = self._teacher() if use_ema else self.net
        label_setter = getattr(net, 'set_labels', None)
        if labels is not None and label_setter is None:
            raise TypeError('Labeled EI-Euler sampling requires net.set_labels().')
        if label_setter is not None:
            label_setter(labels)
        try:
            return ei_euler_sampler(
                net,
                self.schedule,
                latents,
                num_steps=num_steps,
                degree=self.loss.degree,
                t0=self.loss.t0,
                shift=shift,
                generator=generator,
            )
        finally:
            if label_setter is not None:
                label_setter(None)
