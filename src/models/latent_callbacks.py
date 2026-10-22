"""VAE-decoding image and FID callbacks for latent-space flow maps."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from lightning import LightningModule, Trainer
from torch import Tensor

from src.models.callbacks import (
    FIDCallback,
    SampleGridCallback,
    _bf16_inference,
    _resolution_grid,
    _training_zeta,
)
from src.models.components.gaussian import randn
from src.models.components.samplers import (
    chen_matched_ei_doss_rows,
    ei_doss_sussmann_sampler,
    ei_euler_sampler,
    multistep_ei_euler_sampler,
)
from src.models.components.vae_decoder import StabilityVAEDecoder
from src.models.ei_doss_module import EIDossSussmannLitModule


def _labels(net: torch.nn.Module, count: int, seed: int, device: torch.device) -> Tensor:
    # Seeded on CPU so labels stay identical across devices and epochs.
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, net.num_classes, (count,), generator=generator).to(device)


class LatentSampleGridCallback(SampleGridCallback):
    """Sample EI-Euler or EI-Doss latents and decode each step-count row."""

    def __init__(self, vae_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decoder = StabilityVAEDecoder(vae_name)

    def _grid_state_shape(self, trainer) -> tuple[int, int, int]:
        """Return the latent state shape used for the grid start noise.

        Overridden by decoder-specific subclasses with a different latent
        geometry (e.g. the Sana DC-AE callbacks).
        """
        del trainer
        return 4, 32, 32

    def _render_grid(self, pl_module: LightningModule) -> dict[str, Tensor]:
        """Render one decoded image row per dyadic step count.

        Parameters
        ----------
        pl_module : lightning.LightningModule
            Module supplying the network, schedule, and loss configuration.

        Returns
        -------
        dict[str, torch.Tensor]
            Single-entry mapping from log suffix to the stacked image grid.
        """
        device = pl_module.device
        self.decoder.to(device)
        generator = torch.Generator(device=device).manual_seed(self.seed or 0)
        shape = (self.num_samples, *self._grid_state_shape(pl_module.trainer))
        latents = randn(shape, device=device, generator=generator)
        net = pl_module.net
        if self.use_ema and getattr(pl_module, 'ema', None) is not None:
            net = pl_module.ema.get_model(0)
        labels = _labels(net, self.num_samples, self.label_seed, device)
        step_counts = [2**i for i in reversed(range(self.max_steps.bit_length()))]
        grids = {'': _resolution_grid(self._chen_rows(net, pl_module, latents, labels, generator))}
        if self.log_independent_brownian_grid:
            independent = torch.stack(
                [
                    self._sample_rows(
                        net, pl_module, latents, labels, [steps], self._row_generator(device, row)
                    )
                    for row, steps in enumerate(step_counts)
                ]
            ).squeeze(1)
            grids['_stochastic'] = _resolution_grid(independent)
        return grids

    def _chen_rows(
        self,
        net: torch.nn.Module,
        pl_module: LightningModule,
        latents: Tensor,
        labels: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        """Decode every dyadic row from one Chen-coupled EI-clock realization.

        EI-Doss/WZ modules use :func:`chen_matched_ei_doss_rows`: fresh
        degree-N leaves are drawn on the finest clock grid from the *shared*
        generator and composed with exact Chen relations for every coarser
        row, so all rows ride one matched Brownian path and differ only in
        the time discretization.
        """
        was_training = net.training
        net.eval()
        net.set_labels(labels)
        try:
            step_counts = [2**i for i in reversed(range(self.max_steps.bit_length()))]
            if isinstance(pl_module, EIDossSussmannLitModule):
                zeta = _training_zeta(pl_module)
                samples = chen_matched_ei_doss_rows(
                    net,
                    pl_module.schedule,
                    latents,
                    step_counts,
                    degree=self.degree,
                    t0=pl_module.loss.t0,
                    zeta=zeta,
                    generator=generator,
                )
            else:
                samples, _ = multistep_ei_euler_sampler(
                    net,
                    pl_module.schedule,
                    latents,
                    max_steps=self.max_steps,
                    degree=self.degree,
                    t0=pl_module.loss.t0,
                    generator=generator,
                )
            return torch.stack([self.decoder(row) for row in samples])
        finally:
            net.set_labels(None)
            if was_training:
                net.train()

    def _sample_rows(
        self,
        net: torch.nn.Module,
        pl_module: LightningModule,
        latents: Tensor,
        labels: Tensor,
        step_counts: list[int],
        generator: torch.Generator,
    ) -> Tensor:
        """Sample and decode one row per step count on shared latents."""
        rows = []
        was_training = net.training
        net.eval()
        try:
            for steps in step_counts:
                net.set_labels(labels)
                try:
                    if isinstance(pl_module, EIDossSussmannLitModule):
                        samples = ei_doss_sussmann_sampler(
                            net,
                            pl_module.schedule,
                            latents,
                            num_steps=steps,
                            degree=self.degree,
                            t0=pl_module.loss.t0,
                            zeta=_training_zeta(pl_module),
                            generator=generator,
                        )
                    else:
                        samples = ei_euler_sampler(
                            net,
                            pl_module.schedule,
                            latents,
                            num_steps=steps,
                            degree=self.degree,
                            t0=pl_module.loss.t0,
                            generator=generator,
                        )
                finally:
                    net.set_labels(None)
                rows.append(self.decoder(samples))
        finally:
            if was_training:
                net.train()
        return torch.stack(rows)


class LatentFIDCallback(FIDCallback):
    """Generate EI-Euler or EI-Doss latents and compute decoded pixel-space FID."""

    def __init__(self, vae_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.decoder = StabilityVAEDecoder(vae_name)

    def _sample_shape(self, trainer: Trainer) -> tuple[int, int, int]:
        """Return the fixed SD-VAE latent shape."""
        del trainer
        return 4, 32, 32

    @torch.no_grad()
    def _sample_fake(
        self,
        net: torch.nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        schedule=None,
        prediction: str = 'velocity',
        doss: bool = False,
        ei_doss: bool = False,
        xpred_sde: bool = False,
        t0: float = 0.01,
        labels: Tensor | None = None,
        zeta: float = 1.0,
    ) -> Iterator[Tensor]:
        """Yield VAE-decoded image batches in model range ``[-1, 1]``.

        FM sampler options are fixed for the latent pipeline, so ``prediction``,
        ``doss``, and ``xpred_sde`` are accepted for base-class compatibility and
        ignored; ``shape`` is the latent geometry from ``_sample_shape``.

        Parameters
        ----------
        net : torch.nn.Module
            Network to sample from.
        shape : tuple[int, ...]
            Latent shape ``(C, H, W)``; the batch dimension is added per chunk.
        device : torch.device
            Device used for sampling and decoding.
        schedule : optional
            Affine schedule passed to the selected EI sampler.
        prediction : str, optional
            Ignored; retained for base-class compatibility.
        doss : bool, optional
            Ignored; retained for base-class compatibility.
        ei_doss : bool, optional
            Select the EI-Doss-Sussmann sampler instead of EI-Euler.
        xpred_sde : bool, optional
            Ignored; retained for base-class compatibility.
        t0 : float, optional
            Martingale clock origin, by default ``0.01``.
        labels : torch.Tensor or None, optional
            Class labels; sampled deterministically when ``None``.
        zeta : float, optional
            Annealing amplitude of the sampled (annealed) EI-Doss SDE, by
            default ``1.0``; the caller threads the module's training zeta.

        Yields
        ------
        torch.Tensor
            Decoded image batches in the model's ``[-1, 1]`` range, matching
            :class:`FIDCallback`'s metric preprocessing contract.
        """
        del prediction, doss, xpred_sde
        self.decoder.to(device)
        generator = torch.Generator(device=device).manual_seed(self.seed or 0)
        if labels is None:
            labels = _labels(net, self.num_samples, self.label_seed, device)
        for offset in range(0, self.num_samples, self.batch_size):
            count = min(self.batch_size, self.num_samples - offset)
            latents = randn((count, *shape), device=device, generator=generator)
            net.set_labels(labels[offset : offset + count])
            try:
                with _bf16_inference(device):
                    sampler = ei_doss_sussmann_sampler if ei_doss else ei_euler_sampler
                    if ei_doss:
                        # Sample the SDE the module is currently trained on
                        # (mid-ramp panels must not draw the zeta = 1 SDE).
                        samples = sampler(
                            net,
                            schedule,
                            latents,
                            num_steps=self.sampling_steps,
                            degree=self.degree,
                            t0=t0,
                            zeta=zeta,
                            generator=generator,
                        )
                    else:
                        samples = sampler(
                            net,
                            schedule,
                            latents,
                            num_steps=self.sampling_steps,
                            degree=self.degree,
                            t0=t0,
                            generator=generator,
                        )
            finally:
                net.set_labels(None)
            # The base callback converts model-range images for each FID protocol.
            yield self.decoder(samples).mul(2.0).sub(1.0)

    def _evaluate(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step_counts: tuple[int, ...] | None = None,
    ) -> None:
        """Delegate labeled latent profile evaluation to the base callback."""
        super()._evaluate(trainer, pl_module, step_counts)
