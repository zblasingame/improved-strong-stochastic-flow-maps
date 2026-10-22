"""Lightning module for the augmented FKC toy experiments.

Training is fully analytic: the :class:`FKCProblem` supplies exact expert
densities/scores and composed targets, so the datamodule only needs to set the
batch size (it yields placeholder index batches). The module owns

* the augmented :class:`~src.models.components.fkc.network.AugmentedFKCMLP`;
* the analytic :class:`~src.models.components.fkc.problem.FKCProblem`;
* the teacher-driven :class:`~src.models.components.fkc.loss.FKCWZIMFLoss`;
* an exponential-moving-average copy of the network, used for sampling when
  ``use_ema`` is requested (restored from the checkpoint like any submodule).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.fkc.sampling import FKCSamples, learned_fkc


class FKCLitModule(LightningModule):
    """Train and sample the augmented FKC flow map on 2D toy problems."""

    def __init__(
        self,
        net: torch.nn.Module,
        problem: torch.nn.Module,
        loss: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        schedule: Any = None,
        ema_decay: float = 0.999,
        compile: bool = False,
    ) -> None:
        """Initialize the module.

        The ``optimizer``/``scheduler`` partials are kept out of ``hparams`` so
        checkpoints stay free of unpicklable callables; evaluation reconstructs
        the module from the saved Hydra config and loads only the ``state_dict``.

        :param net: Augmented FKC backbone ``(x, s, t, coefficients) -> (u, g)``.
        :param problem: Analytic :class:`FKCProblem` (buffers only).
        :param loss: :class:`FKCWZIMFLoss` instance.
        :param optimizer: Optimizer partial (Hydra ``_partial_``).
        :param scheduler: Optional LR-scheduler partial.
        :param schedule: The affine noise schedule (also carried by ``problem``
            and ``loss``; accepted so the Hydra ``schedule`` group composes).
        :param ema_decay: EMA decay for the sampling teacher; ``< 0`` disables.
        :param compile: Whether to ``torch.compile`` the network at fit time.
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=['net', 'problem', 'loss', 'optimizer', 'scheduler', 'schedule'],
        )
        self.net = net
        self.problem = problem
        self.loss = loss
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.schedule = schedule
        self.ema_net = deepcopy(net)
        for parameter in self.ema_net.parameters():
            parameter.requires_grad_(False)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def _teacher(self, use_ema: bool) -> torch.nn.Module:
        """Return the EMA teacher when available, else the live network."""
        if use_ema and self.hparams.ema_decay >= 0.0:
            return self.ema_net
        return self.net

    def _generator(self, seed: int) -> torch.Generator:
        """Build a deterministic RNG on the module's device."""
        device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        return torch.Generator(device=device).manual_seed(seed)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute the augmented WZ-iMF loss for one batch of placeholder indices."""
        batch_size = batch[0].shape[0]
        result = self._run_loss(batch_size)
        self.train_loss(result['loss'])
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # iMF flow-map arm (75% of the batch) and flow-matching diagonal arm
        # (25%), each split into the augmented x (spatial) and w (log-weight)
        # blocks.
        self.log('train/imf_loss', result['imf_loss'], on_step=False, on_epoch=True)
        self.log('train/imf_x', result['imf_x'], on_step=False, on_epoch=True)
        self.log('train/imf_w', result['imf_w'], on_step=False, on_epoch=True)
        self.log('train/fm_loss', result['fm_loss'], on_step=False, on_epoch=True)
        self.log('train/fm_x', result['fm_x'], on_step=False, on_epoch=True)
        self.log('train/fm_w', result['fm_w'], on_step=False, on_epoch=True)
        return result['loss']

    def _run_loss(self, batch_size: int, *, generator: torch.Generator | None = None) -> dict:
        """Run the loss, passing the EMA teacher for PSD-style losses."""
        if getattr(self.loss, 'uses_ema', False):
            return self.loss(self.net, self.ema_net, self.problem, batch_size, generator=generator)
        return self.loss(self.net, self.problem, batch_size, generator=generator)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update the EMA teacher copy after each optimizer step."""
        if self.hparams.ema_decay < 0.0:
            return
        decay = self.hparams.ema_decay
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_net.parameters(), self.net.parameters(), strict=True
            ):
                ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Log a seeded validation loss so the curve is comparable across epochs."""
        batch_size = batch[0].shape[0]
        result = self._run_loss(batch_size, generator=self._generator(1234))
        self.val_loss(result['loss'])
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Log a seeded test loss; ``src.train`` runs ``trainer.test`` after fit."""
        batch_size = batch[0].shape[0]
        result = self._run_loss(batch_size, generator=self._generator(4321))
        self.test_loss(result['loss'])
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        *,
        steps: int = 1,
        t0: float | None = None,
        t1: float = 1.0,
        use_ema: bool = True,
        generator: torch.Generator | None = None,
    ) -> FKCSamples:
        """Draw ``n`` weighted FKC samples with the few-step learned map.

        :param n: Number of particles.
        :param steps: Number of learned steps (the NFE budget).
        :param t0: Start time; defaults to the loss's clock origin.
        :param t1: Terminal time.
        :param use_ema: Use the EMA teacher (default) or the live network.
        :param generator: Optional RNG.
        :return: Weighted :class:`FKCSamples`.
        """
        net = self._teacher(use_ema)
        start = self.loss.t0 if t0 is None else t0
        x0 = self.problem.sample_target(n, start, generator=generator)
        return learned_fkc(
            net,
            x0,
            t0=start,
            t1=t1,
            steps=steps,
            degree=self.loss.degree,
            generator=generator,
            schedule=getattr(self.loss, 'schedule', self.schedule),
        )

    def configure_optimizers(self) -> Any:
        """Build the optimizer (and optional scheduler) from injected partials."""
        if self.optimizer_partial is None:
            raise RuntimeError(
                'No optimizer partial was supplied; pass the Hydra optimizer config '
                'when reconstructing the module for training.'
            )
        optimizer = self.optimizer_partial(params=self.net.parameters())
        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'train/loss',
                },
            }
        return optimizer

    def setup(self, stage: str) -> None:
        """Optionally compile the network at fit time."""
        if stage == 'fit' and self.hparams.compile:
            self.net = torch.compile(self.net)
