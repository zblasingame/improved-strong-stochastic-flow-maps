"""Stochastic flow-map ``LightningModule``.

Trains the SSFM network with a joint objective supplied by the configured loss:
each batch is split by a distillation fraction ``p`` into a diagonal anchor part
and an off-diagonal distillation part. Per-sample losses may be reweighted by a
learned uncertainty MLP (owned by the loss) before the batch mean.

The distillation losses consult an exponential-moving-average (EMA) teacher copy
of the network; the teacher is built at fit time and updated after every
optimizer step.
"""

from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.ema import build_ema
from src.models.components.gaussian import randn_like
from src.models.components.lora import cast_frozen_to
from src.models.components.schedule import AffineSchedule
from src.models.components.ssfm import SSFMNet

_AUTOCAST_DTYPES = {
    'bf16-mixed': torch.bfloat16,
    'bf16': torch.bfloat16,
    '16-mixed': torch.float16,
    '16': torch.float16,
}


def _autocast_dtype(trainer: object) -> torch.dtype | None:
    """Return the autocast dtype for the trainer's precision, if it has one.

    Only mixed-precision settings are mapped. Full ``32``/``64`` precision
    returns ``None`` so frozen weights are left untouched.
    """
    precision = getattr(trainer, 'precision', None)
    return _AUTOCAST_DTYPES.get(str(precision))


def needs_pretrained_compile(loss: object) -> bool:
    """Return whether the loss calls the frozen pretrained teacher path.

    True for the baked-CFG anchor, the legacy ``'pretrained'`` alias (the
    direct form at the frozen checkpoint), and any guidance whose field
    source is the frozen teacher (``'frozen'`` or its legacy spelling
    ``'teacher'``) -- whether the form is ``direct`` or ``mg``.
    """
    guidance = getattr(loss, 'guidance', None)
    source = getattr(loss, 'guidance_source', None)
    if guidance in ('cfg', 'pretrained'):
        return True
    return source in ('teacher', 'frozen')


class SSFMLitModule(LightningModule):
    """SSFM training module using the joint instantaneous + diagonal loss."""

    def __init__(
        self,
        net: SSFMNet,
        schedule: AffineSchedule,
        loss: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        compile: bool = False,
        ema_decay: float = 0.9999,
        scheduler_interval: str = 'epoch',
        scheduler_monitor: str | None = 'train/loss',
        log_per_bin_scalars: bool = False,
    ) -> None:
        """Initialize an ``SSFMLitModule``.

        :param net: SSFM backbone, called via the loss as ``net.forward``/``net.drift``.
        :param schedule: Affine noise schedule providing interpolation/drift targets.
        :param loss: Joint SSFM loss (may own the uncertainty MLP; trains jointly).
        :param optimizer: Partially-instantiated optimizer (``params`` injected later).
        :param scheduler: Partially-instantiated LR scheduler (``optimizer`` injected later).
        :param compile: Whether to ``torch.compile`` the net at fit time.
        :param ema_decay: EMA decay for the vanilla EMA teacher.
        :param scheduler_interval: Lightning LR-scheduler call interval — ``'epoch'``
            (default, for ReduceLROnPlateau) or ``'step'`` (for warmup-cosine).
        :param scheduler_monitor: Metric the scheduler monitors (``'train/loss'``
            by default, for ReduceLROnPlateau). Set to ``None`` for schedulers that
            do not need a monitor (e.g. warmup-cosine).
        :param log_per_bin_scalars: If ``True``, also log each per-bin diagnostic
            value as an individual scalar (``{stage}/{parent}/{axis}_in_[lo,hi]``).
            Defaults to ``False``; the per-bin vectors are instead surfaced as
            heatmaps by :class:`~src.models.callbacks.LossFieldHeatmap`, which
            reads :attr:`last_bin_diagnostics`.
        """
        super().__init__()
        # net/schedule are modules/objects, not plain hyperparameters.
        self.save_hyperparameters(logger=False, ignore=['net', 'schedule', 'loss'])

        self.net = net
        self.schedule = schedule
        # loss is an nn.Module owning the uncertainty MLP -> its params join self.parameters().
        self.loss = loss

        self.ema: Any = None
        self._pending_ema_state: dict | None = None
        self._resume_stripped_frozen_base: list[str] = []
        self._nimg = 0
        self._skipped_batch = False
        # Latest 1-D per-bin diagnostic vectors, stashed for LossFieldHeatmap.
        self.last_bin_diagnostics: dict[str, torch.Tensor] = {}

        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def _teacher(self) -> SSFMNet:
        """Return the EMA teacher network (falls back to the live net pre-fit)."""
        if self.ema is None:
            return self.net
        return self.ema.get_model(0)

    @torch.no_grad()
    def sample(
        self,
        latents: torch.Tensor,
        num_steps: int = 4,
        shift: float = 1.0,
        generator: torch.Generator | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Generate samples by chaining the generic stochastic flow map.

        Without this hook, sampling callbacks fall back to a deterministic
        ``fm_euler_sampler`` ODE, which silently drops the Brownian term the
        x-space SSFM objectives are trained with.

        :param latents: Initial noise ``x_0`` of shape ``(B, ...)``.
        :param num_steps: Number of flow-map steps.
        :param shift: Time-grid warp factor.
        :param generator: Optional RNG for reproducible per-step paths.
        :param use_ema: Sample from the EMA teacher (default) or the live net.
        :return: The generated data sample ``(B, ...)``.
        """
        from src.models.components.samplers import flow_map_sampler

        net = self._teacher() if use_ema else self.net
        return flow_map_sampler(
            net,
            latents,
            num_steps=num_steps,
            degree=self.loss.degree,
            shift=shift,
            generator=generator,
            schedule=self.schedule,
            analytic_diffusion=getattr(self.loss, 'analytic_diffusion', False),
            t_start=self.loss.epsilon,
        )

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the joint SSFM loss and diagnostics for one batch.

        :param batch: Tuple ``(x_1, _)`` of data images and (ignored) labels.
        :return: Tuple ``(total_loss, diagnostics)``.
        """
        x_1, labels = batch
        x_0 = randn_like(x_1)
        teacher = self._teacher()
        self._set_labels(self.net, labels)
        self._set_labels(teacher, labels)
        try:
            return self.loss(self.net, teacher, x_1, x_0, self.global_step)
        finally:
            self._set_labels(self.net, None)
            self._set_labels(teacher, None)

    @staticmethod
    def _set_labels(net: SSFMNet, labels: torch.Tensor | None) -> None:
        """Assign labels when a network supports transient class conditioning."""
        setter = getattr(net, 'set_labels', None)
        if setter is not None:
            setter(labels)

    def _log_diagnostics(self, stage: str, diagnostics: dict[str, torch.Tensor]) -> None:
        """Log scalar diagnostics and stash per-bin vectors for the heatmap.

        Scalars (0-D) are logged directly as ``{stage}/{key}``.

        Binned (1-D) keys carry a ``_by_{axis}`` suffix (e.g.
        ``uncertainty_mlp_by_s``). They are stashed on
        :attr:`last_bin_diagnostics` (keyed by the bare diagnostic name) so
        :class:`~src.models.callbacks.LossFieldHeatmap` can accumulate them into
        step-vs-bin heatmaps. When :attr:`hparams.log_per_bin_scalars` is set,
        each bin is *additionally* logged as an individual scalar
        ``{stage}/{parent}/{axis}_in_[{lo},{hi}]`` (legacy behaviour).

        :param stage: Log-key stage prefix, e.g. ``'train'``.
        :param diagnostics: Mixed 0-D scalar and 1-D per-bin diagnostic tensors.
        """
        edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_vectors: dict[str, torch.Tensor] = {}
        for key, value in diagnostics.items():
            if value.ndim == 0:
                self.log(f'{stage}/{key}', value, on_step=True, on_epoch=False)
                continue
            bin_vectors[key] = value.detach()
            if not self.hparams.log_per_bin_scalars:
                continue
            # Legacy: parse ``{parent}_by_{axis}`` -> one scalar per bin.
            parent, _, axis = key.rpartition('_by_')
            if not axis:  # unexpected shape; fall back to flat bin index
                parent, axis = key, 'bin'
            for i, binned in enumerate(value):
                lo, hi = edges[i], edges[i + 1]
                self.log(
                    f'{stage}/{parent}/{axis}_in_[{lo},{hi}]',
                    binned,
                    on_step=True,
                    on_epoch=False,
                )
        self.last_bin_diagnostics = bin_vectors

    def on_train_start(self) -> None:
        """No-op (val_loss no longer tracked; FID/grids come from callbacks)."""

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor | None:
        """Single training step; logs ``train/loss`` and diagnostics.

        If the loss is non-finite (NaN/inf) the batch is skipped: no gradients
        are accumulated, no optimizer step is taken, and the EMA teacher is not
        updated. This prevents a single bad batch from permanently poisoning the
        parameters. ``train/skipped_batches`` is logged as a sentinel.
        """
        loss, diagnostics = self.model_step(batch)
        if not torch.isfinite(loss):
            self._skipped_batch = True
            self.log('train/skipped_batches', 1.0, on_step=True, on_epoch=False)
            return None
        self._skipped_batch = False
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
        self._log_diagnostics('train', diagnostics)
        return loss

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update the EMA teacher after each optimizer step (skip on bad batch)."""
        if self.ema is None or self._skipped_batch:
            return
        batch_size = batch[0].shape[0]
        self._nimg += batch_size
        self.ema.update(self.net, t_next=self._nimg, t_delta=batch_size)

    def _frozen_base_state_keys(self) -> list[str]:
        """Net-relative names of the frozen ``net.base`` parameters.

        LoRA-adapted pretrained backbones keep their frozen ``base`` weights
        bit-identical to the upstream checkpoint they were built from (a resume
        rebuilds them from the same source), so re-writing ~1.3 GB of module
        weights plus ~2.7 GB of float32 EMA shadow per save is pure I/O
        overhead. Only *frozen* parameters are named here: LoRA adapters and
        every other trainable live under ``net.base`` too and must be saved.
        Nets without a ``base`` attribute return nothing (full checkpoint).
        """
        base = getattr(self.net, 'base', None)
        if base is None:
            return []
        return [name for name, parameter in base.named_parameters() if not parameter.requires_grad]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Embed the EMA shadow weights and strip the frozen base from the file.

        The EMA tracker is a plain object outside the module tree, so without
        the embed its thermal history is lost on every save (a resume would
        re-seed the shadow from the live weights and need ~1/ddecay steps to
        forget the restart). The LightningModule hook receives only the
        checkpoint dict (unlike the Callback variant).

        The frozen ``net.base`` weights (and their EMA shadow entries) are
        dropped from the payload and their names recorded under
        ``stripped_frozen_base``: they are reconstructible from the upstream
        pretrained checkpoint at resume time, and they dominate the write
        cost on network storage.
        """
        frozen = self._frozen_base_state_keys()
        if frozen:
            # Module state_dict keys are 'net.base.<name>' (the frozen names
            # are base-relative); the EMA shadow is a deepcopy of the *net*,
            # so its keys carry the 'base.' prefix only.
            for name in frozen:
                checkpoint['state_dict'].pop(f'net.base.{name}', None)
            checkpoint['stripped_frozen_base'] = sorted(frozen)
        if self.ema is not None:
            ema_state = self.ema.state_dict()
            if frozen and isinstance(ema_state.get('ema_model'), dict):
                for name in frozen:
                    ema_state['ema_model'].pop(f'base.{name}', None)
            checkpoint['ema_state'] = ema_state

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Stash embedded EMA weights and the strip manifest until :meth:`setup`."""
        self._pending_ema_state = checkpoint.pop('ema_state', None)
        self._resume_stripped_frozen_base = list(checkpoint.pop('stripped_frozen_base', []))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: ANN001
        """Restore weights, allowing exactly the stripped frozen-base keys missing.

        Checkpoints written by :meth:`on_save_checkpoint` intentionally omit
        the frozen ``net.base`` weights; the freshly built net already carries
        them (bit-identical, from the same upstream checkpoint). Any other
        missing key -- and every unexpected key -- is still a hard error, so
        corrupt or mismatched checkpoints fail loudly. Legacy (unstripped)
        checkpoints restore through the default strict path untouched.
        """
        stripped = set(self._resume_stripped_frozen_base)
        if not strict or not stripped:
            return super().load_state_dict(state_dict, strict=strict)
        result = super().load_state_dict(state_dict, strict=False)
        allowed = {f'net.base.{name}' for name in stripped}
        missing = [key for key in result.missing_keys if key not in allowed]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f'Error(s) in loading state_dict for {type(self).__name__}: '
                f'missing keys {sorted(missing)[:8]}, '
                f'unexpected keys {sorted(result.unexpected_keys)[:8]}.'
            )
        return result

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """No-op validation step.

        Validation exists only to drive the FID / sample-grid callbacks
        (``on_validation_epoch_end``); no val loss is logged.
        """

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Single test step; logs ``test/loss``."""
        loss, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Build the EMA teacher and optionally ``torch.compile`` the net at fit time."""
        if stage != 'fit':
            return
        # Lightning moves the LightningModule to the accelerator device before
        # ``setup``, but the nested ``net`` may still be on CPU. Move it now so
        # the EMA clone and ``init_from_model`` land on the same device.
        self.net = self.net.to(self.device)
        # Under bf16 autocast every frozen weight is rounded to bfloat16 before
        # each matmul anyway, so materializing that cast once is bitwise
        # identical while halving the resident frozen backbone. Trainable
        # (LoRA) parameters stay float32. This runs before the EMA clone so the
        # shadow copy is seeded from the same values; _to_float32_ema then keeps
        # the EMA itself in float32 so small decays are not quantized away.
        autocast_dtype = _autocast_dtype(self.trainer)
        if autocast_dtype is not None:
            cast_frozen_to(self.net, autocast_dtype)
        self.ema = build_ema(
            self.net,
            decay=self.hparams.ema_decay,
            device=self.device,
        )
        self.ema.init_from_model(self.net)
        # Restore shadow weights from a resumed checkpoint (the EMA tracker is
        # not part of the module tree, so Lightning's state_dict never sees it;
        # on_load_checkpoint stashes it here until the EMA is rebuilt). A
        # stripped payload (frozen base omitted) merges over the freshly
        # seeded shadow: the frozen entries keep their fresh values -- EMA of
        # constants -- while every trainable entry comes from the checkpoint.
        if self._pending_ema_state is not None:
            pending = self._pending_ema_state
            if self._resume_stripped_frozen_base and isinstance(pending.get('ema_model'), dict):
                merged = self.ema.state_dict()
                merged['ema_model'].update(pending['ema_model'])
                for key, value in pending.items():
                    if key != 'ema_model':
                        merged[key] = value
                pending = merged
            self.ema.load_state_dict(pending)
            self._pending_ema_state = None
        if self.hparams.compile:
            compile_backbone = getattr(self.net, 'compile_backbone', None)
            compile_ema_backbone = getattr(self.ema.get_model(0), 'compile_backbone', None)
            if compile_backbone is None or compile_ema_backbone is None:
                raise TypeError('Compiled SSFM training requires net.compile_backbone().')
            # EMA is the PSD teacher and, for mg-from-ema, the MG diagonal.
            # Compile the frozen pretrained path only when the loss actually
            # calls it (CFG distillation or guidance_source='teacher').
            compile_backbone(
                pretrained=needs_pretrained_compile(self.loss),
                diagonal=True,
                field=True,
            )
            compile_ema_backbone(pretrained=False, diagonal=True, field=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer (and optional LR scheduler) from injected partials."""
        optimizer = self._build_optimizer()
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            lr_sched: dict[str, Any] = {
                'scheduler': scheduler,
                'interval': self.hparams.scheduler_interval,
                'frequency': 1,
            }
            if self.hparams.scheduler_monitor is not None:
                lr_sched['monitor'] = self.hparams.scheduler_monitor
            return {'optimizer': optimizer, 'lr_scheduler': lr_sched}
        return {'optimizer': optimizer}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer over all trainable parameters."""
        opt_partial = self.hparams.optimizer
        trainable = [
            parameter for parameter in self.trainer.model.parameters() if parameter.requires_grad
        ]
        return opt_partial(params=trainable)


if __name__ == '__main__':
    _ = SSFMLitModule(None, None, None, None, None)
