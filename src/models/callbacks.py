"""Lightning callbacks for the SSFM module.

:class:`SampleGridCallback` periodically renders a *multi-resolution* sample grid
during validation: one row per dyadic step count (``max_steps, max_steps/2, ...,
1``), all rows sharing the **same source noise and the same realized Brownian
motion** via the Chen-relation dyadic tree
(:func:`~src.models.components.samplers.multistep_flow_map_sampler`). The only
difference between rows is the number of flow-map steps (discretization), making
the grid a clean step-count / strong-convergence sweep. The grid is logged to
Weights & Biases when a ``WandbLogger`` is attached.

:class:`FIDCallback` periodically computes Frechet Inception Distance during
validation, one score per EMA profile (``fid/ema``) plus the
live non-EMA model (``fid/non_ema``). Real reference statistics are precomputed
offline (see ``scripts/compute_fid_stats.py``) and reused across all profiles;
fake samples are drawn with shared seeded noise so the profiles are compared on an
identical noise budget.

:class:`LossFieldHeatmap` accumulates the per-bin loss / uncertainty-MLP
diagnostics (binned by the flow-map start time ``s`` or step size ``h = t - s``)
into ``(num_bins, num_steps)`` histories and periodically logs them to Weights &
Biases as matplotlib heatmaps (x = training step, y = time bin), so one image
shows how each loss / weight profile over time evolves throughout training.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from torchvision.utils import make_grid

from src.models.components.gaussian import randn
from src.models.components.metrics import (
    FIDComputer,
    HistoricalFIDComputer,
    to_fid_uint8,
    to_uint8_images,
)
from src.models.components.samplers import (
    ei_doss_sussmann_sampler,
    flow_map_sampler,
    fm_em_sampler,
    fm_euler_sampler,
    multistep_flow_map_sampler,
)
from src.models.components.ssfm import SSFMNet
from src.models.components.strong_convergence import (
    ei_doss_rollout_batch,
    sample_piecewise_brownian,
    sample_piecewise_ei_clock,
    strong_error_batch,
)
from src.models.ei_doss_module import EIDossSussmannLitModule
from src.models.ei_euler_module import EIEulerLitModule

log = logging.getLogger(__name__)


def _is_ssfm(net: torch.nn.Module) -> bool:
    """Return ``True`` when *net* is a stochastic-flow-map network."""
    return isinstance(net, SSFMNet)


def _is_ei_euler(pl_module: LightningModule) -> bool:
    """Return ``True`` for exponential-integrator Euler flow-map modules."""
    return isinstance(pl_module, EIEulerLitModule)


def _is_ei_doss(pl_module: LightningModule) -> bool:
    """Return ``True`` for EI-then-Doss-Sussmann RODE flow-map modules."""
    return isinstance(pl_module, EIDossSussmannLitModule)


def _training_zeta(pl_module: LightningModule) -> float:
    """Return the annealing amplitude the module is currently trained at.

    Inference (grids, FID) must sample the SDE the network is being trained
    on: a mid-ramp validation panel drawn at ``zeta = 1`` would show the full
    flow-as-SDE while the model only fits the (near-)ODE. Losses without a
    ``zeta_at`` (no annealing support) default to the un-annealed ``zeta = 1``.
    """
    zeta_at = getattr(getattr(pl_module, 'loss', None), 'zeta_at', None)
    return float(zeta_at(pl_module.global_step)) if zeta_at is not None else 1.0


def _resolution_grid(samples: Tensor) -> Tensor:
    """Tile a ``(R, B, C, H, W)`` sample stack into one ``(C, H', W')`` grid.

    Each of the ``R`` resolutions becomes a row of ``B`` images, so reading top to
    bottom sweeps the step count while each column shares the same source noise.

    :param samples: Samples of shape ``(R, B, C, H, W)`` in ``[0, 1]``.
    :return: A single image grid of shape ``(C, H', W')`` in ``[0, 1]``.
    """
    rows, batch = samples.shape[0], samples.shape[1]
    flat = samples.reshape(rows * batch, *samples.shape[2:])
    # nrow = batch -> each row of the grid is one resolution's B samples.
    return make_grid(flat.clamp(0, 1), nrow=batch, value_range=(0, 1))


def _bf16_inference(device: torch.device):
    """Use bf16 autocast for callback inference on CUDA, preserving CPU test behavior."""
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def _strong_convergence_precision_context(trainer: Trainer, module: LightningModule):
    """Return the precision-plugin context for callback-owned model evaluation."""
    device = getattr(module, 'device', None)
    if device is None or device.type == 'cpu':
        return nullcontext()

    plugin = getattr(trainer, 'precision_plugin', None)
    precision = getattr(trainer, 'precision', getattr(plugin, 'precision', None))
    if not isinstance(precision, str) or not precision.endswith('-mixed'):
        return nullcontext()

    autocast_context = getattr(plugin, 'autocast_context', None)
    if callable(autocast_context):
        return autocast_context()
    if precision == 'bf16-mixed':
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _image_shape(trainer: Trainer) -> tuple[int, int, int]:
    """Read the network-state shape without consuming a validation batch.

    Pixel datamodules expose ``image_size`` and are RGB. Latent ImageNet has no
    pixel size; strong-error and other state-space diagnostics stay in the
    SD-VAE latent ``(4, 32, 32)`` rather than decoding to pixels. Sana
    adapters expose their DC-AE latent geometry on ``net.base``
    (``in_channels``/``sample_size``), which takes precedence; DMF backbones
    use ``input_size`` instead and keep the legacy fallback.
    """
    hparams = getattr(trainer.datamodule, 'hparams', None)
    image_size = getattr(hparams, 'image_size', None)
    if image_size is not None:
        return 3, int(image_size), int(image_size)
    base = getattr(getattr(trainer, 'model', None), 'net', None)
    base = getattr(base, 'base', None)
    sample_size = getattr(base, 'sample_size', None)
    in_channels = getattr(base, 'in_channels', None)
    if sample_size is not None and in_channels is not None:
        return int(in_channels), int(sample_size), int(sample_size)
    return 4, 32, 32


_DEFINED_STEP_METRICS: dict[int, tuple[Any, set[str]]] = {}


def _register_step_metrics(experiment: Any, keys: Any) -> None:
    """Point the wandb panels for ``keys`` at the optimizer-step axis.

    Media and scalar panels key their x-axis off wandb's internal ``_step``
    counter by default -- one per ``experiment.log`` commit, not per optimizer
    step -- so event images (sample grids, heatmaps) display as sequential
    commit indices (2, 22, 44, ...) instead of the true optimizer step.
    ``define_metric(key, step_metric='trainer/global_step')`` re-keys each
    panel onto the optimizer-step axis every commit already carries. The
    registration is stored in the run config, so it also re-labels already
    committed history. Idempotent per (experiment, key); the set is cached
    per logger process (a reference to the experiment is held so the cache
    key cannot be recycled).
    """
    cached = _DEFINED_STEP_METRICS.get(id(experiment))
    registered = set() if cached is None else cached[1]
    for key in keys:
        if key in ('trainer/global_step', 'epoch') or key in registered:
            continue
        experiment.define_metric(key, step_metric='trainer/global_step')
        registered.add(key)
    _DEFINED_STEP_METRICS[id(experiment)] = (experiment, registered)


def _wandb_commit(trainer: Trainer, payload: dict[str, Any]) -> None:
    """Commit event metrics to wandb exactly once, at their true optimizer step.

    Event metrics (FID scores, strong-error series, grids, heatmaps) must not
    pass through ``pl_module.log``: Lightning files callback-issued logs as
    step metrics, so the value rides the result pool and is re-emitted at
    every logger flush until the next event overwrites it -- rendering as a
    staircase (flat between events) on the wandb x-axis.

    The commit passes an explicit ``step=trainer.global_step``. Media rows
    (sample grids, heatmaps) are stored, named, and panel-labeled by the row
    step; without it, wandb's auto-increment counter -- one per commit, i.e.
    ~21 rows per 1000 optimizer steps -- labels the grids 0, 20, 41, ... The
    explicit step is safe here because Lightning's WandbLogger flushes never
    carry one (they embed ``trainer/global_step`` as a payload metric), so an
    event row never merges into a flush row; the counter resyncs to the true
    step at every event and drifts at most one 50-step flush window between
    events. Two events at the same optimizer step (e.g. a grid and a FID
    sweep) merge into that single row, which is the correct behavior.

    Scalar event keys additionally get ``define_metric`` registered
    (:func:`_register_step_metrics`) so their charts resolve the x-axis from
    the ``trainer/global_step`` values exactly.
    """
    logger = next((entry for entry in trainer.loggers if isinstance(entry, WandbLogger)), None)
    if logger is None:
        return
    import wandb  # local import: wandb is only needed when a WandbLogger exists.

    body = dict(payload)
    body.setdefault('trainer/global_step', trainer.global_step)
    body.setdefault('epoch', trainer.current_epoch)
    experiment = logger.experiment
    try:
        _register_step_metrics(experiment, payload.keys())
    except wandb.errors.Error as exc:
        log.warning('wandb define_metric registration failed: %s', exc)
    try:
        experiment.log(body, step=trainer.global_step)
    except wandb.errors.Error as exc:
        log.warning('wandb event-metric log failed: %s', exc)


class SampleGridCallback(Callback):
    """Log a shared-noise, multi-resolution sample grid during training.

    Init grids are logged from ``on_train_start`` at optimizer step 0. Later
    grids follow ``every_n_epochs`` / ``every_n_steps`` / ``initial_steps``.
    """

    def __init__(
        self,
        max_steps: int = 16,
        num_samples: int = 8,
        degree: int = 1,
        every_n_epochs: int | None = 5,
        every_n_steps: int | None = None,
        initial_steps: tuple[int, ...] = (),
        seed: int | None = 0,
        log_key: str = 'val/sample_grid',
        sampler: str = 'euler',
        log_both_solvers: bool = False,
        log_independent_brownian_grid: bool = False,
        num_classes: int | None = None,
        label_seed: int = 0,
        cfg_scale: float | None = None,
        log_on_sanity: bool = False,
        use_ema: bool = True,
    ) -> None:
        """Initialize the sampling callback.

        :param max_steps: Finest (largest) step count; must be a power of two. The
            grid shows rows for ``max_steps, max_steps/2, ..., 1``.
        :param num_samples: Number of images per row (shared across resolutions).
        :param degree: Polynomial Brownian degree ``N`` per flow-map step.
        :param every_n_epochs: Render every this many epochs. Init logging is
            independent of this cadence and always runs at train start.
        :param seed: Optional RNG seed so the latents and Brownian motion are
            reproducible across epochs (set ``None`` for fresh noise each time).
        :param log_key: Logger key under which the grid image is stored.
        :param sampler: FM sampling scheme — ``"euler"`` (deterministic ODE) or
            ``"em"`` (stochastic Euler-Maruyama SDE). Ignored for SSFM nets.
        :param log_both_solvers: When ``True`` and the net is a plain FM model,
            render *two* grids — one with the ODE (Euler) solver and one with the
            SDE (Euler-Maruyama) solver — using the same seeded latents, and log
            them under ``{log_key}_ode`` and ``{log_key}_sde`` respectively. This
            makes the ``sampler`` field irrelevant.
        :param log_independent_brownian_grid: Also log a step-count grid with
            shared latents but a fresh independent Brownian path for every row.
        """
        super().__init__()
        self.max_steps = max_steps
        self.num_samples = num_samples
        self.degree = degree
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError('every_n_epochs must be positive when configured')
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError('every_n_steps must be positive when configured')
        if any(step < 1 for step in initial_steps):
            raise ValueError('initial_steps must contain positive optimizer steps')
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.initial_steps = tuple(initial_steps)
        self._last_step = -1
        self.seed = seed
        self.log_key = log_key
        self.sampler = sampler
        self.log_both_solvers = log_both_solvers
        self.log_independent_brownian_grid = log_independent_brownian_grid
        self.num_classes = num_classes
        self.label_seed = label_seed
        self.cfg_scale = cfg_scale
        self.log_on_sanity = log_on_sanity
        self.use_ema = use_ema

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        """Return the attached ``WandbLogger``, if any."""
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the untrained grid at step 0, before the first optimizer step."""
        self._log_grid(trainer, pl_module)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Render and log the multi-resolution sample grid (rank zero only)."""
        if self.every_n_steps is not None:
            return
        if trainer.sanity_checking and not self.log_on_sanity:
            return
        if not trainer.sanity_checking and self.log_on_sanity and trainer.current_epoch == 0:
            return
        if self.every_n_epochs is None or (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        self._log_grid(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Log once when the completed optimizer step matches the configured cadence."""
        del outputs, batch, batch_idx
        step = trainer.global_step
        periodic = self.every_n_steps is not None and step % self.every_n_steps == 0
        if (
            step < 1
            or step == self._last_step
            or (step not in self.initial_steps and not periodic)
        ):
            return
        self._last_step = step
        self._log_grid(trainer, pl_module)

    def state_dict(self) -> dict[str, int]:
        """Persist the last evaluated optimizer step for resume-safe cadence."""
        return {'last_step': self._last_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore resume-safe cadence state."""
        self._last_step = int(state_dict.get('last_step', -1))

    @rank_zero_only
    def _log_grid(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Render and log the multi-resolution sample grid."""
        if self._wandb_logger(trainer) is None:
            return

        with _bf16_inference(pl_module.device):
            grids = self._render_grid(pl_module)
        import wandb  # local import: wandb is only needed when a WandbLogger exists.

        rows = ', '.join(str(2**i) for i in reversed(range(self.max_steps.bit_length())))
        caption = f'optimizer step {trainer.global_step} | step counts (top->bottom): {rows}'

        images: dict[str, wandb.Image] = {}
        for suffix, grid in grids.items():
            image = grid.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255)
            image = image.to('cpu', torch.uint8).numpy()
            grid_caption = caption
            if suffix == '_stochastic':
                grid_caption += ' — fresh independent Brownian path per row'
            images[f'{self.log_key}{suffix}'] = wandb.Image(image, caption=grid_caption)
        _wandb_commit(trainer, images)

    def _row_generator(self, device: torch.device, row: int) -> torch.Generator | None:
        """Return a reproducible generator unique to one stochastic-grid row."""
        if self.seed is None:
            return None
        return torch.Generator(device=device).manual_seed(self.seed + 1 + row)

    def _independent_samples(
        self, pl_module: LightningModule, latents: Tensor, step_counts: list[int]
    ) -> Tensor | None:
        """Sample each row from an independent Brownian realization."""
        net, device = pl_module.net, pl_module.device
        if _is_ssfm(net):
            return torch.stack(
                [
                    flow_map_sampler(
                        net,
                        latents,
                        num_steps=steps,
                        degree=self.degree,
                        generator=self._row_generator(device, row),
                    )
                    for row, steps in enumerate(step_counts)
                ]
            )
        if self.sampler != 'em' and not self.log_both_solvers:
            return None
        return torch.stack(
            [
                fm_em_sampler(
                    net,
                    pl_module.schedule,
                    latents,
                    prediction=pl_module.prediction,
                    num_steps=steps,
                    generator=self._row_generator(device, row),
                )
                for row, steps in enumerate(step_counts)
            ]
        )

    def _render_grid(self, pl_module: LightningModule) -> dict[str, Tensor]:
        """Sample the shared-noise multi-resolution grid for ``pl_module``.

        When ``self.log_both_solvers`` is ``True`` and the net is a plain FM model,
        returns a ``(ode_grid, sde_grid)`` tuple using the same seeded latents so
        the two solvers are directly comparable.
        """
        net = pl_module.net
        if self.use_ema and getattr(pl_module, 'ema', None) is not None:
            net = pl_module.ema.get_model(0)
        device = pl_module.device
        shape = (self.num_samples, *_image_shape(pl_module.trainer))

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)
        # The grid start is standardized noise ``X_0 ~ N(0, I)``; the samplers map
        # it onto the schedule's own noise-end marginal (Sana's TrigFlow prior
        # is sigma_data-scaled), so no scaling happens here.
        latents = randn(shape, device=device, generator=generator)

        step_counts = [2**i for i in reversed(range(self.max_steps.bit_length()))]
        was_training = net.training
        net.eval()
        grids: dict[str, Tensor]
        if _is_ei_doss(pl_module):
            zeta = _training_zeta(pl_module)
            samples = torch.stack(
                [
                    ei_doss_sussmann_sampler(
                        net,
                        pl_module.schedule,
                        latents,
                        num_steps=s,
                        degree=pl_module.loss.degree,
                        t0=pl_module.loss.t0,
                        zeta=zeta,
                        generator=generator,
                    )
                    for s in step_counts
                ],
                dim=0,
            )
            grids = {'': _resolution_grid(to_uint8_images(samples))}
        elif _is_ssfm(net):
            samples, _ = multistep_flow_map_sampler(
                net,
                latents,
                max_steps=self.max_steps,
                degree=self.degree,
                generator=generator,
            )
            grids = {'': _resolution_grid(to_uint8_images(samples))}
        else:
            # Plain flow-matching: sweep step counts on the same latents.
            if self.log_both_solvers:
                # Render both ODE and SDE grids on shared latents.
                schedule = pl_module.schedule
                prediction = pl_module.prediction
                ode_samples = torch.stack(
                    [fm_euler_sampler(net, latents, num_steps=s) for s in step_counts],
                    dim=0,
                )
                # Separate generator for SDE Brownian increments so the ODE grid
                # is unaffected by the stochastic path's RNG draws.
                em_generator = (
                    torch.Generator(device=device).manual_seed(self.seed)
                    if self.seed is not None
                    else None
                )
                sde_samples = torch.stack(
                    [
                        fm_em_sampler(
                            net,
                            schedule,
                            latents,
                            prediction=prediction,
                            num_steps=s,
                            generator=em_generator,
                        )
                        for s in step_counts
                    ],
                    dim=0,
                )
                grids = {
                    '_ode': _resolution_grid(to_uint8_images(ode_samples)),
                    '_sde': _resolution_grid(to_uint8_images(sde_samples)),
                }
            elif self.sampler == 'em':
                schedule = pl_module.schedule
                prediction = pl_module.prediction
                samples = torch.stack(
                    [
                        fm_em_sampler(
                            net,
                            schedule,
                            latents,
                            prediction=prediction,
                            num_steps=s,
                            generator=generator,
                        )
                        for s in step_counts
                    ],
                    dim=0,
                )
                grids = {'': _resolution_grid(to_uint8_images(samples))}
            else:
                samples = torch.stack(
                    [fm_euler_sampler(net, latents, num_steps=s) for s in step_counts],
                    dim=0,
                )
                grids = {'': _resolution_grid(to_uint8_images(samples))}
        if self.log_independent_brownian_grid:
            independent = self._independent_samples(pl_module, latents, step_counts)
            if independent is not None:
                grids['_stochastic'] = _resolution_grid(to_uint8_images(independent))
        if was_training:
            net.train()

        return grids


class StrongConvergenceCallback(Callback):
    """Log live-network pathwise strong error against a fine EM reference."""

    def __init__(
        self,
        num_samples: int = 1024,
        batch_size: int = 64,
        em_steps: int = 100,
        step_counts: tuple[int, ...] = tuple(range(1, 17)),
        degree: int = 3,
        every_n_epochs: int | None = 10,
        every_n_steps: int | None = None,
        initial_steps: tuple[int, ...] | list[int] = (),
        seed: int = 0,
        plot_key: str = 'val/strong_convergence',
        num_classes: int | None = None,
        label_seed: int = 0,
        log_on_sanity: bool = False,
    ) -> None:
        super().__init__()
        if num_samples <= 0 or batch_size <= 0 or em_steps <= 0:
            raise ValueError('Strong-convergence sample and step counts must be positive.')
        if num_samples % batch_size:
            raise ValueError('num_samples must be divisible by batch_size.')
        if not step_counts or min(step_counts) < 1 or max(step_counts) > em_steps:
            raise ValueError('step_counts must lie in [1, em_steps].')
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError('every_n_epochs must be positive when configured.')
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError('every_n_steps must be positive when configured.')
        if any(step < 1 for step in initial_steps):
            raise ValueError('initial_steps must contain positive optimizer steps.')
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.em_steps = em_steps
        self.step_counts = list(step_counts)
        self.degree = degree
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self.initial_steps = tuple(initial_steps)
        self._last_step = -1
        self.seed = seed
        self.plot_key = plot_key
        self.num_classes = num_classes
        self.label_seed = label_seed
        self.log_on_sanity = log_on_sanity
        self._fixed_labels: Tensor | None = None

    def _labels(self, net: torch.nn.Module, device: torch.device) -> Tensor | None:
        num_classes = self.num_classes or getattr(net, 'num_classes', None)
        if num_classes is None:
            return None
        if self._fixed_labels is None or self._fixed_labels.shape[0] != self.num_samples:
            generator = torch.Generator().manual_seed(self.label_seed)
            self._fixed_labels = torch.randint(
                0, num_classes, (self.num_samples,), generator=generator
            )
        return self._fixed_labels.to(device)

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        """Return the attached W&B logger, if present."""
        return next(
            (logger for logger in trainer.loggers if isinstance(logger, WandbLogger)), None
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Log once when the completed optimizer step matches the configured cadence."""
        del outputs, batch, batch_idx
        step = trainer.global_step
        periodic = self.every_n_steps is not None and step % self.every_n_steps == 0
        if (
            step < 1
            or step == self._last_step
            or (step not in self.initial_steps and not periodic)
        ):
            return
        self._last_step = step
        self._log_errors(trainer, pl_module)

    def state_dict(self) -> dict[str, int]:
        """Persist the last evaluated optimizer step for resume-safe cadence."""
        return {'last_step': self._last_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore resume-safe cadence state."""
        self._last_step = int(state_dict.get('last_step', -1))

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate independent initial-condition/path pairs on the live net."""
        if self.every_n_steps is not None:
            return
        if trainer.sanity_checking and not self.log_on_sanity:
            return
        if self.every_n_epochs is None or trainer.current_epoch % self.every_n_epochs:
            return
        self._log_errors(trainer, pl_module)

    @rank_zero_only
    def _log_errors(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute pathwise strong error and log scalar series for W&B line charts."""
        net = pl_module.net
        schedule = pl_module.schedule
        ei_euler = _is_ei_euler(pl_module)
        ei_doss = _is_ei_doss(pl_module)
        if not ei_euler and not _is_ssfm(net):
            return
        shape = _image_shape(pl_module.trainer)
        # Polynomial clock math is not safe in bfloat16: reconstructing an
        # EI-clock leaf via Python float and re-applying ei_weighted_clock
        # undershoots leaf.s on the bf16 grid, and restrict() raises. Frozen
        # LoRA bases may be stored in bfloat16, so do not inherit parameter
        # dtype here.
        device, dtype = pl_module.device, torch.float32
        t0 = float(getattr(pl_module.loss, 't0', getattr(pl_module.loss, 'epsilon', 0.01)))
        # Annealed EI-Doss runs carry the ODE->SDE amplitude on the loss; the
        # rollout rides the SDE the network is currently trained on.
        zeta_at = getattr(pl_module.loss, 'zeta_at', None)
        zeta = float(zeta_at(pl_module.global_step)) if zeta_at is not None else 1.0
        generator = torch.Generator(device=device).manual_seed(self.seed)
        total = torch.zeros(len(self.step_counts), device=device, dtype=torch.float64)
        was_training = net.training
        net.eval()
        fixed_labels = self._labels(net, device)
        for batch_index in range(self.num_samples // self.batch_size):
            labels = None
            if fixed_labels is not None:
                start = batch_index * self.batch_size
                labels = fixed_labels[start : start + self.batch_size]
                net.set_labels(labels)
            latents = randn(
                (self.batch_size, *shape), device=device, dtype=dtype, generator=generator
            )
            # EI-Doss leaves live on the EI-weighted clock (their raw increments
            # are ``∫ α ν dW``); ei_doss_rollout_batch re-clocks them via
            # ei_weighted_clock before restricting, so plain physical-time leaves
            # undershoot the parent start and make restrict() raise.
            if ei_doss:
                realization = sample_piecewise_ei_clock(
                    schedule,
                    tuple(latents.shape),
                    self.degree,
                    self.em_steps,
                    t0,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                    zeta=zeta,
                )
            else:
                realization = sample_piecewise_brownian(
                    tuple(latents.shape),
                    self.degree,
                    self.em_steps,
                    t0,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            with _strong_convergence_precision_context(trainer, pl_module):
                if ei_doss:
                    reference, endpoints = ei_doss_rollout_batch(
                        net,
                        schedule,
                        latents,
                        realization,
                        self.step_counts,
                        fine_steps=self.em_steps,
                        zeta=zeta,
                    )
                    batch_error = torch.stack(
                        [
                            ((endpoints[steps] - reference) ** 2).mean()
                            for steps in self.step_counts
                        ]
                    )
                else:
                    batch_error = strong_error_batch(
                        net,
                        schedule,
                        latents,
                        realization,
                        self.step_counts,
                        ei_euler=ei_euler,
                        t0=t0,
                    )
            total += batch_error.double()
        if fixed_labels is not None:
            net.set_labels(None)
        errors = (total / (self.num_samples // self.batch_size)).float()
        if was_training:
            net.train()
        scalars = {
            f'strong_error/mse_n{steps}': float(value)
            for steps, value in zip(self.step_counts, errors, strict=True)
        }
        _wandb_commit(trainer, scalars)


class FIDCallback(Callback):
    """Compute multi-profile FID during validation against precomputed real stats.

    One FID score is logged per EMA profile (``fid/ema``) plus
    the live non-EMA model (``fid/non_ema``). The real reference statistics are
    loaded once from a precomputed file and reused across profiles; fake samples
    for every profile are drawn from the **same seeded source noise** so that the
    profiles are compared on an identical noise budget. Scores are logged both to
    Weights & Biases (when attached) and via ``LightningModule.log`` so they are
    available for checkpoint monitoring.
    """

    def __init__(
        self,
        ref_stats_path: str,
        num_samples: int = 50000,
        sampling_steps: int | list[int] | tuple[int, ...] = 16,
        degree: int = 1,
        batch_size: int = 64,
        feature: int = 2048,
        every_n_epochs: int | None = 10,
        every_n_steps: int | None = None,
        mini_sampling_steps: int | list[int] | tuple[int, ...] = 4,
        full_every_n_steps: int | None = None,
        checkpoint_dir: str | None = None,
        seed: int | None = 0,
        include_ema: bool = True,
        include_non_ema: bool = True,
        log_prefix: str = 'fid_torchvision',
        sampler: str = 'euler',
        num_classes: int | None = None,
        label_seed: int = 0,
        cfg_scale: float | None = None,
        log_on_sanity: bool = False,
        torch_fidelity_ref_stats_path: str | None = None,
        torch_fidelity_log_prefix: str = 'fid_torch_fidelity',
    ) -> None:
        """Initialize the FID callback.

        :param ref_stats_path: Path to the precomputed real-reference statistics
            file (``torch.save`` of the three ``real_features_*`` buffers). If the
            file is absent the callback no-ops with a warning.
        :param num_samples: Total fake samples per profile (the FID sample budget).
        :param sampling_steps: Flow-map step count, or a list of counts to score.
        :param degree: Polynomial Brownian degree ``N`` per flow-map step.
        :param batch_size: Per-iteration sampling batch size.
        :param feature: Inception feature dimensionality (must match the ref stats).
        :param every_n_epochs: Evaluate every this many epochs.
        :param every_n_steps: Evaluate every this many optimizer steps (the
            mini-sweep cadence when ``full_every_n_steps`` is set).
        :param mini_sampling_steps: Step counts scored at each
            ``every_n_steps`` event when ``full_every_n_steps`` is set
            (ignored otherwise).
        :param full_every_n_steps: When set, run the *full*
            ``sampling_steps`` sweep every this many optimizer steps. At a step
            on both grids the full sweep takes precedence (the mini sweep is
            skipped, since the full sweep already covers it). Each full sweep
            also logs ``fid_ema_gm`` — the geometric mean of the EMA FIDs over
            ``sampling_steps`` — which drives best-checkpoint selection.
        :param checkpoint_dir: When set, save ``current.ckpt`` (full trainer
            state) at every evaluation event and ``best.ckpt`` (weights only)
            whenever a full sweep improves the best ``fid_ema_gm``. The EMA
            shadow weights ride along via the module's ``on_save_checkpoint``.
        :param seed: Base RNG seed for the shared source noise (``None`` = fresh).
        :param include_non_ema: Also evaluate the live non-EMA model.
        :param log_prefix: Prefix for the logged score keys.
        :param sampler: FM sampling scheme — ``"euler"`` (deterministic ODE) or
            ``"em"`` (stochastic Euler-Maruyama SDE). Ignored for SSFM nets.
        """
        super().__init__()
        self.ref_stats_path = ref_stats_path
        self.num_samples = num_samples
        if isinstance(sampling_steps, int):
            step_counts = (sampling_steps,)
        else:
            step_counts = tuple(int(steps) for steps in sampling_steps)
        if not step_counts or any(steps < 1 for steps in step_counts):
            raise ValueError(
                f'sampling_steps must be a positive int or list; got {sampling_steps!r}.'
            )
        self.sampling_step_counts = step_counts
        self.sampling_steps = step_counts[0]
        if isinstance(mini_sampling_steps, int):
            mini_counts = (mini_sampling_steps,)
        else:
            mini_counts = tuple(int(steps) for steps in mini_sampling_steps)
        if not mini_counts or any(steps < 1 for steps in mini_counts):
            raise ValueError(
                f'mini_sampling_steps must be a positive int or list; got {mini_sampling_steps!r}.'
            )
        self.mini_step_counts = mini_counts
        if full_every_n_steps is not None and full_every_n_steps < 1:
            raise ValueError('full_every_n_steps must be positive when configured')
        if full_every_n_steps is not None and every_n_steps is None:
            raise ValueError('full_every_n_steps requires every_n_steps (the mini cadence).')
        self.full_every_n_steps = full_every_n_steps
        self.checkpoint_dir = checkpoint_dir
        self._best_gm: float | None = None
        self.degree = degree
        self.batch_size = batch_size
        self.feature = feature
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError('every_n_epochs must be positive when configured')
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError('every_n_steps must be positive when configured')
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self._last_step = -1
        self.sampler = sampler
        self.seed = seed
        self.include_ema = include_ema
        self.include_non_ema = include_non_ema
        self.log_prefix = log_prefix
        self.num_classes = num_classes
        self.label_seed = label_seed
        self.cfg_scale = cfg_scale
        self.log_on_sanity = log_on_sanity
        self.torch_fidelity_ref_stats_path = torch_fidelity_ref_stats_path
        self.torch_fidelity_log_prefix = torch_fidelity_log_prefix
        self._fid: HistoricalFIDComputer | None = None
        self._torch_fidelity_fid: FIDComputer | None = None

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        """Return the attached ``WandbLogger``, if any."""
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _sample_shape(self, trainer: Trainer) -> tuple[int, int, int]:
        """Return the network-space sample shape used for FID generation."""
        return _image_shape(trainer)

    def _get_fid(self, device: torch.device) -> HistoricalFIDComputer | None:
        """Return the primary torchvision FID backend with loaded real moments.

        :param device: Device to place the metric (and reference buffers) on.
        :return: The computer, or ``None`` if the reference file is missing.
        """
        if self._fid is not None:
            return self._fid
        if self.ref_stats_path is None:
            return None
        path = Path(self.ref_stats_path)
        if not path.is_file():
            log.warning('FID reference stats not found at %s; skipping FID.', path)
            return None
        archive = np.load(path, allow_pickle=False)
        protocol = str(archive['protocol'].item()) if 'protocol' in archive else None
        stats = {
            'mu': archive['mu'],
            'sigma': archive['sigma'],
            'n_samples': int(archive['n_samples']),
            'metadata': {'protocol': protocol},
        }
        fid = HistoricalFIDComputer(device=device)
        fid.load_real_stats(stats)
        self._fid = fid
        return fid

    def _get_torch_fidelity_fid(self, device: torch.device) -> FIDComputer | None:
        """Return the optional torch-fidelity backend with loaded real statistics."""
        if self.torch_fidelity_ref_stats_path is None:
            return None
        if self._torch_fidelity_fid is not None:
            return self._torch_fidelity_fid
        path = Path(self.torch_fidelity_ref_stats_path)
        if not path.is_file():
            log.warning('torch-fidelity FID reference stats not found at %s; skipping it.', path)
            return None
        stats = torch.load(path, map_location=device)
        fid = FIDComputer(feature=self.feature, device=device)
        fid.load_real_stats(stats)
        self._torch_fidelity_fid = fid
        return fid

    def _profiles(self, pl_module: LightningModule) -> Iterator[tuple[str, torch.nn.Module]]:
        """Yield ``(name, net)`` pairs for the EMA shadow and the non-EMA model.

        :param pl_module: The SSFM module exposing ``ema`` and ``net``.
        """
        ema = getattr(pl_module, 'ema', None)
        if self.include_ema and ema is not None:
            yield 'ema', ema.get_model(0)
        if self.include_non_ema:
            yield 'non_ema', pl_module.net

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
        """Yield batches of generated images in ``[0, 1]`` totalling ``num_samples``.

        Source noise is seeded per profile (from ``self.seed``) so every profile
        sees the same latents and Brownian motion.

        :param net: The network to sample from.
        :param shape: Per-sample image shape ``(C, H, W)``.
        :param device: Device for sampling.
        :param schedule: Affine schedule (required for the ``"em"`` FM sampler).
        :param prediction: Network-output parameterization (for the ``"em"`` sampler).
        :param ei_doss: When ``True``, sample with
            :func:`~src.models.components.samplers.ei_doss_sussmann_sampler`.
        :param t0: Martingale clock origin for the EI-Doss sampler.
        """
        del doss, xpred_sde
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)

        was_training = net.training
        net.eval()
        remaining = self.num_samples
        offset = 0
        while remaining > 0:
            n = min(self.batch_size, remaining)
            latents = randn((n, *shape), device=device, generator=generator)
            with _bf16_inference(device):
                if ei_doss:
                    batch_labels = labels[offset : offset + n] if labels is not None else None
                    setter = getattr(net, 'set_labels', None)
                    if setter is not None:
                        setter(batch_labels)
                    try:
                        samples = ei_doss_sussmann_sampler(
                            net,
                            schedule,
                            latents,
                            num_steps=self.sampling_steps,
                            degree=self.degree,
                            t0=t0,
                            zeta=zeta,
                            generator=generator,
                        )
                    finally:
                        if setter is not None:
                            setter(None)
                elif _is_ssfm(net):
                    samples = flow_map_sampler(
                        net,
                        latents,
                        num_steps=self.sampling_steps,
                        degree=self.degree,
                        generator=generator,
                    )
                elif self.sampler == 'em':
                    samples = fm_em_sampler(
                        net,
                        schedule,
                        latents,
                        prediction=prediction,
                        num_steps=self.sampling_steps,
                        generator=generator,
                    )
                else:
                    samples = fm_euler_sampler(net, latents, num_steps=self.sampling_steps)
            remaining -= n
            offset += n
            yield samples.float()
        if was_training:
            net.train()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and log one FID per profile (rank zero only)."""
        if self.every_n_steps is not None:
            return
        if trainer.sanity_checking and not self.log_on_sanity:
            return
        if not trainer.sanity_checking and (
            trainer.current_epoch == 0
            or self.every_n_epochs is None
            or (trainer.current_epoch % self.every_n_epochs) != 0
        ):
            return
        self._evaluate(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Evaluate once at each configured completed optimizer step."""
        del outputs, batch, batch_idx
        step = trainer.global_step
        on_mini = self.every_n_steps is not None and step % self.every_n_steps == 0
        on_full = self.full_every_n_steps is not None and step % self.full_every_n_steps == 0
        if step < 1 or step == self._last_step or not (on_mini or on_full):
            return
        self._last_step = step
        # A full-sweep step takes precedence: the full sweep already covers the
        # mini step counts, so no duplicate mini evaluation runs.
        self._evaluate(
            trainer, pl_module, self.sampling_step_counts if on_full else self.mini_step_counts
        )

    def state_dict(self) -> dict[str, object]:
        """Persist the last evaluated step and best geometric mean for resume."""
        return {'last_step': self._last_step, 'best_gm': self._best_gm}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore resume-safe cadence and best-checkpoint state."""
        self._last_step = int(state_dict.get('last_step', -1))
        best = state_dict.get('best_gm')
        self._best_gm = None if best is None else float(best)

    @rank_zero_only
    def _evaluate(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step_counts: tuple[int, ...] | None = None,
    ) -> None:
        """Compute and log one FID per profile and (optionally) save checkpoints.

        :param step_counts: Step counts to sweep; defaults to the full
            ``sampling_step_counts`` (epoch-cadence path). The step-cadence path
            passes either the full sweep or the mini step counts.
        """
        device = pl_module.device
        fid = self._get_fid(device)
        torch_fidelity_fid = self._get_torch_fidelity_fid(device)
        if fid is None and torch_fidelity_fid is None:
            return
        if step_counts is None:
            step_counts = self.sampling_step_counts
        is_full_sweep = len(step_counts) == len(self.sampling_step_counts) and (
            tuple(step_counts) == self.sampling_step_counts
        )

        shape = self._sample_shape(trainer)

        scores: dict[str, float] = {}
        schedule = getattr(pl_module, 'schedule', None)
        prediction = getattr(pl_module, 'prediction', 'velocity')
        ei_doss = _is_ei_doss(pl_module)
        t0 = pl_module.loss.t0 if ei_doss else 0.01
        labels = None
        for name, net in self._profiles(pl_module):
            for steps in step_counts:
                self.sampling_steps = steps
                # Two-tier mode always disambiguates with the steps suffix so a
                # mini 4-step score lands on the same key as the full sweep's
                # 4-step column instead of a bare profile key.
                use_suffix = self.full_every_n_steps is not None or len(step_counts) > 1
                key = f'{name}/steps_{steps}' if use_suffix else name
                if fid is not None:
                    fid.reset_fake()
                if torch_fidelity_fid is not None:
                    torch_fidelity_fid.reset_fake()
                for batch in self._sample_fake(
                    net,
                    shape,
                    device,
                    schedule,
                    prediction,
                    ei_doss=ei_doss,
                    t0=t0,
                    labels=labels,
                    zeta=_training_zeta(pl_module),
                ):
                    if fid is not None:
                        fid.update_fake(batch)
                    if torch_fidelity_fid is not None:
                        torch_fidelity_fid.update_fake(to_fid_uint8(batch))
                if fid is not None:
                    scores[f'{self.log_prefix}/{key}'] = fid.compute()
                if torch_fidelity_fid is not None:
                    scores[f'{self.torch_fidelity_log_prefix}/{key}'] = (
                        torch_fidelity_fid.compute()
                    )

        geometric_means = self._log_geometric_means(trainer, pl_module, scores, is_full_sweep)

        payload: dict[str, Any] = dict(scores)
        payload.update(geometric_means)
        _wandb_commit(trainer, payload)

        # Checkpoint writes (two 4.5 GB files) only ride the full sweep: the
        # mini events exist for cheap live diagnostics, and best.ckpt needs
        # the full-sweep geometric mean anyway.
        if self.checkpoint_dir is not None and is_full_sweep:
            self._save_checkpoints(trainer, geometric_means)

    def _log_geometric_means(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        scores: dict[str, float],
        is_full_sweep: bool,
    ) -> dict[str, float]:
        """Return the per-EMA-profile geometric mean of FID over the full sweep.

        ``fid_ema_gm`` is the best-checkpoint criterion: the geometric mean of
        the EMA FID over every step count in the full sweep. Only full sweeps
        produce it (a mini sweep sees too few step counts for a comparable
        mean). The caller commits the means to wandb alongside the sweep scores.
        """
        del trainer
        if not is_full_sweep or not self.include_ema:
            return {}
        if getattr(pl_module, 'ema', None) is None:
            return {}
        means: dict[str, float] = {}
        values = []
        for prefix in (
            self.torch_fidelity_log_prefix,
            self.log_prefix,
        ):
            values = [
                scores[f'{prefix}/ema/steps_{steps}']
                for steps in self.sampling_step_counts
                if f'{prefix}/ema/steps_{steps}' in scores
            ]
            if values:
                break
        if len(values) != len(self.sampling_step_counts):
            return {}
        means['fid_ema_gm'] = float(np.exp(np.mean(np.log(values))))
        return means

    def _save_checkpoints(self, trainer: Trainer, geometric_means: dict[str, float]) -> None:
        """Save ``current.ckpt`` every full sweep and ``best.ckpt`` on GM improvement.

        ``current.ckpt`` is the full trainer state (optimizer, schedulers,
        callback positions, EMA weights) at the most recent full-sweep step;
        it is the resume point. ``best.ckpt`` is weights-only (net + EMA) and
        is rewritten only when a *full* sweep's EMA geometric mean improves on
        the best ever seen. Both overwrite in place, capping the run at two
        files. Mini sweeps are diagnostic-only and never write checkpoints.

        The best GM is updated *before* ``current.ckpt`` is written: the
        trainer checkpoint serializes this callback's :meth:`state_dict`, so a
        run resumed from ``current.ckpt`` must carry the best GM that includes
        this sweep (otherwise every resume restarts best-selection from
        ``None`` and the next sweep unconditionally overwrites ``best.ckpt``).
        """
        directory = Path(self.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        improved = False
        if geometric_means:
            candidate = min(geometric_means.values())
            improved = self._best_gm is None or candidate < self._best_gm
            if improved:
                self._best_gm = candidate
        trainer.save_checkpoint(str(directory / 'current.ckpt'))
        if improved:
            trainer.save_checkpoint(str(directory / 'best.ckpt'), weights_only=True)


BIN_LABELS: tuple[str, ...] = (
    '[0.0,0.1]',
    '[0.1,0.2]',
    '[0.2,0.3]',
    '[0.3,0.4]',
    '[0.4,0.5]',
    '[0.5,0.6]',
    '[0.6,0.7]',
    '[0.7,0.8]',
    '[0.8,0.9]',
    '[0.9,1.0]',
)


def append_column(history: Tensor | None, column: Tensor) -> Tensor:
    """Append a per-bin *column* to a ``(num_bins, num_steps)`` history matrix.

    Pure: returns a new tensor and never mutates ``history``. The first call
    (``history is None``) seeds the matrix from ``column``.

    :param history: Existing history of shape ``(num_bins, num_steps)`` on CPU,
        or ``None`` to start a fresh history.
    :param column: New per-bin vector of shape ``(num_bins,)``.
    :return: History of shape ``(num_bins, num_steps + 1)``.
    """
    col = column.detach().reshape(-1, 1).cpu()
    if history is None:
        return col
    return torch.cat([history, col], dim=1)


def render_heatmap(matrix: Tensor, steps: list[int], title: str) -> object:
    """Render a ``(num_bins, num_steps)`` matrix as a labelled wandb heatmap image.

    x-axis is the training step, y-axis is the time bin (bottom = earliest bin),
    and colour is the diagnostic value. Signed uncertainty diagnostics use a
    linear colour scale; non-negative loss diagnostics use a logarithmic scale.
    Uses the non-interactive ``Agg`` backend so it is safe on headless nodes.

    :param matrix: History matrix of shape ``(num_bins, num_steps)``.
    :param steps: Training-step index for each column (length ``num_steps``).
    :param title: Figure title / colourbar label.
    :return: A :class:`wandb.Image` wrapping the rendered figure.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    import wandb

    data = matrix.numpy()
    is_uncertainty = title.startswith('uncertainty_mlp')
    if is_uncertainty:
        plot_data = np.ma.masked_where(~np.isfinite(data), data)
        default_bounds = (-1.0, 1.0)
    else:
        plot_data = np.ma.masked_where(~np.isfinite(data) | (data <= 0), data)
        default_bounds = (1.0, 10.0)
    if plot_data.count():
        vmin = float(plot_data.min())
        vmax = float(plot_data.max())
        if vmin == vmax:
            padding = max(abs(vmin) * 0.1, 1e-6)
            vmin = vmin - padding if is_uncertainty else vmin
            vmax += padding
    else:
        vmin, vmax = default_bounds
    norm = Normalize(vmin=vmin, vmax=vmax) if is_uncertainty else LogNorm(vmin=vmin, vmax=vmax)
    num_bins = data.shape[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.imshow(
        plot_data,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        extent=(steps[0], steps[-1], 0, num_bins),
        norm=norm,
    )
    ax.set_yticks([i + 0.5 for i in range(num_bins)])
    ax.set_yticklabels(BIN_LABELS[:num_bins])
    ax.set_xlabel('training step')
    ax.set_ylabel('time bin')
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=title)
    fig.tight_layout()
    image = wandb.Image(fig)
    plt.close(fig)
    return image


class LossFieldHeatmap(Callback):
    """Log per-bin loss / uncertainty diagnostics as step-vs-bin heatmaps.

    Reads :attr:`~src.models.ssfm_module.SSFMLitModule.last_bin_diagnostics`
    after each training batch, accumulates each configured ``*_by_{axis}``
    vector into a ``(num_bins, num_steps)`` history, and periodically renders
    each history as a matplotlib heatmap logged to Weights & Biases. The x-axis
    is the training step and the y-axis is the time bin, so a single image shows
    how the loss / weight profile over time evolves throughout training.

    Only active when a :class:`WandbLogger` is attached; otherwise a no-op.
    """

    def __init__(
        self,
        keys: tuple[str, ...] = (
            'diagonal_loss_by_s',
            'diagonal_loss_by_h',
            'diagonal_loss_by_t',
            'distillation_loss_by_s',
            'distillation_loss_by_h',
            'psd_loss_by_s',
            'psd_loss_by_h',
            'psd_loss_by_t',
            'uncertainty_mlp_by_s',
            'uncertainty_mlp_by_h',
        ),
        every_n_steps: int = 50,
        log_prefix: str = 'train/heatmap',
    ) -> None:
        """Initialize a ``LossFieldHeatmap``.

        :param keys: Diagnostic bin-vector names to track. Keys absent from a
            given run's diagnostics (e.g. ``*_by_t`` under ``em_step``) are
            skipped silently.
        :param every_n_steps: Render + log the heatmaps every this many steps.
        :param log_prefix: Log-key prefix; each key logs to ``{prefix}/{key}``.
        """
        super().__init__()
        self.keys = keys
        self.every_n_steps = max(1, every_n_steps)
        self.log_prefix = log_prefix
        self._history: dict[str, Tensor] = {}
        self._steps: list[int] = []

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        """Return the attached ``WandbLogger``, if any."""
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Accumulate this step's bin vectors and periodically log heatmaps.

        :param trainer: Active Lightning trainer.
        :param pl_module: SSFM module exposing ``last_bin_diagnostics``.
        :param outputs: Step outputs (unused).
        :param batch: Current batch (unused).
        :param batch_idx: Batch index within the epoch (unused).
        """
        diagnostics = getattr(pl_module, 'last_bin_diagnostics', None)
        if not diagnostics:
            return
        step = trainer.global_step
        self._steps.append(step)
        for key in self.keys:
            vector = diagnostics.get(key)
            if vector is None:
                continue
            self._history[key] = append_column(self._history.get(key), vector)

        if (batch_idx + 1) % self.every_n_steps != 0:
            return
        if self._wandb_logger(trainer) is None or not self._history:
            return
        self._log_heatmaps(trainer)

    def _log_heatmaps(self, trainer: Trainer) -> None:
        """Render and log all tracked histories as heatmap images.

        :param trainer: Active trainer (used for the wandb x-axis keys).
        """
        images = {
            f'{self.log_prefix}/{key}': render_heatmap(matrix, self._steps, key)
            for key, matrix in self._history.items()
        }
        _wandb_commit(trainer, images)
