"""Lightning callbacks for the augmented FKC toy experiments.

:class:`FKCSamplePlotCallback` renders the exact composed target against the
learned map's weighted samples at several NFE, plus centered log-weight
histograms and ESS, and logs the figure to WandB (and the run directory) during
validation. It is intentionally self-contained (matplotlib Agg + the module's
own :meth:`sample`) so a run can be monitored without a full eval pass.

Panels are diagnostics, not silent pictures:

* every sample panel is annotated with its finite fraction, the fraction of
  finite points inside the target window, and the finite coordinate mean and
  range, and its view is expanded to include every finite generated point, so
  a diverged or out-of-support cloud is visible instead of silently clipped;
* nonfinite states or weights never crash the renderer: affected statistics
  and the ESS label degrade to ``n/a`` (with a finite-entry count);
* the few-step map panels do not resample, so the terminal log weights are
  cumulative: they report the Kish ESS and its fraction of ``n`` in scientific
  notation (the fraction minimum is ``1 / n``, never exactly zero);
* the learned-drift SDE rollout (second figure) resamples after every step, so
  the returned log weights cover only the final step: its panels show spatial
  statistics, the resampling count, and a reset disclaimer instead of an ESS
  summary. They are a monitoring view of the rollout, not evidence that the
  learned fields are correct;
* every learned cloud is also shown BEFORE and AFTER terminal weighting: the
  after view is a systematic resampling of the *same* particles with their
  *existing* log weights (no reroll, no analytic reweighting), so equal weights
  leave the cloud unchanged and invalid weights/points display ``unavailable``
  instead of a fabricated picture;
* the base expert ``q_1`` is plotted next to the tempered target ``pi_1`` as a
  reference only — it is NOT the law of the unweighted proposal, a distinction
  stated on the panel itself;
* a third figure shows no-resampling analytic and learned SDE rollouts before
  and after terminal weighting (same particles), separately from the
  every-step SMC panels of the second figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from lightning import Callback, LightningModule, Trainer  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from torch import Tensor  # noqa: E402

from src.models.components.fkc.sampling import (  # noqa: E402
    FKCSamples,
    LearnedFKCFields,
    numerical_fkc,
)
from src.models.components.fkc.weights import (  # noqa: E402
    effective_sample_size,
    systematic_resample,
)

log = logging.getLogger(__name__)

# Shown verbatim on every SDE rollout panel: with systematic resampling after
# each step the returned log weights only describe the final step, so a
# terminal ESS is not a quality diagnostic for the rollout.
_RESET_DISCLAIMER = 'weights reset after every step; final ESS not diagnostic'

# Shown verbatim on the q1 reference panel: q1 is the first expert's marginal,
# not the law of the unweighted tempered proposal, so it must never be read as
# "what the sampler would produce without weighting".
_Q1_DISCLAIMER = 'Base q1 is a reference, not the unweighted proposal law'


def _finite_rows(cloud: Tensor) -> Tensor:
    """Return the rows of ``cloud`` whose coordinates are all finite, on CPU.

    :param cloud: Particle cloud of shape ``(n, 2)``.
    :return: Finite rows ``(k, 2)`` with ``k <= n`` (possibly empty).
    :raises ValueError: If ``cloud`` is not a two-column particle cloud.
    """
    if cloud.ndim != 2 or cloud.shape[-1] != 2:
        raise ValueError(f'cloud must be of shape (n, 2); got {tuple(cloud.shape)}.')
    cloud_cpu = cloud.detach().cpu()
    return cloud_cpu[torch.isfinite(cloud_cpu).all(dim=-1)]


def _target_bounds(target: Tensor) -> tuple[Tensor, Tensor] | None:
    """Return the padded per-axis window of the finite target rows.

    :param target: Target cloud ``(n, 2)``.
    :return: ``(lo, hi)`` of shape ``(2,)``, or ``None`` when the target has no
        finite rows (panels then fall back to the generated data).
    """
    finite = _finite_rows(target)
    if finite.shape[0] == 0:
        return None
    return finite.min(0).values - 0.1, finite.max(0).values + 0.1


def _panel_annotation(cloud: Tensor, bounds: tuple[Tensor, Tensor] | None) -> str:
    """Return a compact diagnostics line for a sample panel.

    Reports the finite fraction, the fraction of finite rows inside the target
    window, and the finite coordinate mean and largest span, so divergence,
    clipping, or NaN states are visible in the panel instead of silently
    hidden. Undefined statistics (no finite points, or no target window) are
    reported as ``n/a`` rather than raised.

    :param cloud: Particle cloud ``(n, 2)``.
    :param bounds: Target window ``(lo, hi)`` from :func:`_target_bounds`.
    :return: One-line annotation string.
    """
    total = int(cloud.shape[0])
    if total == 0:
        return 'n=0'
    finite = _finite_rows(cloud)
    n_finite = int(finite.shape[0])
    parts = [f'fin {n_finite / total:.1%}']
    if bounds is None or n_finite == 0:
        parts.append('in n/a')
    else:
        lo, hi = bounds
        inside = ((finite >= lo) & (finite <= hi)).all(dim=-1)
        parts.append(f'in {float(inside.float().mean()):.1%}')
    if n_finite == 0:
        parts.append('mu n/a rng n/a')
    else:
        mean = finite.mean(0)
        span = (finite.max(0).values - finite.min(0).values).max()
        parts.append(f'mu=({float(mean[0]):.3g}, {float(mean[1]):.3g}) rng={float(span):.3g}')
    return ' '.join(parts)


def _ess_label(log_weights: Tensor) -> str:
    """Return the Kish ESS and its fraction of ``n`` in scientific notation.

    The absolute ESS lies in ``[1, n]`` and its fraction in ``[1 / n, 1]``;
    both are shown because the fraction minimum is ``1 / n``, never exactly
    zero. Any nonfinite log weight invalidates the whole self-normalized
    diagnostic, so the label degrades to ``ESS=n/a`` with a finite-entry count
    instead of a number recomputed from a masked subset.

    :param log_weights: Log weights ``(n,)``.
    :return: Label such as ``ESS=8.00e+00 (1.00e+00 of n=8)`` or ``ESS=n/a``.
    """
    log_weights = log_weights.detach()
    n = int(log_weights.numel())
    if not bool(torch.isfinite(log_weights).all()):
        n_finite = int(torch.isfinite(log_weights).sum())
        return f'ESS=n/a ({n_finite}/{n} finite)'
    ess = float(effective_sample_size(log_weights))
    return f'ESS={ess:.2e} ({ess / n:.2e} of n={n})'


def _terminal_resample(
    result: FKCSamples, generator: torch.Generator | None
) -> Tensor | None:
    """Return the cloud after one systematic resample by its existing weights.

    This is a pure display transform of the SAME particles: their accumulated
    log weights are used as-is, nothing is re-simulated and no new weights are
    computed from an analytic target. Equal (uniform) weights reproduce the
    input cloud exactly; nonfinite log weights invalidate the weighting, and
    ``None`` is returned so panels show ``unavailable`` instead of a
    fabricated picture.

    :param result: Terminal :class:`FKCSamples` from a sampler.
    :param generator: Optional RNG for the systematic-resampling offset.
    :return: Resampled cloud ``(n, 2)``, or ``None`` when the weights are
        nonfinite (or a finite cloud cannot be produced).
    """
    if result.x.numel() == 0:
        return None
    if not bool(torch.isfinite(result.log_weights).all() and torch.isfinite(result.x).all()):
        return None
    indices = systematic_resample(result.log_weights.detach().cpu(), generator=generator)
    return result.x.detach().cpu()[indices]


def _display_resample_generator(seed: int | None) -> torch.Generator | None:
    """Return a fresh CPU generator for display-only resampling.

    Display resampling must not perturb the sampling stream used for the
    baseline clouds, so it draws from its own local generator instead of the
    callback's shared one.

    :param seed: Seed for the local generator (``None`` disables determinism).
    :return: A fresh generator, or ``None`` when ``seed`` is ``None``.
    """
    if seed is None:
        return None
    return torch.Generator(device='cpu').manual_seed(seed)


def _sde_stats_text(
    name: str,
    result: FKCSamples,
    steps: int,
    bounds: tuple[Tensor, Tensor] | None,
) -> str:
    """Summarize a resampled SDE rollout without presenting a terminal ESS.

    With ``resample_every_step`` the returned log weights cover only the final
    step, so the panel shows spatial statistics and the resampling count plus
    an explicit reset disclaimer; it makes no claim about rollout quality.

    :param name: Drift label used in the summary heading.
    :param result: Rollout output from :func:`numerical_fkc`.
    :param steps: Number of integration steps that were configured.
    :param bounds: Target window ``(lo, hi)`` from :func:`_target_bounds`.
    :return: Multi-line summary text.
    """
    return '\n'.join(
        (
            f'{name} drift + FKC (every-step SMC)',
            f'steps={steps} resamples={result.resampling_count}',
            _panel_annotation(result.x, bounds),
            _RESET_DISCLAIMER,
        )
    )


def _view_limits(
    cloud: Tensor, bounds: tuple[Tensor, Tensor] | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``(x0, x1), (y0, y1)`` covering the target window and the cloud.

    Sample panels never silently clip generated states: the view is the union
    of the target window and every finite generated row, padded by 5% of the
    span. When the target has no finite window, the finite cloud (or a unit
    square) defines the view.

    :param cloud: Particle cloud ``(n, 2)`` to include in the view.
    :param bounds: Target window ``(lo, hi)`` from :func:`_target_bounds`.
    :return: ``((x0, x1), (y0, y1))`` axis limits.
    """
    if bounds is None:
        x0, y0, x1, y1 = -1.0, -1.0, 1.0, 1.0
    else:
        lo, hi = bounds
        x0, y0 = float(lo[0]), float(lo[1])
        x1, y1 = float(hi[0]), float(hi[1])
    finite = _finite_rows(cloud)
    if finite.shape[0] > 0:
        low = finite.min(0).values
        high = finite.max(0).values
        x0 = min(x0, float(low[0]))
        y0 = min(y0, float(low[1]))
        x1 = max(x1, float(high[0]))
        y1 = max(y1, float(high[1]))
    pad_x = max(0.05 * (x1 - x0), 1e-3)
    pad_y = max(0.05 * (y1 - y0), 1e-3)
    return (x0 - pad_x, x1 + pad_x), (y0 - pad_y, y1 + pad_y)


class FKCSamplePlotCallback(Callback):
    """Log target-vs-learned sample clouds with explicit panel diagnostics."""

    def __init__(
        self,
        num_samples: int = 2048,
        step_counts: tuple[int, ...] = (1, 4, 16, 64),
        sde_steps: int = 256,
        every_n_epochs: int = 10,
        seed: int | None = 0,
        log_key: str = 'val/fkc_samples',
        use_ema: bool = True,
    ) -> None:
        """Initialize the callback.

        :param num_samples: Particles per cloud (target and learned).
        :param step_counts: NFE sweep rendered as subplots.
        :param sde_steps: Fine Euler-Maruyama steps for the learned-drift FKC
            rollout sanity check (second figure).
        :param every_n_epochs: Render on validation epochs that are multiples of
            this value (aligned with ``check_val_every_n_epoch``).
        :param seed: RNG seed for reproducible panels.
        :param log_key: WandB image key / filename stem.
        :param use_ema: Sample with the EMA teacher when the module has one;
            set ``False`` to always render the live network.
        """
        super().__init__()
        if num_samples < 1:
            raise ValueError('num_samples must be positive.')
        if every_n_epochs < 1:
            raise ValueError('every_n_epochs must be positive.')
        self.num_samples = num_samples
        self.step_counts = tuple(step_counts)
        self.sde_steps = sde_steps
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.log_key = log_key
        self.use_ema = use_ema

    def _wandb_logger(self, trainer: Trainer) -> WandbLogger | None:
        """Return the attached ``WandbLogger``, if any."""
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def _generator(self, module: LightningModule) -> torch.Generator | None:
        """Return a seeded generator on the module's device, or ``None``."""
        if self.seed is None:
            return None
        device = 'cuda' if module.device.type == 'cuda' else 'cpu'
        return torch.Generator(device=device).manual_seed(self.seed)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Render and log the figure on the configured cadence.

        Skipped outside rank zero, during the Lightning sanity check, and on
        epochs that are not multiples of ``every_n_epochs``.
        """
        if not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        try:
            self._render(trainer, pl_module)
        except Exception as exc:  # pragma: no cover - monitoring must not kill training
            log.warning('FKC sample plot failed: %s', exc)
        try:
            self._log_strong_errors(trainer, pl_module)
        except Exception as exc:  # pragma: no cover - monitoring must not kill training
            log.warning('FKC strong-error eval failed: %s', exc)

    @torch.no_grad()
    def _render(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Sample, render, save the flow-map and learned-drift SDE figures.

        Each figure is closed in a ``finally`` block, so a rendering or logging
        failure cannot leak open figures into the pyplot registry.
        """
        problem = pl_module.problem
        weight_loss_scale = getattr(pl_module.loss, 'weight_loss_scale', 1.0)
        weight_note = (
            '\nw losses disabled: rate head untrained; learned weights are not an FKC check'
            if getattr(pl_module.loss, 'weight_loss_scale', 1.0) == 0.0
            else ''
        )
        generator = self._generator(pl_module)
        target = problem.sample_target(self.num_samples, 1.0, generator=generator)
        # Base q1: the FIRST expert's marginal (not the unweighted proposal law;
        # a reference cloud only). Drawn on its own generator so adding this
        # reference does not shift any downstream baseline sampling.
        q1 = problem.experts[0].sample(
            self.num_samples,
            1.0,
            generator=torch.Generator(device=target.device).manual_seed(self.seed or 0),
        )
        bounds = _target_bounds(torch.cat((target, q1)))
        # Display-only resampling draws from a separate local generator so the
        # extra per-panel resampling cannot perturb the baseline sampling
        # stream above.
        display_gen = _display_resample_generator(self.seed)
        out_dir = Path(trainer.default_root_dir or '.') / 'fkc_monitor'
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.log_key.split('/')[-1]

        def style(axis, title, annotation=None):
            axis.set_title(title, fontsize=9)
            axis.tick_params(labelsize=7)
            if annotation is not None:
                axis.text(
                    0.02,
                    0.02,
                    annotation,
                    transform=axis.transAxes,
                    fontsize=6,
                    va='bottom',
                    ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.85),
                )

        def scatter(axis, cloud, color):
            cloud_cpu = cloud.detach().cpu()
            axis.scatter(cloud_cpu[:, 0], cloud_cpu[:, 1], s=2, alpha=0.4, c=color)

        def view(axis, cloud):
            (x0, x1), (y0, y1) = _view_limits(cloud, bounds)
            axis.set_xlim(x0, x1)
            axis.set_ylim(y0, y1)
            axis.set_aspect('equal')

        # -- Figure 1: few-step flow map at several NFE (no resampling) -------
        samples = {
            steps: pl_module.sample(
                self.num_samples, steps=steps, use_ema=self.use_ema, generator=generator
            )
            for steps in self.step_counts
        }
        ncols = 1 + len(self.step_counts)
        figure, axes = plt.subplots(3, ncols, figsize=(3.0 * ncols, 9.4))
        try:
            scatter(axes[0, 0], target, '#9aa0a6')
            view(axes[0, 0], target)
            style(axes[0, 0], 'exact target pi_1')
            scatter(axes[1, 0], q1, '#843C0C')
            view(axes[1, 0], q1)
            style(axes[1, 0], 'base q1 (first expert)', _Q1_DISCLAIMER)
            axes[2, 0].axis('off')
            for col, steps in enumerate(self.step_counts):
                result = samples[steps]
                rate_note = ' (untrained rate)' if weight_loss_scale == 0.0 else ''
                axis = axes[0, 1 + col]
                scatter(axis, result.x, '#1a73e8')
                view(axis, result.x)
                style(
                    axis,
                    f'learned NFE={steps}{rate_note} (before weighting)',
                    _panel_annotation(result.x, bounds),
                )
                # After weighting: systematic resample of the SAME particles
                # with their EXISTING log weights (no reroll, no analytic
                # reweighting); identical views so before/after are comparable.
                weighted_cloud = _terminal_resample(result, display_gen)
                waxis = axes[1, 1 + col]
                if weighted_cloud is None:
                    waxis.text(
                        0.5,
                        0.5,
                        'after weighting unavailable\n(nonfinite log weights)',
                        transform=waxis.transAxes,
                        ha='center',
                        va='center',
                        fontsize=8,
                    )
                else:
                    scatter(waxis, weighted_cloud, '#188038')
                view(waxis, result.x)
                style(
                    waxis,
                    f'learned NFE={steps} after weighting',
                    _panel_annotation(weighted_cloud, bounds)
                    if weighted_cloud is not None
                    else None,
                )
                weight_axis = axes[2, 1 + col]
                centered = (result.log_weights - result.log_weights.mean()).detach().cpu()
                finite = centered[torch.isfinite(centered)]
                if finite.numel() > 0:
                    weight_axis.hist(finite, bins=40, color='#1a73e8', alpha=0.85)
                else:
                    weight_axis.text(
                        0.5,
                        0.5,
                        'no finite log weights',
                        transform=weight_axis.transAxes,
                        ha='center',
                        va='center',
                        fontsize=8,
                    )
                weight_axis.set_title(_ess_label(result.log_weights), fontsize=8)
                weight_axis.tick_params(labelsize=7)
            figure.suptitle(
                f'FKC toy2d flow map @ epoch {trainer.current_epoch}{weight_note}', fontsize=11
            )
            figure.tight_layout(rect=(0, 0, 1, 0.97))
            self._log_figure(
                trainer,
                figure,
                out_dir / f'{stem}_epoch{trainer.current_epoch:03d}.png',
                self.log_key,
            )
        finally:
            plt.close(figure)

        # -- Figure 2: fine SDE rollout with the LEARNED drift + FKC ----------
        # resample_every_step resets the log weights at every step, so the
        # terminal weights describe only the final step: no ESS/hist summary.
        try:
            schedule = getattr(pl_module.loss, 'schedule', pl_module.schedule)
            t0 = pl_module.loss.t0
            net = (
                pl_module.ema_net
                if self.use_ema and pl_module.hparams.ema_decay >= 0.0
                else pl_module.net
            )
            x0 = problem.sample_target(self.num_samples, t0, generator=generator)
            analytic = numerical_fkc(
                problem,
                x0.clone(),
                t0=t0,
                steps=self.sde_steps,
                resample_every_step=True,
                schedule=schedule,
                generator=generator,
            )
            learned = numerical_fkc(
                LearnedFKCFields(net, schedule),
                x0.clone(),
                t0=t0,
                steps=self.sde_steps,
                resample_every_step=True,
                schedule=schedule,
                generator=generator,
            )
            figure2, axes2 = plt.subplots(2, 3, figsize=(9.0, 6.4))
            try:
                for col, (title, cloud, color) in enumerate(
                    (
                        ('exact target', target, '#9aa0a6'),
                        (f'analytic drift + FKC ({self.sde_steps})', analytic.x, '#1a73e8'),
                        (f'LEARNED drift + FKC ({self.sde_steps})', learned.x, '#D96002'),
                    )
                ):
                    axis = axes2[0, col]
                    scatter(axis, cloud, color)
                    view(axis, cloud)
                    style(axis, title, None if col == 0 else _panel_annotation(cloud, bounds))
                axes2[1, 0].axis('off')
                for col, (name, result) in enumerate(
                    (('analytic', analytic), ('learned', learned))
                ):
                    stats_axis = axes2[1, 1 + col]
                    stats_axis.axis('off')
                    stats_axis.text(
                        0.02,
                        0.98,
                        _sde_stats_text(name, result, self.sde_steps, bounds),
                        transform=stats_axis.transAxes,
                        fontsize=8,
                        va='top',
                        ha='left',
                        family='monospace',
                    )
                figure2.suptitle(
                    f'FKC toy2d learned-drift SDE rollout @ epoch {trainer.current_epoch}{weight_note}',
                    fontsize=11,
                )
                figure2.tight_layout(rect=(0, 0, 1, 0.97))
                self._log_figure(
                    trainer,
                    figure2,
                    out_dir / f'{stem}_sde_epoch{trainer.current_epoch:03d}.png',
                    f'{self.log_key}_sde',
                )
            finally:
                plt.close(figure2)
        except Exception as exc:  # pragma: no cover - sanity panel must not kill training
            log.warning('FKC learned-drift SDE panel failed: %s', exc)

        # -- Figure 3: NO-resampling rollouts, before vs after terminal
        # weighting ---------------------------------------------------------
        # The every-step SMC panels above reset weights each step, so they
        # cannot be reused for a before/after comparison. Here the SAME
        # rollouts are run WITHOUT resampling and shown before and after ONE
        # terminal systematic resampling by their existing (cumulative) log
        # weights -- same particles, no reroll, no analytic reweighting.
        try:
            schedule = getattr(pl_module.loss, 'schedule', pl_module.schedule)
            t0 = pl_module.loss.t0
            net = (
                pl_module.ema_net
                if self.use_ema and pl_module.hparams.ema_decay >= 0.0
                else pl_module.net
            )
            x0 = problem.sample_target(self.num_samples, t0, generator=generator)
            analytic_free = numerical_fkc(
                problem,
                x0.clone(),
                t0=t0,
                steps=self.sde_steps,
                resample_every_step=False,
                schedule=schedule,
                generator=generator,
            )
            learned_free = numerical_fkc(
                LearnedFKCFields(net, schedule),
                x0.clone(),
                t0=t0,
                steps=self.sde_steps,
                resample_every_step=False,
                schedule=schedule,
                generator=generator,
            )
            figure3, axes3 = plt.subplots(2, 4, figsize=(12.0, 6.4))
            try:
                # Identical views per pair so before/after are comparable.
                for col, (name, result, color) in enumerate(
                    (
                        ('analytic', analytic_free, '#1a73e8'),
                        ('LEARNED', learned_free, '#D96002'),
                    )
                ):
                    rate_note = ' (untrained rate)' if name == 'LEARNED' and weight_loss_scale == 0.0 else ''
                    top = axes3[0, 2 + col]
                    scatter(top, result.x, color)
                    view(top, result.x)
                    style(
                        top,
                        f'{name} drift, no resampling{rate_note} (before)',
                        _panel_annotation(result.x, bounds),
                    )
                    weighted_cloud = _terminal_resample(result, display_gen)
                    bottom = axes3[1, 2 + col]
                    if weighted_cloud is None:
                        bottom.text(
                            0.5,
                            0.5,
                            'after weighting unavailable\n(nonfinite log weights)',
                            transform=bottom.transAxes,
                            ha='center',
                            va='center',
                            fontsize=8,
                        )
                    else:
                        scatter(bottom, weighted_cloud, '#188038')
                    view(bottom, result.x)
                    style(
                        bottom,
                        f'{name} drift, after terminal weighting',
                        _panel_annotation(weighted_cloud, bounds)
                        if weighted_cloud is not None
                        else None,
                    )
                for row in range(2):
                    (tx0, tx1), (ty0, ty1) = _view_limits(target, bounds)
                    for col in range(2):
                        axes3[row, col].set_xlim(tx0, tx1)
                        axes3[row, col].set_ylim(ty0, ty1)
                        axes3[row, col].set_aspect('equal')
                axes3[0, 0].scatter(
                    target.detach().cpu()[:, 0],
                    target.detach().cpu()[:, 1],
                    s=2,
                    alpha=0.4,
                    c='#9aa0a6',
                )
                axes3[0, 0].set_title('exact target pi_1', fontsize=9)
                scatter(axes3[1, 0], q1, '#843C0C')
                view(axes3[1, 0], q1)
                style(axes3[1, 0], 'base q1 (first expert)', _Q1_DISCLAIMER)
                axes3[0, 1].axis('off')
                axes3[1, 1].axis('off')
                for col, result in enumerate((analytic_free, learned_free)):
                    ess_axis = axes3[0, 2 + col]
                    ess_axis.text(
                        0.98,
                        0.02,
                        _ess_label(result.log_weights),
                        transform=ess_axis.transAxes,
                        fontsize=7,
                        va='bottom',
                        ha='right',
                    )
                figure3.suptitle(
                    f'FKC toy2d no-resampling rollouts: before vs after terminal weighting '
                    f'@ epoch {trainer.current_epoch}{weight_note}',
                    fontsize=11,
                )
                figure3.tight_layout(rect=(0, 0, 1, 0.97))
                self._log_figure(
                    trainer,
                    figure3,
                    out_dir / f'{stem}_terminal_weight_epoch{trainer.current_epoch:03d}.png',
                    f'{self.log_key}_terminal_weight',
                )
            finally:
                plt.close(figure3)
        except Exception as exc:  # pragma: no cover - sanity panel must not kill training
            log.warning('FKC terminal-weighting panel failed: %s', exc)

    @torch.no_grad()
    def _log_strong_errors(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log pathwise strong errors in X and W against one fine EM reference.

        All arms share ONE Brownian realization: the fine EM reference rides the
        fine increments, and the learned map rides their degree-N Wong-Zakai
        projection, so the errors are pathwise (strong), not distributional.
        """
        from src.models.components.fkc.evaluation import (
            centered_log_weight_rmse,
            spatial_rmse,
        )
        from src.models.components.fkc.sampling import (
            ChenBrownianGrid,
            _ei_step,
            learned_fkc,
        )

        problem = pl_module.problem
        loss = pl_module.loss
        schedule = getattr(loss, 'schedule', pl_module.schedule)
        t0 = loss.t0
        net = (
            pl_module.ema_net
            if self.use_ema and pl_module.hparams.ema_decay >= 0.0
            else pl_module.net
        )
        generator = self._generator(pl_module)
        x0 = problem.sample_target(self.num_samples, t0, generator=generator)
        # ONE realization: degree-N paths sampled on a 1024-step reference grid,
        # Chen-merged for every coarser learned step count. A 64-step EM
        # reference is NOT converged here (strong X ~0.14 vs 1024 steps).
        fine_steps = 1024
        if any(fine_steps % steps for steps in self.step_counts):
            raise ValueError(f'step counts {self.step_counts} must divide {fine_steps}.')
        grid = ChenBrownianGrid(
            schedule,
            tuple(x0.shape),
            fine_steps,
            loss.degree,
            t0,
            1.0,
            device=x0.device,
            dtype=x0.dtype,
            generator=generator,
        )
        # Reference: the TRUE analytic coupled FKC SDE on the finest grid, riding
        # the same exact fine increments the learned map sees merged.
        times = torch.linspace(t0, 1.0, fine_steps + 1, device=x0.device, dtype=x0.dtype)
        ref_x = x0.clone()
        ref_w = x0.new_zeros(x0.shape[0])
        for j, path in enumerate(grid.steps(fine_steps)):
            ref_x, ref_w = _ei_step(
                problem, ref_x, ref_w, times[j], times[j + 1], path.raw_increment(), schedule, t0
            )
        reference = FKCSamples(ref_x, ref_w, 0)
        metrics: dict[str, float] = {}
        for steps in self.step_counts:
            learned = learned_fkc(
                net,
                x0,
                t0=t0,
                steps=steps,
                degree=loss.degree,
                schedule=schedule,
                chen_grid=grid,
            )
            metrics[f'val/strong_x_nfe{steps}'] = float(spatial_rmse(learned.x, reference.x))
            metrics[f'val/strong_w_nfe{steps}'] = float(
                centered_log_weight_rmse(learned.log_weights, reference.log_weights)
            )
        for key, value in metrics.items():
            pl_module.log(key, value, on_step=False, on_epoch=True, rank_zero_only=True)
        log.info('FKC strong errors @ epoch %d: %s', trainer.current_epoch, metrics)

    def _log_figure(self, trainer: Trainer, figure, path: Path, log_key: str) -> None:
        """Save a figure to ``path`` and log it to WandB (if attached).

        The caller owns the figure lifetime: :meth:`_render` closes each figure
        in a ``finally`` block, also on save/log failures.
        """
        figure.savefig(path, dpi=120)
        wandb_logger = self._wandb_logger(trainer)
        if wandb_logger is not None:
            import wandb

            try:
                wandb_logger.experiment.log({log_key: wandb.Image(figure)}, step=trainer.global_step)
            except wandb.errors.Error as exc:  # pragma: no cover
                log.warning('wandb fkc log failed: %s', exc)
        log.info('FKC plot written to %s', path)
