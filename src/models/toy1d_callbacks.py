"""Strong-error evaluation callbacks for the 1D non-linear SDE form grid.

One shared fine realization couples the tamed Euler-Maruyama reference of
the *true* SDE to each learned map rolled out at a sweep of step counts
(the 1D analogue of the 2D toy's coupled-rollout panels). Every arm rides
the *closed-form martingale* on the clock: piecewise clock-polynomial
leaves whose raw increments are the exact SDE noise per fine step; coarse
steps consume cross-leaf restrictions of the same realization
(``PiecewiseBrownianPath`` on the clock grid), so reference and map stay
pathwise coupled. The arms differ only in their training objectives —
``wz`` and ``none`` share the x-space increment map
``x + (t-s) u + (M_t - M_s)``; ``ds`` reattaches after a Z-map step;
``ds_global`` chains the globally anchored Doss state ``Z_r = X_r - M_r``
and reattaches the full ``M_1`` only at the endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from lightning import Callback, LightningModule, Trainer
from torch import Tensor

from src.models.components.brownian import BrownianPolynomial
from src.models.components.strong_convergence import PiecewiseBrownianPath
from src.models.components.toy_martingale import (
    sample_piecewise_toy_martingale,
    toy_clock,
)
from src.models.ssfm_module import SSFMLitModule


class Toy1DStrongErrorCallback(Callback):
    """Log the strong endpoint error of the learned map per step count.

    :param num_samples: Number of shared-realization particles per panel.
    :param step_counts: Coarse step counts of the map rollout sweep.
    :param em_steps: Fine tamed-EM step count of the true-SDE reference.
    :param every_n_epochs: Validate the panels every this many epochs.
    :param seed: Seed of the shared evaluation draw.
    :param log_key: Metric prefix for the logged strong errors.
    :param use_ema: Evaluate the EMA teacher (default) or the live net.
    """

    def __init__(
        self,
        num_samples: int = 4096,
        step_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
        em_steps: int = 512,
        every_n_epochs: int = 20,
        seed: int = 0,
        log_key: str = 'val/strong_error',
        use_ema: bool = True,
    ) -> None:
        super().__init__()
        if num_samples < 1:
            raise ValueError('num_samples must be positive.')
        if em_steps < 1:
            raise ValueError('em_steps must be positive.')
        if any(count < 1 or count > em_steps for count in step_counts):
            raise ValueError(f'step_counts must lie in [1, em_steps]; got {tuple(step_counts)}.')
        self.num_samples = num_samples
        self.step_counts = tuple(dict.fromkeys([*step_counts]))
        self.em_steps = em_steps
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.log_key = log_key
        self.use_ema = use_ema

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log strong errors every ``every_n_epochs`` epochs."""
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 0 or (epoch + 1) % self.every_n_epochs != 0:
            return
        for count, error in self.strong_errors(pl_module).items():
            pl_module.log(f'{self.log_key}/steps_{count}', error, sync_dist=True)

    def strong_errors(self, pl_module: LightningModule) -> dict[int, Tensor]:
        """Return the mean strong endpoint error per step count."""
        if not isinstance(pl_module, SSFMLitModule):
            raise TypeError('Toy1DStrongErrorCallback requires an SSFMLitModule.')
        loss = pl_module.loss
        sde = getattr(loss, 'sde', None)
        arm = getattr(loss, 'arm', None)
        degree = getattr(loss, 'degree', None)
        if sde is None or arm is None or degree is None:
            raise TypeError('Toy1DStrongErrorCallback requires module.loss.sde/.arm/.degree.')
        net = pl_module._teacher() if self.use_ema else pl_module.net
        device = pl_module.device
        dtype = torch.float32
        generator = torch.Generator(device=device).manual_seed(self.seed)
        noise = torch.randn(self.num_samples, 1, device=device, dtype=dtype, generator=generator)
        x_0 = sde.initial_condition(noise)
        fine_nodes = torch.linspace(0.0, 1.0, self.em_steps + 1, device=device, dtype=dtype)

        clock_nodes, leaves = sample_piecewise_toy_martingale(
            sde,
            (self.num_samples, 1),
            degree,
            self.em_steps,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        reference = self._reference(sde, x_0, fine_nodes, leaves, scale_by_diffusion=False)
        piecewise = PiecewiseBrownianPath(nodes=clock_nodes, leaves=leaves)
        if arm == 'ds':
            rollout = self._rollout_ds(net, sde, piecewise, x_0)
        elif arm == 'ds_global':
            rollout = self._rollout_ds_global(net, sde, piecewise, x_0)
        else:
            # wz and none share the x-space increment map: the arms differ
            # only in their training objectives.
            rollout = self._rollout_wz(net, sde, piecewise, x_0)

        errors: dict[int, Tensor] = {}
        for count in self.step_counts:
            endpoints = rollout(count)
            error = (endpoints - reference).square().flatten(1).mean(1)
            errors[count] = error.mean().detach()
        return errors

    @staticmethod
    def _reference(
        sde,
        x_0: Tensor,
        nodes: Tensor,
        leaves: list[BrownianPolynomial],
        *,
        scale_by_diffusion: bool,
    ) -> Tensor:
        """Fine tamed EM reference riding the shared realization increments.

        ``scale_by_diffusion`` multiplies each leaf increment by
        ``sqrt(beta_mid)`` — the plain arm's native Brownian leaves need it;
        the clock-polynomial leaves of the closed-form martingale already
        are the exact SDE noise per step.
        """
        state = x_0
        for index in range(len(leaves)):
            mid = 0.5 * (nodes[index] + nodes[index + 1])
            step = nodes[index + 1] - nodes[index]
            increment = leaves[index].raw_increment().to(dtype=state.dtype)
            if scale_by_diffusion:
                increment = sde.diffusion(mid) * increment
            state = state + sde.tamed_drift(state, mid, step) * step + increment
        return state

    def _rollout_wz(self, net, sde, piecewise: PiecewiseBrownianPath, x_0: Tensor):
        """WZ map chain: ``x + (t-s) u + M_t - M_s`` on clock restrictions."""

        def rollout(count: int) -> Tensor:
            device, dtype = x_0.device, x_0.dtype
            ones = torch.ones(x_0.shape[0], device=device, dtype=dtype)
            grid = torch.linspace(0.0, 1.0, count + 1, device=device, dtype=dtype)
            state = x_0
            for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
                path = piecewise.interval(toy_clock(sde, s_cur), toy_clock(sde, t_nxt))
                features = path.coefficient_features(normalize=net.normalize_coefficients)
                velocity = net.drift(state, s_cur.expand_as(ones), t_nxt.expand_as(ones), features)
                state = state + (t_nxt - s_cur) * velocity + path.raw_increment()
            return state

        return rollout

    def _rollout_ds(self, net, sde, piecewise: PiecewiseBrownianPath, x_0: Tensor):
        """DS map chain: Z-map step then reattach ``M_t - M_s``."""

        def rollout(count: int) -> Tensor:
            device, dtype = x_0.device, x_0.dtype
            ones = torch.ones(x_0.shape[0], device=device, dtype=dtype)
            grid = torch.linspace(0.0, 1.0, count + 1, device=device, dtype=dtype)
            state = x_0
            for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
                path = piecewise.interval(toy_clock(sde, s_cur), toy_clock(sde, t_nxt))
                features = path.coefficient_features(normalize=net.normalize_coefficients)
                z_t = net(state, s_cur.expand_as(ones), t_nxt.expand_as(ones), features)
                state = z_t + path.raw_increment()
            return state

        return rollout

    def _rollout_ds_global(self, net, sde, piecewise: PiecewiseBrownianPath, x_0: Tensor):
        """DS-global chain: global Z-map steps; reattach the full ``M_1`` at the end.

        The state stays in global Doss coordinates ``Z_r = X_r - M_r`` (``M``
        anchored at 0, so ``Z_0 = X_0``); each step consumes the same
        interval-restriction features as the ``ds`` arm. Only the endpoint is
        reattached, ``X_1 = Z_1 + M_1``, with ``M_1`` the exact full-span
        increment of the shared piecewise realization.
        """
        total = piecewise.interval(toy_clock(sde, 0.0), toy_clock(sde, 1.0))
        m_1 = total.raw_increment()

        def rollout(count: int) -> Tensor:
            device, dtype = x_0.device, x_0.dtype
            ones = torch.ones(x_0.shape[0], device=device, dtype=dtype)
            grid = torch.linspace(0.0, 1.0, count + 1, device=device, dtype=dtype)
            state = x_0  # Z_0 = X_0 because M_0 = 0.
            for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
                path = piecewise.interval(toy_clock(sde, s_cur), toy_clock(sde, t_nxt))
                features = path.coefficient_features(normalize=net.normalize_coefficients)
                state = net(state, s_cur.expand_as(ones), t_nxt.expand_as(ones), features)
            return state + m_1.to(dtype=state.dtype)

        return rollout
