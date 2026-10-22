"""Independent, diagonal-score diffusion baseline for the Gaussian40 experiment."""

from copy import deepcopy

import lightning as L
import torch
from torch import Tensor, nn

from .network import _TimeEmbed, _trunk


class Gaussian40ScoreMLP(nn.Module):
    """Predict ``-sqrt(sigma(t)^2 + floor^2) * score_qt(x)`` without a flow map."""

    def __init__(
        self, hidden_size: int = 512, depth: int = 6, time_emb_size: int = 128,
        spatial_input_scale: float = 50.0, score_scale_floor: float = 0.04253856,
    ) -> None:
        super().__init__()
        self.spatial_input_scale = spatial_input_scale
        self.score_scale_floor = score_scale_floor
        self.time_embed = _TimeEmbed(hidden_size, time_emb_size)
        self.trunk = _trunk(2 + hidden_size, hidden_size, depth)
        self.head = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.head(self.trunk(torch.cat((self.spatial_input_scale * x, self.time_embed(t)), -1)))


class Gaussian40ScoreModule(L.LightningModule):
    """Fit the *untempered* 40-GMM score, not the analytic beta=3 FKC field.

    At inference the trained score is multiplied by beta to obtain the target
    score drift, and its squared norm supplies the FKC potential. No analytic
    scores enter either inference arm. Training uses the same Gaussian modes,
    geometric VE schedule, state pool, batch size and optimizer budget as the
    augmented flow-map run. Only a diagonal score regression is learned.
    """

    def __init__(
        self, net: Gaussian40ScoreMLP, problem: nn.Module, schedule: nn.Module,
        optimizer, ema_decay: float = 0.999, pool_num_s: int = 64,
        pool_per_s: int = 2048,
    ) -> None:
        super().__init__()
        if problem.powers != (1,):
            raise ValueError('Diffusion baseline must train on beta=1, not the beta=3 target.')
        self.save_hyperparameters(ignore=['net', 'problem', 'schedule', 'optimizer'])
        self.net, self.problem, self.schedule = net, problem, schedule
        self.optimizer_partial = optimizer
        self.ema_net = deepcopy(net)
        self.ema_net.requires_grad_(False)
        self.pool_num_s, self.pool_per_s = pool_num_s, pool_per_s

    def _loss(self, batch_size: int, generator: torch.Generator | None = None) -> Tensor:
        if getattr(self.problem, '_state_pool', None) is None:
            self.problem.build_state_pool(
                num_s=self.pool_num_s, per_s=self.pool_per_s,
                s_range=(0.0, 1.0), generator=generator,
            )
        x, t = self.problem.sample_pool(batch_size, generator=generator)
        with torch.no_grad():
            sigma = self.schedule.sigma(t)
            target = -(sigma.square() + self.net.score_scale_floor**2).sqrt()[:, None] * (
                self.problem.experts[0].score(x, t)
            )
        return (self.net(x, t) - target).square().sum(-1).mean()

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch[0].shape[0])
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        gen = torch.Generator(device=self.device).manual_seed(1234)
        loss = self._loss(batch[0].shape[0], gen)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        gen = torch.Generator(device=self.device).manual_seed(4321)
        self.log('test/loss', self._loss(batch[0].shape[0], gen), on_epoch=True)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        with torch.no_grad():
            for ema, live in zip(self.ema_net.parameters(), self.net.parameters(), strict=True):
                ema.lerp_(live, 1.0 - self.hparams.ema_decay)

    def configure_optimizers(self):
        return self.optimizer_partial(params=self.net.parameters())


class LearnedScoreFKCFields:
    """Anneal a *learned beta=1 score* to beta=3, including its FKC potential.

    Exactly the target-score (unscaled-noise) SDE of the reference notebook:
    ``b = beta * nu^2 * score`` and
    ``g = beta*(beta-1)/2 * nu^2 * ||score||^2``. The analytic score and
    beta=3 composed field are never queried during a rollout.
    """

    def __init__(self, net: Gaussian40ScoreMLP, schedule: nn.Module, beta: float = 3.0):
        self.net, self.schedule, self.beta = net, schedule, beta

    def fields(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).expand(x.shape[0])
        scale = (self.schedule.sigma(t).square() + self.net.score_scale_floor**2).sqrt()
        score = -self.net(x, t) / scale[:, None]
        nu2 = self.schedule.nu_squared(t)
        velocity = 0.5 * self.beta * nu2[:, None] * score
        potential = 0.5 * self.beta * (self.beta - 1.0) * nu2 * score.square().sum(-1)
        return velocity, potential
