"""Shared FKC sampling helpers: load a trained augmented map and run the sampler.

``run_sampler`` follows Skreta et al. (2025), arXiv:2503.02819, Sec. 5: particles
are split into independent groups of ``K`` that are systematically resampled among
themselves after every step (log weights reset to zero); ``K = 1`` is the
no-resampling baseline. Used by every script in ``scripts/fkc``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.models.components.fkc.network import augmented_fkc_map
from src.models.components.fkc.sampling import _ei_step, _time_grid, sample_ei_martingale_path


def _load(run_dir: Path, checkpoint: str | None):
    config = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    ckpt = Path(checkpoint) if checkpoint else run_dir / 'checkpoints' / 'last.ckpt'
    module = hydra.utils.instantiate(config.model)
    module.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'])
    return module.eval(), ckpt


def batched_systematic_resample(
    x: torch.Tensor, log_w: torch.Tensor, k: int, generator: torch.Generator
) -> torch.Tensor:
    """Systematically resample each consecutive group of ``k`` particles independently.

    :param x: States ``(B * k, d)``, grouped contiguously.
    :param log_w: Log weights ``(B * k,)``.
    :return: Resampled states ``(B * k, d)``; particles never cross groups.
    """
    if k == 1:
        return x
    groups = log_w.shape[0] // k
    w = torch.softmax(log_w.view(groups, k), dim=-1)
    cdf = torch.cumsum(w, dim=-1)
    cdf[:, -1] = 1.0
    offset = torch.rand(groups, 1, generator=generator, dtype=log_w.dtype)
    positions = (torch.arange(k, dtype=log_w.dtype)[None] + offset) / k
    local = torch.searchsorted(cdf, positions).clamp_max(k - 1)
    index = (local + k * torch.arange(groups)[:, None]).reshape(-1)
    return x[index]


def run_sampler(
    sampler: str,
    steps: int,
    x0: torch.Tensor,
    k: int,
    resample_every: int,
    *,
    problem,
    net,
    schedule,
    t0: float,
    degree: int,
    generator: torch.Generator,
    diffusion_net=None,
    ess_trace: list[float] | None = None,
    clip_total_fraction: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Run independent K-particle SMC groups; optionally record pre-resampling ESS/K.

    Each trace entry averages the ESS fraction across groups at that step,
    BEFORE resetting the log weights. The last entry is the terminal ESS
    before the final weighted draw. K=1 has ESS/K=1 by definition.

    ``clip_total_fraction`` is the TOTAL share winsorized across both tails:
    0.10 clamps at the global 5th/95th log-weight percentiles every step.
    Clipping is applied to the accumulated log weights before resampling and
    their ESS, not to spatial positions or the downstream evaluation metrics.
    It matches ``_clip_log_weights(log_w, clip_total_fraction / 2)`` in the
    existing FKC sampler (which uses a per-tail fraction).
    """
    if not 0.0 <= clip_total_fraction < 1.0:
        raise ValueError('clip_total_fraction must be in [0, 1).')
    x = x0.clone()
    log_w = x.new_zeros(x.shape[0])
    times = _time_grid(t0, 1.0, steps, device=x.device, dtype=x.dtype)
    clock = schedule.ei_weighted_clock(times, t0)
    if sampler == 'diffusion':
        if diffusion_net is None:
            raise ValueError('diffusion sampler needs a separately trained score network.')
        from src.models.components.fkc.diffusion_baseline import LearnedScoreFKCFields

        fields = LearnedScoreFKCFields(diffusion_net, schedule)
    resamples = 0
    for step in range(steps):
        s, t = times[step], times[step + 1]
        if sampler == 'analytic':
            delta_m = (clock[step + 1] - clock[step]).sqrt() * torch.randn(
                x.shape, dtype=x.dtype, generator=generator
            )
            x, log_w = _ei_step(problem, x, log_w, s, t, delta_m, schedule, t0)
        else:
            # Sample precisely the same polynomial for both arms. The diffusion
            # Euler step only consumes its zeroth coefficient (Ito increment),
            # whereas the flow map also conditions on the higher coefficients.
            path = sample_ei_martingale_path(
                schedule, tuple(x.shape), s, t, degree, t0, dtype=x.dtype, generator=generator
            )
            if sampler == 'diffusion':
                x, log_w = _ei_step(fields, x, log_w, s, t, path.terminal_increment(), schedule, t0)
            elif sampler == 'learned':
                coefficients = path.coefficient_features(normalize=net.normalize_coefficients)
                x, log_w = augmented_fkc_map(
                    net, x, log_w, s, t, coefficients, path.terminal_increment(),
                    schedule.weight_span(s, t, t0), schedule,
                )
            else:
                raise ValueError(f'unknown sampler {sampler!r}')
        if clip_total_fraction > 0.0:
            # Keep the same global-cloud quantile convention as the existing
            # sampling._clip_log_weights; never clip within K-particle groups.
            from src.models.components.fkc.sampling import _clip_log_weights

            log_w = _clip_log_weights(log_w, clip_total_fraction / 2.0)
        if ess_trace is not None:
            # Equal-sized groups; float64 softmax prevents underflow with
            # widely separated finite log weights. Nonfinite weights are
            # recorded as NaN instead of silently counted as healthy ESS.
            weights = log_w.reshape(-1, k).double()
            if bool(torch.isfinite(weights).all()):
                weights = torch.softmax(weights, dim=-1)
                ess_trace.append(float((1.0 / (k * weights.square().sum(-1))).mean()))
            else:
                ess_trace.append(float('nan'))
        # Resample within each K-group after every `resample_every`-th step; the
        # last step's weights are always consumed by the terminal draw below.
        if k > 1 and (step + 1) % resample_every == 0 and step + 1 < steps:
            x = batched_systematic_resample(x, log_w, k, generator)
            log_w = torch.zeros_like(log_w)
            resamples += 1
    # Terminal draw from each group according to the weights since the last resample.
    x = batched_systematic_resample(x, log_w, k, generator)
    return x, resamples
