"""SDE rollout builders for SSFM students and the SiT/DMF teacher.

Evaluation is SDE-only by design; there is deliberately no ODE rollout.

* EI-Doss-Sussmann / WZ students: ``ei_doss_sussmann_sampler`` at the training
  ``zeta`` (the SDE the model is trained on, as in ``LatentFIDCallback``).
* EI-Euler students: ``ei_euler_sampler``.
* SiT/DMF teacher: Euler--Maruyama on the flow-as-SDE with interval CFG.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.evaluation.generation import RolloutFn
from src.models.components.dmf import DMFT
from src.models.components.samplers import (
    ei_doss_sussmann_sampler,
    ei_euler_sampler,
    fm_em_sampler,
)

SDE = 'sde'


def project_time(paper_time: Tensor | float) -> Tensor:
    """Convert DMF-paper time (``0`` data, ``1`` noise) to project time (``0`` noise).

    :param paper_time: Time(s) in the DMF/SiT convention.
    :return: ``1 - paper_time`` as a tensor.
    """
    return 1.0 - torch.as_tensor(paper_time)


def sit_velocity(model: DMFT, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
    """Evaluate the SiT/DMF teacher as a project-convention velocity field.

    The DMF checkpoint is called as ``model(x, dmf_t, dmf_t, labels)`` and predicts
    the velocity pointing from data to noise; the project convention needs the
    noise-to-data velocity at ``t_project = 1 - t_paper``, hence the sign flip
    (mirrors ``FixedCoefficientDMFEIEulerSSFM.pretrained_prediction``).

    :param model: A loaded :class:`DMFT` checkpoint.
    :param x: Latents of shape ``(B, 4, 32, 32)``.
    :param t: Project times of shape ``(B,)``.
    :param labels: Class labels of shape ``(B,)``.
    :return: Velocity ``u(x, t)`` in the project (noise-to-data) convention.
    """
    dmf_t = 1.0 - t.reshape(-1)
    return -model(x, dmf_t, dmf_t, labels)


def make_guided_velocity(
    model: DMFT,
    labels: Tensor,
    cfg_scale: float | None,
    guidance_interval: tuple[float, float] = (0.0, 0.7),
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build the ``model(x, t) -> velocity`` callable for ``fm_em_sampler``.

    When ``cfg_scale`` is ``None`` the plain conditional velocity is returned.
    Otherwise classifier-free guidance ``uncond + w (cond - uncond)`` is applied
    only inside the guidance interval (Kynkaanniemi et al., 2024). The interval is
    given in DMF paper time ``[t_min, t_max]`` (``0`` = data, ``1`` = noise) and
    converted to project time: guidance is active for project times in
    ``[1 - t_max, 1 - t_min]``.

    :param model: A loaded :class:`DMFT` checkpoint.
    :param labels: Per-sample class labels of shape ``(B,)``.
    :param cfg_scale: CFG scale ``w``; ``None`` disables guidance entirely.
    :param guidance_interval: ``(t_min, t_max)`` in paper time.
    :return: Callable matching the ``fm_em_sampler`` model signature.
    """
    if cfg_scale is None:

        def unguided(x: Tensor, t: Tensor) -> Tensor:
            return sit_velocity(model, x, t, labels)

        return unguided

    t_lo, t_hi = project_time(guidance_interval[1]), project_time(guidance_interval[0])
    null_labels = torch.full_like(labels, model.num_classes)

    def guided(x: Tensor, t: Tensor) -> Tensor:
        if bool(((t >= t_lo) & (t <= t_hi)).all()):
            combined = sit_velocity(
                model,
                torch.cat((x, x)),
                torch.cat((t, t)),
                torch.cat((labels, null_labels)),
            )
            cond, uncond = combined.chunk(2)
            return uncond + cfg_scale * (cond - uncond)
        return sit_velocity(model, x, t, labels)

    return guided


@dataclass(frozen=True)
class RolloutSet:
    """Named rollouts plus per-rollout metadata recorded in each manifest."""

    fns: dict[str, RolloutFn]
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)


def training_zeta(config: DictConfig, global_step: int) -> float:
    """Annealing amplitude at ``global_step`` (mirrors ``DMFWZEIIMFAnchorLoss.zeta_at``)."""
    steps = OmegaConf.select(config, 'model.loss.zeta_anneal_steps', default=None)
    if not steps or steps <= 0:
        return 1.0
    return min(1.0, global_step / float(steps))


def _labelled(net, labels: Tensor, sample):  # noqa: ANN001, ANN202
    """Run ``sample()`` with ``labels`` bound on the net, always unbinding."""
    net.set_labels(labels)
    try:
        return sample()
    finally:
        net.set_labels(None)


def student_rollouts(
    net,  # noqa: ANN001
    schedule,  # noqa: ANN001
    *,
    ei_doss: bool,
    degree: int,
    t0: float,
    zeta: float,
) -> RolloutSet:
    """Build the student SDE rollout (EI-Doss-Sussmann at ``zeta``, or EI-Euler)."""
    common = {'degree': degree, 't0': t0, 'cfg_scale': None}
    if ei_doss and zeta <= 0.0:
        raise ValueError(f'SDE evaluation requires zeta > 0; got {zeta}')

    def run(latents: Tensor, labels: Tensor, nfe: int, gen: torch.Generator) -> Tensor:
        if ei_doss:
            sample = lambda: ei_doss_sussmann_sampler(  # noqa: E731
                net,
                schedule,
                latents,
                num_steps=nfe,
                degree=degree,
                t0=t0,
                zeta=zeta,
                generator=gen,
            )
        else:
            sample = lambda: ei_euler_sampler(  # noqa: E731
                net, schedule, latents, num_steps=nfe, degree=degree, t0=t0, generator=gen
            )
        return _labelled(net, labels, sample)

    meta = (
        {'sampler': 'ei_doss_sussmann_sampler', 'zeta': zeta, **common}
        if ei_doss
        else {'sampler': 'ei_euler_sampler', **common}
    )
    return RolloutSet(fns={SDE: run}, meta={SDE: meta})


def teacher_nfe(
    num_steps: int, *, eps: float, cfg_scale: float | None, guidance_interval: tuple[float, float]
) -> int:
    """Model evaluations used by ``fm_em_sampler`` + interval CFG at ``num_steps``.

    Mirrors the sampler's uniform grid on ``[eps, 1]`` (one evaluation at each
    step's start time) and ``make_guided_velocity``'s rule: steps whose start
    time lies in the project-time guidance interval ``[1 - t_max, 1 - t_min]``
    cost two evaluations (conditional + unconditional), all others one.
    """
    from src.models.components.samplers import get_tsteps

    starts = get_tsteps(num_steps, t_start=eps, t_end=1.0)[:-1].float()
    if cfg_scale is None:
        return num_steps
    # Same float32 tensor arithmetic as project_time above.
    t_lo = 1.0 - torch.as_tensor(guidance_interval[1])
    t_hi = 1.0 - torch.as_tensor(guidance_interval[0])
    guided = int(((starts >= t_lo) & (starts <= t_hi)).sum())
    return num_steps + guided


def teacher_rollouts(
    model,  # noqa: ANN001
    schedule,  # noqa: ANN001
    *,
    cfg_scale: float | None,
    guidance_interval: tuple[float, float],
    eps: float,
) -> RolloutSet:
    """Build the SiT/DMF teacher Euler--Maruyama SDE rollout with interval CFG."""
    def run(latents: Tensor, labels: Tensor, nfe: int, gen: torch.Generator) -> Tensor:
        velocity = make_guided_velocity(model, labels, cfg_scale, guidance_interval)
        return fm_em_sampler(
            velocity,
            schedule,
            latents,
            prediction='velocity',
            num_steps=nfe,
            eps=eps,
            generator=gen,
        )

    meta = {
        'sampler': 'fm_em_sampler',
        'eps': eps,
        'cfg_scale': cfg_scale,
        'guidance_interval_paper_time': list(guidance_interval),
        'cfg_evals_per_step': 2 if cfg_scale else 1,
    }
    return RolloutSet(fns={SDE: run}, meta={SDE: meta})
