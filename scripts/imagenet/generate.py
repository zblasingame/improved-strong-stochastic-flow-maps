"""Generate the shared 50k ImageNet-256 sample sets for one checkpoint.

Writes sharded uint8 ``.npz`` samples for every ``(rollout, nfe)`` under::

    {eval_dir}/{weights}/{rollout}/nfe_{k}/

using the fixed :class:`~src.evaluation.sample_plan.SamplePlan` so every
checkpoint sees identical labels and noise. Work is stride-partitioned over
SLURM array tasks (``--task-index`` / ``--num-tasks`` default to the SLURM env);
finished shards are skipped so resubmission resumes.

Examples::

    # SSFM student (EMA weights, SDE rollout at the training zeta)
    uv run python -m scripts.imagenet.generate student \
        --checkpoint RUN/checkpoints/best.ckpt --config RUN/.hydra/config.yaml \
        --eval-dir OUT/RUN_NAME/step_25000

    # SiT-XL/2+REPA teacher (CFG 2.0 on paper-time [0, 0.7])
    uv run python -m scripts.imagenet.generate teacher --eval-dir OUT/sit_xl2_repa
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import rootutils
import torch

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.evaluation import checkpoints as ckpt  # noqa: E402
from src.evaluation.generation import run_units  # noqa: E402
from src.evaluation.rollouts import (  # noqa: E402
    SDE,
    RolloutSet,
    student_rollouts,
    teacher_nfe,
    teacher_rollouts,
    training_zeta,
)
from src.evaluation.sample_plan import (  # noqa: E402
    SamplePlan,
    balanced_labels,
    enumerate_units,
    units_for_task,
)
from src.evaluation.sample_store import (  # noqa: E402
    MANIFEST_NAME,
    SCHEMA_VERSION,
    nfe_dir,
    write_json_atomic,
)

DEFAULT_NFES = (1, 2, 4, 8, 16)
LATENT_SHAPE = (4, 32, 32)
DEFAULT_TEACHER = Path(os.environ.get('SIT_CHECKPOINT_PATH', ''))
VAE_NAME = 'stabilityai/sd-vae-ft-mse'


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(  # noqa: S603
            ['git', 'rev-parse', 'HEAD'],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_student(args: argparse.Namespace, device: torch.device):
    """Instantiate the recorded student and load the requested weights."""
    import hydra
    from omegaconf import OmegaConf

    from src.models.ei_doss_module import EIDossSussmannLitModule

    net, config, meta = ckpt.load_student(args.config, args.checkpoint, device, args.weights)
    weights, global_step, checkpoint_epoch = meta['weights'], meta['global_step'], meta['epoch']
    ei_doss = issubclass(hydra.utils.get_class(config.model._target_), EIDossSussmannLitModule)
    rollouts = student_rollouts(
        net,
        hydra.utils.instantiate(config.model.schedule),
        ei_doss=ei_doss,
        degree=int(config.model.loss.degree),
        t0=float(config.model.loss.epsilon),
        zeta=training_zeta(config, global_step),
    )
    source = {
        'kind': 'student',
        'checkpoint': str(args.checkpoint),
        'config': str(args.config),
        'global_step': global_step,
        'checkpoint_epoch': checkpoint_epoch,
        'run_name': OmegaConf.select(config, 'logger.wandb.name', default=None),
        'module': config.model._target_,
    }
    return weights, rollouts, source


def _build_release(args: argparse.Namespace, device: torch.device):
    """Load a released model (Hugging Face name or local folder) with its SDE rollout."""
    import hydra

    from src.evaluation.pretrained import load_pretrained
    from src.models.ei_doss_module import EIDossSussmannLitModule

    net, config = load_pretrained(args.model, device)
    ei_doss = issubclass(hydra.utils.get_class(config._target_), EIDossSussmannLitModule)
    rollouts = student_rollouts(
        net,
        hydra.utils.instantiate(config.schedule),
        ei_doss=ei_doss,
        degree=int(config.loss.degree),
        t0=float(config.loss.epsilon),
        zeta=1.0,
    )
    source = {'kind': 'release', 'model': args.model, 'module': config._target_, 'global_step': None}
    return 'ema', rollouts, source


def _build_teacher(args: argparse.Namespace, device: torch.device):
    """Load the SiT/DMF teacher with its interval-CFG Euler--Maruyama SDE rollout."""
    from src.models.components.dmf import load_dmf_xl2
    from src.models.components.schedule import LinearSchedule

    model = load_dmf_xl2(str(args.checkpoint)).to(device).eval()
    cfg_scale = None if args.no_cfg else args.cfg_scale
    rollouts = teacher_rollouts(
        model,
        LinearSchedule(noise_scale=1.0),
        cfg_scale=cfg_scale,
        guidance_interval=tuple(args.guidance_interval),
        eps=args.eps,
    )
    source = {'kind': 'teacher', 'checkpoint': str(args.checkpoint), 'global_step': None}
    return 'teacher', rollouts, source


def _step_accounting(args: argparse.Namespace, steps: int) -> dict[str, int]:
    """Record sampler steps and the model evaluations they actually cost.

    For the teacher the ``nfe_{k}`` directory key is the step count; interval
    CFG doubles the evaluations of guided steps, so ``actual_nfe`` differs.
    """
    if args.kind != 'teacher':
        return {'steps': steps, 'actual_nfe': steps}
    cfg_scale = None if args.no_cfg else args.cfg_scale
    actual = teacher_nfe(
        steps, eps=args.eps, cfg_scale=cfg_scale, guidance_interval=tuple(args.guidance_interval)
    )
    return {'steps': steps, 'actual_nfe': actual}


def _write_manifests(
    args: argparse.Namespace,
    plan: SamplePlan,
    weights: str,
    rollouts: RolloutSet,
    source: dict[str, Any],
) -> None:
    """Write one manifest per sample set (idempotent; identical across tasks)."""
    for rollout in args.rollouts:
        for nfe in args.nfes:
            directory = nfe_dir(args.eval_dir, weights, rollout, nfe)
            write_json_atomic(
                directory / MANIFEST_NAME,
                {
                    'schema_version': SCHEMA_VERSION,
                    'weights': weights,
                    'rollout': rollout,
                    'nfe': nfe,
                    **_step_accounting(args, nfe),
                    'plan': {**vars(plan), 'num_blocks': plan.num_blocks},
                    'label_sampling': 'class-balanced, shuffled by label_seed',
                    'sampler': rollouts.meta[rollout],
                    'source': source,
                    'vae': VAE_NAME,
                    'image_format': 'uint8 NHWC RGB 256x256',
                    'autocast': 'bf16 sampling, fp32 VAE decode',
                    'git_commit': _git_commit(),
                },
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest='kind', required=True)
    student = sub.add_parser('student')
    student.add_argument('--checkpoint', required=True, type=Path)
    student.add_argument('--config', required=True, type=Path, help='Run .hydra/config.yaml')
    student.add_argument(
        '--weights', default=None, help="'ema' (default), 'sigma_x.xxx', 'non_ema'"
    )
    release = sub.add_parser('release', help='a released model from the Hugging Face Hub')
    release.add_argument('--model', required=True, help="e.g. 'issfm' or a local folder")
    teacher = sub.add_parser('teacher')
    teacher.add_argument('--checkpoint', type=Path, default=DEFAULT_TEACHER)
    teacher.add_argument('--cfg-scale', type=float, default=2.0)
    teacher.add_argument('--no-cfg', action='store_true')
    teacher.add_argument('--guidance-interval', nargs=2, type=float, default=(0.0, 0.7))
    teacher.add_argument('--eps', type=float, default=1e-2)
    for p in (student, release, teacher):
        p.add_argument('--eval-dir', required=True, type=Path)
        p.add_argument('--nfes', nargs='+', type=int, default=list(DEFAULT_NFES))
        p.add_argument('--num-samples', type=int, default=50_000)
        p.add_argument('--block-size', type=int, default=250)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--label-seed', type=int, default=0)
        p.add_argument(
            '--task-index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        )
        p.add_argument(
            '--num-tasks', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        )
        p.add_argument('--manifest-only', action='store_true', help='Write manifests and exit')
    return parser.parse_args()


def main() -> None:
    """Generate this task's share of the configured sample sets."""
    args = _parse_args()
    if any(nfe < 1 for nfe in args.nfes):
        raise ValueError(f'NFE levels must be positive; got {args.nfes}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build = {'student': _build_student, 'release': _build_release, 'teacher': _build_teacher}[
        args.kind
    ]
    weights, rollouts, source = build(args, device)
    # SDE-only by design: every builder exposes exactly one rollout.
    args.rollouts = [SDE]
    if set(rollouts.fns) != {SDE}:
        raise ValueError(f'expected an SDE-only rollout set; got {sorted(rollouts.fns)}')

    plan = SamplePlan(args.num_samples, args.block_size, args.seed, args.label_seed)
    _write_manifests(args, plan, weights, rollouts, source)
    if args.manifest_only:
        return
    units = units_for_task(
        enumerate_units(plan, tuple(args.rollouts), tuple(args.nfes)),
        args.task_index,
        args.num_tasks,
    )
    print(f'task {args.task_index}/{args.num_tasks}: {len(units)} units, weights={weights}')

    from src.models.components.vae_decoder import StabilityVAEDecoder

    decoder = StabilityVAEDecoder(VAE_NAME).to(device)
    written = run_units(
        units,
        rollouts.fns,
        decoder,
        plan,
        balanced_labels(plan),
        lambda unit: nfe_dir(args.eval_dir, weights, unit.rollout, unit.nfe),
        LATENT_SHAPE,
        device,
    )
    print(f'task {args.task_index}: wrote {written} shards')


if __name__ == '__main__':
    main()
