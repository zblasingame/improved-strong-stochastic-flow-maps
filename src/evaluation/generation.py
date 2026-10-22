"""Model-agnostic block generation onto the sharded sample store.

A *rollout* is any callable ``(latents, labels, nfe, generator) -> latents`` that
integrates initial noise to a data latent at a given step count; model-specific
rollouts are built by ``src/evaluation/rollouts.py``. Sampling runs under
bf16 autocast on CUDA and VAE decoding in fp32 for every model, so students and
the teacher share one image pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.evaluation.sample_plan import SamplePlan, WorkUnit, latent_seed, sampler_seed
from src.evaluation.sample_store import Shard, shard_path, write_shard
from src.models.components.gaussian import randn
from src.models.components.metrics import to_fid_uint8

RolloutFn = Callable[[Tensor, Tensor, int, torch.Generator], Tensor]
DecodeFn = Callable[[Tensor], Tensor]  # latents -> float images in [0, 1]


def sampling_autocast(device: torch.device):
    """bf16 autocast on CUDA (matching the training-time FID path), no-op on CPU."""
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def to_uint8_nhwc(images: Tensor) -> np.ndarray:
    """Quantize ``[0, 1]`` NCHW images exactly as the FID path does, as NHWC."""
    return to_fid_uint8(images.float().mul(2.0).sub(1.0)).permute(0, 2, 3, 1).cpu().numpy()


@torch.inference_mode()
def generate_block(
    rollout: RolloutFn,
    decode: DecodeFn,
    plan: SamplePlan,
    labels: np.ndarray,
    unit: WorkUnit,
    latent_shape: tuple[int, ...],
    device: torch.device,
) -> Shard:
    """Generate one plan block as a single batch with its plan-derived seeds."""
    indices = plan.block_range(unit.block)
    block_labels = torch.as_tensor(labels[indices.start : indices.stop], device=device)
    latent_gen = torch.Generator(device=device).manual_seed(latent_seed(plan, unit.block))
    sampler_gen = torch.Generator(device=device).manual_seed(
        sampler_seed(plan, unit.block, unit.nfe)
    )
    latents = randn((len(indices), *latent_shape), device=device, generator=latent_gen)
    with sampling_autocast(device):
        samples = rollout(latents, block_labels, unit.nfe, sampler_gen)
    images = decode(samples.float())
    return Shard(
        images=to_uint8_nhwc(images),
        labels=block_labels.cpu().numpy().astype(np.int64),
        indices=np.arange(indices.start, indices.stop, dtype=np.int64),
    )


def run_units(
    units: tuple[WorkUnit, ...],
    rollouts: Mapping[str, RolloutFn],
    decode: DecodeFn,
    plan: SamplePlan,
    labels: np.ndarray,
    set_dir: Callable[[WorkUnit], Path],
    latent_shape: tuple[int, ...],
    device: torch.device,
    log: Callable[[str], None] = print,
) -> int:
    """Generate every unit whose shard is missing; return the number written."""
    written = 0
    for position, unit in enumerate(units, start=1):
        path = shard_path(set_dir(unit), unit.block)
        if path.exists():
            continue
        shard = generate_block(
            rollouts[unit.rollout], decode, plan, labels, unit, latent_shape, device
        )
        write_shard(path, shard)
        written += 1
        log(f'[{position}/{len(units)}] {unit.rollout} nfe={unit.nfe} block={unit.block}')
    return written
