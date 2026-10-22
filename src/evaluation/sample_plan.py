"""Deterministic sample plan shared by every ImageNet-256 evaluation.

The plan fixes, for a given ``(num_samples, block_size, seed, label_seed)``:

* class-balanced labels (``num_samples // num_classes`` per class, remainder
  spread over the lowest ids), shuffled once by ``label_seed``;
* one RNG *block* per ``block_size`` consecutive samples. Block ``b`` draws its
  initial noise from a generator seeded by ``(seed, b)`` and its sampler noise
  (Brownian paths / Euler--Maruyama increments) from ``(seed, b, nfe)``.

Every checkpoint, weight profile, rollout and NFE therefore sees identical labels
and initial noise, and the result is independent of how blocks are split across
GPUs or array tasks. A block is always generated as a single batch so the order
of sampler-RNG draws cannot depend on a batch size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LATENT_STREAM = 0
SAMPLER_STREAM = 1


@dataclass(frozen=True)
class SamplePlan:
    """Immutable description of the shared evaluation sample set."""

    num_samples: int = 50_000
    block_size: int = 250
    seed: int = 0
    label_seed: int = 0
    num_classes: int = 1000

    def __post_init__(self) -> None:
        """Reject non-positive sizes."""
        if self.num_samples < 1 or self.block_size < 1 or self.num_classes < 1:
            raise ValueError(f'invalid sample plan: {self}')

    @property
    def num_blocks(self) -> int:
        """Number of RNG blocks covering ``num_samples``."""
        return -(-self.num_samples // self.block_size)

    def block_range(self, block: int) -> range:
        """Global sample indices covered by ``block``."""
        if not 0 <= block < self.num_blocks:
            raise IndexError(f'block {block} outside [0, {self.num_blocks})')
        start = block * self.block_size
        return range(start, min(start + self.block_size, self.num_samples))


def balanced_labels(plan: SamplePlan) -> np.ndarray:
    """Return shuffled class-balanced int64 labels of length ``num_samples``."""
    ordered = np.arange(plan.num_samples, dtype=np.int64) % plan.num_classes
    return np.random.default_rng(plan.label_seed).permutation(ordered)


def derive_seed(*keys: int) -> int:
    """Hash integer keys into a 63-bit seed via NumPy's ``SeedSequence``."""
    state = np.random.SeedSequence([int(key) for key in keys]).generate_state(2, np.uint32)
    return int((int(state[0]) << 31) ^ int(state[1])) & ((1 << 63) - 1)


def latent_seed(plan: SamplePlan, block: int) -> int:
    """Seed for block ``block``'s initial noise (shared across NFE and rollouts)."""
    return derive_seed(plan.seed, LATENT_STREAM, block)


def sampler_seed(plan: SamplePlan, block: int, nfe: int) -> int:
    """Seed for block ``block``'s in-sampler noise at step count ``nfe``."""
    return derive_seed(plan.seed, SAMPLER_STREAM, block, nfe)


@dataclass(frozen=True, order=True)
class WorkUnit:
    """One ``(rollout, nfe, block)`` generation unit."""

    rollout: str
    nfe: int
    block: int


def enumerate_units(plan: SamplePlan, rollouts: tuple[str, ...], nfes: tuple[int, ...]):
    """List every work unit in a fixed order (rollout, nfe, block)."""
    return tuple(
        WorkUnit(rollout, nfe, block)
        for rollout in rollouts
        for nfe in nfes
        for block in range(plan.num_blocks)
    )


def units_for_task(units: tuple[WorkUnit, ...], task_index: int, num_tasks: int):
    """Stride-partition ``units`` so each array task gets a balanced share."""
    if not 0 <= task_index < num_tasks:
        raise ValueError(f'task_index {task_index} outside [0, {num_tasks})')
    return units[task_index::num_tasks]
