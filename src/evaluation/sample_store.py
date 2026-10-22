"""Sharded uint8 sample store: one ``.npz`` shard per sample-plan block.

Layout under an eval directory::

    {eval_dir}/{weights}/{rollout}/nfe_{k}/
        manifest.json            generation metadata (plan, sampler, checkpoint)
        shard_00000.npz          images uint8 (N, 256, 256, 3), labels, indices
        ...
        metrics/{metric}.json    cached per-metric results

Shards are written atomically (temp file + rename) so an interrupted array task
never leaves a truncated shard and resubmission skips finished blocks.

Stdlib + NumPy only: imported by the isolated preference-scoring environment.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_NAME = 'manifest.json'
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Shard:
    """One loaded block of generated samples."""

    images: np.ndarray  # uint8 (N, H, W, 3)
    labels: np.ndarray  # int64 (N,)
    indices: np.ndarray  # int64 (N,) global sample indices


def nfe_dir(eval_dir: Path, weights: str, rollout: str, nfe: int) -> Path:
    """Directory holding one ``(weights, rollout, nfe)`` sample set."""
    return Path(eval_dir) / weights / rollout / f'nfe_{nfe}'


def shard_path(directory: Path, block: int) -> Path:
    """Path of block ``block``'s shard inside a sample-set directory."""
    return Path(directory) / f'shard_{block:05d}.npz'


def _atomic_write_bytes(path: Path, writer) -> None:  # noqa: ANN001
    """Write via ``writer(file_obj)`` into a temp file, then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_shard(path: Path, shard: Shard) -> None:
    """Validate and atomically write one shard (uncompressed ``.npz``)."""
    images, labels, indices = shard.images, shard.labels, shard.indices
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f'images must be uint8 NHWC RGB; got {images.dtype} {images.shape}')
    if not len(images) == len(labels) == len(indices):
        raise ValueError('images, labels and indices must have equal length')
    _atomic_write_bytes(
        Path(path),
        lambda handle: np.savez(
            handle,
            images=images,
            labels=labels.astype(np.int64),
            indices=indices.astype(np.int64),
        ),
    )


def read_shard(path: Path) -> Shard:
    """Load one shard fully into memory."""
    with np.load(path) as data:
        return Shard(data['images'], data['labels'], data['indices'])


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as indented JSON."""
    text = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode()
    _atomic_write_bytes(Path(path), lambda handle: handle.write(text))


def read_manifest(directory: Path) -> dict[str, Any]:
    """Load a sample-set manifest."""
    return json.loads((Path(directory) / MANIFEST_NAME).read_text())


def missing_blocks(directory: Path, num_blocks: int) -> list[int]:
    """Blocks whose shard file does not exist yet."""
    return [block for block in range(num_blocks) if not shard_path(directory, block).exists()]


def iter_shards(directory: Path) -> Iterator[Shard]:
    """Yield every shard in block order after checking the set is complete."""
    manifest = read_manifest(directory)
    num_blocks = int(manifest['plan']['num_blocks'])
    missing = missing_blocks(directory, num_blocks)
    if missing:
        raise FileNotFoundError(
            f'{directory} is incomplete: {len(missing)} missing shards (first {missing[:5]})'
        )
    for block in range(num_blocks):
        yield read_shard(shard_path(directory, block))


def find_sample_sets(eval_dir: Path) -> list[Path]:
    """Every sample-set directory (one containing a manifest) under ``eval_dir``."""
    return sorted(path.parent for path in Path(eval_dir).glob(f'*/*/nfe_*/{MANIFEST_NAME}'))
