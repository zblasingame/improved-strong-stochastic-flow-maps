"""Atomic sharded storage for precomputed VAE posterior moments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

LATENTS_SCALE = 0.18215

@dataclass(frozen=True)
class ShardEntry:
    """Metadata for one finalized latent shard."""

    filename: str
    count: int
    checksum: str


@dataclass(frozen=True)
class Manifest:
    """Versioned metadata describing a complete latent split."""

    schema_version: int
    vae_name: str
    image_size: int
    latent_shape: tuple[int, ...]
    crop_mode: str
    latents_scale: float
    split: str
    shards: list[ShardEntry]
    total_count: int


def shard_filename(shard_id: int) -> str:
    """Return the deterministic filename for a non-negative shard ID."""
    if shard_id < 0:
        raise ValueError('shard_id must be non-negative')
    return f'shard_{shard_id:05d}.safetensors'


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    _fsync(temp_path)
    os.replace(str(temp_path), str(final_path))


def write_shard(
    path: str | Path,
    moments: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    global_indices: np.ndarray | torch.Tensor,
) -> str:
    """Atomically write one safetensors shard and return its SHA256 checksum."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        'moments': torch.as_tensor(moments, dtype=torch.float16).contiguous(),
        'labels': torch.as_tensor(labels, dtype=torch.int64).contiguous(),
        'global_indices': torch.as_tensor(global_indices, dtype=torch.int64).contiguous(),
    }
    counts = {tensor.shape[0] for tensor in tensors.values()}
    if len(counts) != 1:
        raise ValueError('moments, labels, and global_indices must have equal lengths')

    temp_path = final_path.with_suffix(final_path.suffix + '.tmp')
    try:
        save_file(tensors, str(temp_path))
        _atomic_replace(temp_path, final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return _checksum(final_path)


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    """Atomically serialize a manifest as JSON."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + '.tmp')
    payload = asdict(manifest)
    payload['latent_shape'] = list(manifest.latent_shape)
    try:
        with temp_path.open('w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temp_path), str(final_path))
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def load_manifest(path: str | Path) -> Manifest:
    """Load a latent manifest from JSON."""
    with Path(path).open(encoding='utf-8') as stream:
        payload: dict[str, Any] = json.load(stream)
    return Manifest(
        schema_version=int(payload['schema_version']),
        vae_name=str(payload['vae_name']),
        image_size=int(payload['image_size']),
        latent_shape=tuple(int(value) for value in payload['latent_shape']),
        crop_mode=str(payload['crop_mode']),
        latents_scale=float(payload['latents_scale']),
        split=str(payload['split']),
        shards=[ShardEntry(**entry) for entry in payload['shards']],
        total_count=int(payload['total_count']),
    )


def _validate_entry(root: Path, entry: ShardEntry) -> torch.Tensor:
    path = root / entry.filename
    if not path.is_file():
        raise ValueError(f'missing shard file: {entry.filename}')
    if _checksum(path) != entry.checksum:
        raise ValueError(f'checksum mismatch for shard: {entry.filename}')
    tensors = load_file(str(path))
    required = {'moments', 'labels', 'global_indices'}
    if set(tensors) != required:
        raise ValueError(f'invalid tensors in shard: {entry.filename}')
    counts = {tensor.shape[0] for tensor in tensors.values()}
    if counts != {entry.count}:
        raise ValueError(f'count mismatch for shard: {entry.filename}')
    return tensors['global_indices'].to(torch.int64)


def validate_manifest(path: str | Path) -> Manifest:
    """Validate all manifest counts, checksums, and global indices."""
    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    indices = [_validate_entry(manifest_path.parent, entry) for entry in manifest.shards]
    all_indices = torch.cat(indices) if indices else torch.empty(0, dtype=torch.int64)
    unique = torch.unique(all_indices)
    if unique.numel() != all_indices.numel():
        raise ValueError('duplicate global indices in latent shards')
    expected = torch.arange(manifest.total_count, dtype=torch.int64)
    if not torch.equal(torch.sort(unique).values, expected):
        raise ValueError('global index range has gaps or out-of-range values')
    actual_count = sum(entry.count for entry in manifest.shards)
    if actual_count != manifest.total_count:
        raise ValueError('manifest total count does not equal shard counts')
    return manifest
