"""Encode cached Hugging Face ImageNet into resumable DMF latent shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.data.components.adm_crop import center_crop_imagenet
from src.data.components.latent_store import (
    LATENTS_SCALE,
    Manifest,
    ShardEntry,
    load_manifest,
    shard_filename,
    validate_manifest,
    write_manifest,
    write_shard,
)
from src.data.components.vae_encoder import DEFAULT_VAE_NAME, encode_moments, load_stability_vae


def _images_uint8(dataset: Any, indices: range, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index in indices:
        example = dataset[index]
        images.append(center_crop_imagenet(image_size, np.asarray(example['image'].convert('RGB'))))
        label = int(example['label'])
        if not 0 <= label < 1000:
            raise ValueError(f'ImageNet label must be in [0, 999]; got {label}.')
        labels.append(label)
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return batch, np.asarray(labels, dtype=np.int64)


def encode_dataset(
    dataset: Any,
    output_dir: str | Path,
    vae: torch.nn.Module,
    *,
    split: str,
    image_size: int = 256,
    shard_size: int = 1024,
    batch_size: int = 16,
    max_images: int | None = None,
    device: str = 'cuda',
    vae_name: str = DEFAULT_VAE_NAME,
) -> Manifest:
    """Encode a dataset split, updating an atomic manifest after every shard."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    shards: list[ShardEntry] = []
    start = 0
    if manifest_path.exists():
        previous = validate_manifest(manifest_path)
        if previous.vae_name != vae_name or previous.split != split:
            raise ValueError('existing manifest is incompatible with this encoding request')
        shards = list(previous.shards)
        start = previous.total_count
    total = min(len(dataset), max_images) if max_images is not None else len(dataset)
    if start > total:
        raise ValueError('existing manifest contains more samples than requested')

    for begin in tqdm(range(start, total, shard_size), desc=f'encode {split}'):
        end = min(begin + shard_size, total)
        images, labels = _images_uint8(dataset, range(begin, end), image_size)
        moments = encode_moments(images, vae, batch_size=batch_size, device=device).cpu()
        expected_shape = (end - begin, 8, image_size // 8, image_size // 8)
        if tuple(moments.shape) != expected_shape:
            raise ValueError(f'VAE moments shape {tuple(moments.shape)} != {expected_shape}')
        filename = shard_filename(len(shards))
        checksum = write_shard(
            output / filename, moments, labels, np.arange(begin, end, dtype=np.int64)
        )
        shards.append(ShardEntry(filename, end - begin, checksum))
        manifest = Manifest(
            schema_version=1,
            vae_name=vae_name,
            image_size=image_size,
            latent_shape=(8, image_size // 8, image_size // 8),
            crop_mode='center-crop-dhariwal',
            latents_scale=LATENTS_SCALE,
            split=split,
            shards=shards,
            total_count=end,
        )
        write_manifest(manifest_path, manifest)
    result = validate_manifest(manifest_path) if manifest_path.exists() else load_manifest(manifest_path)
    print(f'Encoded {result.total_count} images in {len(result.shards)} shards at {output}')
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--shard-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--vae-name', default=DEFAULT_VAE_NAME)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    dataset = load_dataset(
        'ILSVRC/imagenet-1k',
        split=args.split,
        cache_dir=args.cache_dir,
        token=True,
        download_mode='reuse_dataset_if_exists',
    )
    vae = load_stability_vae(args.vae_name, args.device)
    encode_dataset(
        dataset,
        output / args.split,
        vae,
        split=args.split,
        max_images=args.max_images,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        device=args.device,
        vae_name=args.vae_name,
    )


if __name__ == '__main__':
    main()
