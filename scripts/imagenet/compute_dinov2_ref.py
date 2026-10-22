"""One-time FD-DINOv2 reference statistics for ImageNet-256 (full train split).

Reads the locally cached Hugging Face ``ILSVRC/imagenet-1k`` parquet shards,
applies the ADM/Dhariwal center crop to 256x256 (the crop used for the latent
store and the FID reference), and accumulates float64 DINOv2 ViT-L/14 CLS
moments. Decoding runs in DataLoader workers over parquet row groups.

Example::

    uv run python -m scripts.imagenet.compute_dinov2_ref \
        --output $DINO_REF_STATS
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import rootutils
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.data.components.adm_crop import center_crop_imagenet  # noqa: E402
from src.evaluation.frechet import (  # noqa: E402
    DINOV2_FEATURE_DIM,
    DINOV2_PROTOCOL,
    DINOv2Features,
    MomentAccumulator,
    save_reference,
)

DEFAULT_HF_DATA = Path.home() / '.cache/huggingface/hub/datasets--ILSVRC--imagenet-1k/snapshots'


class ParquetImages(IterableDataset):
    """Yield ADM-cropped uint8 CHW tensors from parquet row groups, split over workers."""

    def __init__(self, files: list[Path], image_size: int = 256, limit: int | None = None) -> None:
        import pyarrow.parquet as pq

        self.groups = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
        self.image_size = image_size
        self.limit = limit

    def __iter__(self):
        """Yield crops from this worker's share of row groups."""
        import pyarrow.parquet as pq
        from PIL import Image

        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        emitted = 0
        for path, group in self.groups[worker::workers]:
            table = pq.ParquetFile(path).read_row_group(group, columns=['image'])
            for record in table.column('image').to_pylist():
                if self.limit is not None and emitted >= self.limit:
                    return
                rgb = np.asarray(Image.open(io.BytesIO(record['bytes'])).convert('RGB'))
                crop = center_crop_imagenet(self.image_size, rgb)
                emitted += 1
                yield torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)


def main() -> None:
    """Accumulate DINOv2 reference moments over the ImageNet train split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_HF_DATA)
    parser.add_argument('--split', default='train')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--limit-per-worker', type=int, default=None, help='Smoke-test cap')
    args = parser.parse_args()

    files = sorted(args.data_root.glob(f'*/data/{args.split}-*.parquet'))
    if not files:
        raise FileNotFoundError(f'no {args.split} parquet shards under {args.data_root}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = DINOv2Features().to(device)
    acc = MomentAccumulator(DINOV2_FEATURE_DIM)
    loader = DataLoader(
        ParquetImages(files, limit=args.limit_per_worker),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    for step, batch in enumerate(loader):
        acc.update(dino(batch.to(device, non_blocking=True)))
        if step % 200 == 0:
            print(f'{acc.count} images', flush=True)
    save_reference(
        args.output,
        acc,
        DINOV2_PROTOCOL,
        dataset=f'ILSVRC/imagenet-1k {args.split}',
        crop='center-crop-dhariwal-256',
    )
    print(f'saved {acc.count}-image reference to {args.output}')


if __name__ == '__main__':
    main()
