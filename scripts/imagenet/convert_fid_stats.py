"""Convert DMF's ``imgnet256.pkl`` FID statistics to the torchmetrics format.

Writes the Inception mean/covariance as the sum, covariance-sum and count that
``torchmetrics.image.fid.FrechetInceptionDistance`` stores for the real images.

Usage::

    uv run python -m scripts.imagenet.convert_fid_stats \
        --pkl ckpt/imgnet256.pkl --out ckpt/imgnet256_torchmetrics.pt
"""

from __future__ import annotations

import argparse
import pickle  # noqa: S403 -- trusted reference statistics from kyungmnlee/DMF
from pathlib import Path

import torch


def main() -> None:
    """Convert one statistics file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    with args.pkl.open('rb') as handle:
        stats = pickle.load(handle)  # noqa: S301
    n = int(stats['num_images'])
    mu = torch.as_tensor(stats['fid']['mu'], dtype=torch.float64)
    sigma = torch.as_tensor(stats['fid']['sigma'], dtype=torch.float64)
    torch.save(
        {
            'real_features_sum': mu * n,
            'real_features_cov_sum': sigma * (n - 1) + n * torch.outer(mu, mu),
            'real_features_num_samples': torch.tensor(n),
            'metadata': {
                'protocol': 'tf-fidelity-native-uint8-v1',
                'feature': 2048,
                'input': 'native-resolution uint8 [0,255]',
                'normalize': False,
                'antialias': False,
            },
        },
        args.out,
    )
    print(f'Wrote {args.out} ({n} images)')


if __name__ == '__main__':
    main()
