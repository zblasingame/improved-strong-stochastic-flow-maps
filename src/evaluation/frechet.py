"""Fréchet distance on streaming float64 feature moments, plus the DINOv2 extractor.

FD-DINOv2 follows EDM2's ``DINOv2Detector`` (modeled on dgm-eval): DINOv2
ViT-L/14 CLS features (1024-d) of uint8 images resized to 224x224 with
antialiased bicubic interpolation and ImageNet mean/std normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DINOV2_PROTOCOL = 'dinov2-vitl14-cls-bicubic224-antialias-v1'
DINOV2_FEATURE_DIM = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class MomentAccumulator:
    """Streaming float64 sum / cross-sum of feature vectors."""

    dim: int
    count: int = 0
    total: np.ndarray | None = None
    cross: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Allocate zeroed accumulators when not provided."""
        self.total = np.zeros(self.dim, np.float64) if self.total is None else self.total
        self.cross = (
            np.zeros((self.dim, self.dim), np.float64) if self.cross is None else self.cross
        )

    def update(self, features: Tensor | np.ndarray) -> None:
        """Accumulate an ``(N, dim)`` feature batch."""
        values = torch.as_tensor(features).detach().to('cpu', torch.float64).numpy()
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f'expected (N, {self.dim}) features, got {values.shape}')
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.cross += values.T @ values

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample mean and unbiased covariance."""
        if self.count < 2:
            raise ValueError('at least two samples are required')
        mean = self.total / self.count
        cov = (self.cross - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, cov


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """``||mu1 - mu2||^2 + tr(S1 + S2 - 2 sqrtm(S1 S2))`` (real part of sqrtm)."""
    from scipy import linalg

    root = np.real(linalg.sqrtm(sigma1 @ sigma2))
    return float(np.square(mu1 - mu2).sum() + np.trace(sigma1 + sigma2 - 2.0 * root))


def save_reference(path: Path, acc: MomentAccumulator, protocol: str, **meta) -> None:  # noqa: ANN003
    """Persist reference moments (mean / covariance / count) with a protocol tag."""
    mean, cov = acc.moments()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {'mu': mean, 'sigma': cov, 'num_samples': acc.count, 'protocol': protocol, **meta}, path
    )


def load_reference(path: Path, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    """Load reference moments, refusing a mismatched feature protocol."""
    ref = torch.load(path, map_location='cpu', weights_only=False)
    if ref.get('protocol') != protocol:
        raise ValueError(f'{path} has protocol {ref.get("protocol")!r}, expected {protocol!r}')
    return np.asarray(ref['mu'], np.float64), np.asarray(ref['sigma'], np.float64)


class DINOv2Features(nn.Module):
    """DINOv2 ViT-L/14 CLS-feature extractor for NCHW uint8 RGB images."""

    def __init__(self, model: nn.Module | None = None) -> None:
        super().__init__()
        if model is None:
            model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitl14',
                trust_repo=True,
                verbose=False,
                skip_validation=True,
            )
        self.model = model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.inference_mode()
    def forward(self, images: Tensor) -> Tensor:
        """Map uint8 ``(N, 3, H, W)`` images to float32 ``(N, 1024)`` features."""
        if images.dtype != torch.uint8:
            raise TypeError(f'DINOv2 inputs must be uint8, got {images.dtype}')
        x = F.interpolate(images.float(), size=(224, 224), mode='bicubic', antialias=True)
        x = (x / 255.0 - self.mean) / self.std
        return self.model(x).float()
