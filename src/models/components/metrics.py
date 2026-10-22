"""Image-sample logging and FID metrics with multi-profile evaluation.

- :func:`array2grid` tiles a batch of images into a single uint8 ``HxWx3`` array
  ready for ``wandb.Image``.
- :func:`bin_loss` bins per-sample losses by time for diagnostic logging.
- :class:`FIDComputer` wraps torchmetrics' Frechet Inception Distance.
- :class:`HistoricalFIDComputer` reproduces the legacy ``torchvision``-Inception
  protocol (uint8 truncation plus antialiased bicubic ``299x299`` resize) with
  streaming sufficient statistics instead of retained features.
- :func:`evaluate_fid_profiles` computes one FID per named model profile (each
  post-hoc EMA ``sigma_rel`` and, optionally, the live non-EMA model), returning
  a flat dict ready to merge into a wandb / Lightning log dict.

Two mutually incompatible protocols coexist here; statistics are tagged with the
protocol that produced them and are never mixed across that boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import make_grid

FID_PROTOCOL = 'tf-fidelity-native-uint8-v1'
HISTORICAL_FID_PROTOCOL = 'torchvision-bicubic-299-v1'

#: Inception pool3 dimensionality used by the historical protocol.
HISTORICAL_FID_FEATURE_DIM = 2048

#: Spatial resolution the historical protocol resizes to before feature extraction.
HISTORICAL_FID_RESOLUTION = 299


def array2grid(x: Tensor) -> np.ndarray:
    """Tile a batch of images into a single uint8 grid for logging.

    :param x: Images of shape ``(B, C, H, W)`` with values in ``[0, 1]``.
    :return: A ``uint8`` numpy array of shape ``(H', W', C)``.
    """
    nrow = round(math.sqrt(x.size(0)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()


def to_uint8_images(x: Tensor) -> Tensor:
    """Map images from the model's ``[-1, 1]`` range to ``[0, 1]`` floats.

    FID with ``normalize=True`` expects inputs in ``[0, 1]``.

    :param x: Images of shape ``(B, C, H, W)`` in ``[-1, 1]``.
    :return: Images of shape ``(B, C, H, W)`` clamped to ``[0, 1]``.
    """
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def to_fid_uint8(x: Tensor) -> Tensor:
    """Convert model-range images to native-resolution uint8 FID inputs."""
    return (x * 0.5 + 0.5).mul(255).round().clamp(0, 255).to(torch.uint8)


def preprocess_historical_fid(x: Tensor) -> Tensor:
    """Apply the calibrated historical FID image preprocessing.

    :param x: RGB images in the model's ``[-1, 1]`` range.
    :return: Float images in ``[0, 1]`` resized to ``299x299``.
    """
    if not x.is_floating_point():
        raise TypeError(f'historical FID inputs must be floating point, got {x.dtype}')
    uint8 = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = uint8.to(dtype=torch.float32).div(255.0)
    return F.interpolate(
        images,
        size=(HISTORICAL_FID_RESOLUTION, HISTORICAL_FID_RESOLUTION),
        mode='bicubic',
        antialias=True,
    ).clamp(0.0, 1.0)


class _FeatureMoments:
    """Streaming feature sufficient statistics in float64 precision."""

    def __init__(self, feature: int) -> None:
        self.feature = feature
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated feature statistics."""
        self.count = 0
        self.sum = torch.zeros(self.feature, dtype=torch.float64)
        self.cross_sum = torch.zeros((self.feature, self.feature), dtype=torch.float64)

    def update(self, features: Tensor) -> None:
        """Accumulate a two-dimensional feature batch on CPU."""
        values = features.detach().to(device='cpu', dtype=torch.float64)
        if values.ndim != 2 or values.shape[1] != self.feature:
            raise ValueError(
                f'expected features with shape (N, {self.feature}), got {values.shape}'
            )
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.cross_sum += values.T @ values

    def moments(self) -> tuple[Tensor, Tensor, int]:
        """Return sample mean, unbiased covariance, and sample count."""
        if self.count < 2:
            raise ValueError('at least two feature samples are required')
        mean = self.sum / self.count
        covariance = (self.cross_sum - self.count * torch.outer(mean, mean)) / (self.count - 1)
        return mean, covariance, self.count


class HistoricalFIDComputer:
    """Streaming FID using the calibrated historical torchvision protocol."""

    def __init__(
        self,
        device: torch.device | None = None,
        extractor: nn.Module | None = None,
        feature: int = HISTORICAL_FID_FEATURE_DIM,
    ) -> None:
        if extractor is None:
            from torchvision import models

            extractor = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            extractor.fc = nn.Identity()
        self.device = device or torch.device('cpu')
        self.feature = feature
        self.extractor = extractor.eval().to(self.device)
        self.real = _FeatureMoments(feature)
        self.fake = _FeatureMoments(feature)

    @torch.inference_mode()
    def _features(self, images: Tensor) -> Tensor:
        return self.extractor(preprocess_historical_fid(images.to(self.device)))

    def update_real(self, images: Tensor) -> None:
        """Accumulate features for real model-range images."""
        self.real.update(self._features(images))

    def update_fake(self, images: Tensor) -> None:
        """Accumulate features for generated model-range images."""
        self.fake.update(self._features(images))

    def reset_fake(self) -> None:
        """Clear generated-image statistics while preserving real statistics."""
        self.fake.reset()

    def real_stats(self) -> dict[str, Any]:
        """Return protocol-tagged real feature moments."""
        mean, covariance, count = self.real.moments()
        return {
            'mu': mean,
            'sigma': covariance,
            'n_samples': count,
            'metadata': {'protocol': HISTORICAL_FID_PROTOCOL, 'feature': self.feature},
        }

    def load_real_stats(self, stats: Mapping[str, Any]) -> None:
        """Load protocol-compatible real moments."""
        metadata = stats.get('metadata')
        if (
            not isinstance(metadata, Mapping)
            or metadata.get('protocol') != HISTORICAL_FID_PROTOCOL
        ):
            raise ValueError(
                'Incompatible historical FID reference statistics; regenerate them with '
                'scripts/compute_historical_fid_stats.py.'
            )
        for key in ('mu', 'sigma', 'n_samples'):
            if key not in stats:
                raise KeyError(f'missing real-statistics key: {key!r}')
        mean = torch.as_tensor(stats['mu'], dtype=torch.float64)
        covariance = torch.as_tensor(stats['sigma'], dtype=torch.float64)
        count = int(stats['n_samples'])
        if mean.shape != (self.feature,) or covariance.shape != (self.feature, self.feature):
            raise ValueError(f'historical FID stats do not match feature dimension {self.feature}')
        self.real.count = count
        self.real.sum = mean * count
        self.real.cross_sum = covariance * (count - 1) + count * torch.outer(mean, mean)

    def compute(self) -> float:
        """Return FID between loaded real and accumulated fake moments."""
        from scipy import linalg

        real_mean, real_covariance, _ = self.real.moments()
        fake_mean, fake_covariance, _ = self.fake.moments()
        mean_term = torch.square(real_mean - fake_mean).sum().item()
        covariance_root = linalg.sqrtm(real_covariance.numpy() @ fake_covariance.numpy())
        if np.iscomplexobj(covariance_root):
            covariance_root = covariance_root.real
        covariance_term = np.trace(
            real_covariance.numpy() + fake_covariance.numpy() - 2.0 * covariance_root
        )
        return float(mean_term + covariance_term)


def bin_loss(prefix: str, t: Tensor, mse: Tensor, n_bins: int = 10) -> dict[str, float]:
    """Bin per-sample losses by time ``t`` into scalar diagnostics.

    :param prefix: Log-key prefix, e.g. ``"train/fm"``.
    :param t: Per-sample times of shape ``(B,)`` in ``[0, 1]``.
    :param mse: Per-sample losses of shape ``(B,)``.
    :param n_bins: Number of equal-width time bins.
    :return: Flat dict ``{f"{prefix}_loss/bin_t_(lo,hi]": mean_loss}``.
    """
    logs: dict[str, float] = {}
    if t.numel() == 0:
        return logs
    t, mse = t.detach().cpu(), mse.detach().cpu()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    idx = torch.bucketize(t, edges[1:]).clamp(max=n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            lo, hi = edges[b].item(), edges[b + 1].item()
            logs[f'{prefix}_loss/bin_t_({lo:.1f},{hi:.1f}]'] = mse[mask].mean().item()
    return logs


class FIDComputer:
    """Thin wrapper around torchmetrics' Frechet Inception Distance.

    The Inception feature extractor expects ``299x299`` RGB images in ``[0, 1]``;
    inputs are resized and range-mapped here. Real reference statistics can be
    supplied once via :meth:`update_real` (or precomputed buffers loaded into the
    underlying metric) and reused across many fake-sample evaluations.

    Parameters
    ----------
    feature : int, optional
        Inception feature dimensionality (default ``2048``).
    device : torch.device, optional
        Device to place the metric on.
    """

    #: torchmetrics FID state buffers holding the real-image reference statistics.
    REAL_STATE_KEYS = (
        'real_features_sum',
        'real_features_cov_sum',
        'real_features_num_samples',
    )

    def __init__(self, feature: int = 2048, device: torch.device | None = None) -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance

        # reset_real_features=False so that reset() (and thus reset_fake) preserves
        # the precomputed real reference statistics while clearing only the fake
        # accumulators -- letting one real reference be reused across many profiles.
        self.feature = feature
        self.metric = FrechetInceptionDistance(
            feature=feature,
            normalize=False,
            reset_real_features=False,
            antialias=False,
        )
        self.device = device
        if device is not None:
            self.metric = self.metric.to(device)

    @staticmethod
    def _prep(images: Tensor) -> Tensor:
        """Validate native-resolution uint8 inputs for the FID extractor."""
        if images.dtype != torch.uint8:
            raise TypeError(f'FID inputs must have dtype torch.uint8, got {images.dtype}')
        return images

    def update_real(self, images: Tensor) -> None:
        """Accumulate real-image statistics (``images`` in ``[0, 1]``)."""
        self.metric.update(self._prep(images), real=True)

    def update_fake(self, images: Tensor) -> None:
        """Accumulate generated-image statistics (``images`` in ``[0, 1]``)."""
        self.metric.update(self._prep(images), real=False)

    def compute(self) -> float:
        """Return the current FID score as a Python float."""
        return float(self.metric.compute().item())

    def reset_fake(self) -> None:
        """Reset only the fake-image statistics, preserving the real reference.

        Relies on ``reset_real_features=False`` (set in :meth:`__init__`) so that
        the underlying ``reset`` clears the fake accumulators but leaves the real
        reference statistics untouched.
        """
        self.metric.reset()

    def real_stats(self) -> dict[str, Tensor]:
        """Return a CPU copy of the accumulated real reference statistics.

        :return: Dict with keys ``real_features_sum``, ``real_features_cov_sum``
            and ``real_features_num_samples`` ready to ``torch.save``.
        """
        stats = {
            key: getattr(self.metric, key).detach().clone().cpu() for key in self.REAL_STATE_KEYS
        }
        stats['metadata'] = {
            'protocol': FID_PROTOCOL,
            'feature': self.feature,
            'input': 'native-resolution uint8 [0,255]',
            'normalize': False,
            'antialias': False,
        }
        return stats

    def load_real_stats(self, stats: Mapping[str, Tensor]) -> None:
        """Inject precomputed real reference statistics into the metric.

        Mirrors the DMF pattern of loading ``real_features_*`` buffers from a
        precomputed file, avoiding a full pass over the real dataset at eval time.

        :param stats: Mapping containing the three ``real_features_*`` buffers, as
            produced by :meth:`real_stats` (or a compatible preprocessing script).
        :raises KeyError: If any required real-statistics key is missing.
        """
        metadata = stats.get('metadata')
        if not isinstance(metadata, Mapping) or metadata.get('protocol') != FID_PROTOCOL:
            raise ValueError(
                'Incompatible FID reference statistics; regenerate them with '
                'scripts/compute_fid_stats.py.'
            )
        for key in self.REAL_STATE_KEYS:
            if key not in stats:
                raise KeyError(f'missing real-statistics key: {key!r}')
            buffer = getattr(self.metric, key)
            value = stats[key].to(device=buffer.device, dtype=buffer.dtype)
            buffer.copy_(value)


def profile_name(std: float) -> str:
    """Return the canonical log-name for a post-hoc EMA ``sigma_rel`` profile.

    :param std: The profile's ``sigma_rel``.
    :return: A key fragment like ``"sigma_0.050"``.
    """
    return f'sigma_{std:.3f}'
