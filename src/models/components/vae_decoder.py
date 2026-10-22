"""Frozen Stable Diffusion VAE decoder for DMF-compatible latent samples."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor

from src.data.components.latent_store import LATENTS_SCALE
from src.data.components.vae_encoder import DEFAULT_VAE_NAME, load_stability_vae


@lru_cache(maxsize=4)
def _load_cached_vae(vae_name: str, cache_dir: str | None) -> nn.Module:
    """Load one frozen VAE per process and model/cache configuration."""
    return load_stability_vae(vae_name, device='cpu', cache_dir=cache_dir)


class StabilityVAEDecoder(nn.Module):
    """Decode scaled four-channel DMF latents into pixel images in ``[0, 1]``."""

    def __init__(
        self,
        vae_name: str = DEFAULT_VAE_NAME,
        cache_dir: str | None = None,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f'chunk_size must be >= 1; got {chunk_size}.')
        self.vae = _load_cached_vae(vae_name, cache_dir)
        self.chunk_size = chunk_size

    def _decode_chunk(self, latents: Tensor) -> Tensor:
        """Decode one latent chunk and map it into ``[0, 1]``."""
        decoded = self.vae.decode(latents / LATENTS_SCALE)
        # diffusers returns a dataclass output; older/patched paths return a mapping.
        sample = decoded.sample if hasattr(decoded, 'sample') else decoded['sample']
        return sample.float().add(1.0).mul(0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, latents: Tensor) -> Tensor:
        """Decode latents after undoing the checkpoint-matched scale factor.

        Parameters
        ----------
        latents : torch.Tensor
            Scaled four-channel latents of shape ``(B, 4, H, W)``.

        Returns
        -------
        torch.Tensor
            Float32 images of shape ``(B, 3, 8H, 8W)`` in ``[0, 1]``.
        """
        if latents.shape[0] <= self.chunk_size:
            return self._decode_chunk(latents)
        return torch.cat(
            [self._decode_chunk(chunk) for chunk in latents.split(self.chunk_size)],
            dim=0,
        )
