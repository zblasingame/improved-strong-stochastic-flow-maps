"""Frozen ``sd-vae-ft-mse`` posterior-moment encoder.

This module mirrors the DMF preprocessing reference (``encoders.py``,
``StabilityVAEEncoder`` / ``load_stability_vae``) so that latents produced here
are numerically compatible with the DMF checkpoints.

Two properties are load-bearing and must not drift:

* **Normalization.** Raw ``uint8`` pixels are mapped with ``x / 127.5 - 1``
  (``[0, 255] -> [-1, 1]``). The common ``x / 255`` variant would shift every
  latent statistic away from the checkpoint.
* **Moment layout.** The encoder stores ``torch.cat([mean, std], dim=1)`` -- the
  *posterior moments* -- rather than a single sampled latent, so downstream
  consumers can resample ``z = mean + std * eps`` every epoch. Channels
  ``[:4]`` are the mean and ``[4:]`` are the standard deviation.

The VAE is always forced into inference mode (``eval()`` +
``requires_grad_(False)``): it is a frozen feature extractor, never a trainable
part of the model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_VAE_NAME = 'stabilityai/sd-vae-ft-mse'
DEFAULT_BATCH_SIZE = 8

# Pixel normalization constant: uint8 [0, 255] -> float [-1, 1]. Matches the DMF
# reference exactly; do not replace with 255.0.
_UINT8_HALF_RANGE = 127.5


def load_stability_vae(
    vae_name: str = DEFAULT_VAE_NAME,
    device: torch.device | str = 'cpu',
    cache_dir: str | None = None,
) -> nn.Module:
    """Load the frozen Stability ``AutoencoderKL`` used by the DMF checkpoints.

    Resolution is attempted with ``local_files_only=True`` first so that an
    already-cached checkpoint is used without any network round-trip; only if
    that fails is a remote fetch attempted.

    The checkpoint location is left to the Hugging Face Hub defaults, which read
    ``HF_HOME`` / ``HF_HUB_CACHE`` / ``TRANSFORMERS_CACHE`` from the environment.
    Nothing is ever written into the repository tree, and no environment variable
    is mutated as a side effect.

    Parameters
    ----------
    vae_name : str, optional
        Hugging Face model id, by default ``'stabilityai/sd-vae-ft-mse'``.
        Changing this breaks checkpoint compatibility with DMF.
    device : torch.device or str, optional
        Device to place the model on, by default ``'cpu'``.
    cache_dir : str or None, optional
        Explicit Hugging Face cache directory. When ``None`` (the default), the
        Hub resolves the cache from the standard environment variables.

    Returns
    -------
    torch.nn.Module
        The ``AutoencoderKL`` in eval mode with all parameters frozen
        (``requires_grad=False``) and moved to ``device``.

    Raises
    ------
    ImportError
        If the optional ``diffusers`` dependency is not installed.
    """
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            'load_stability_vae requires the `diffusers` package; install it with '
            '`uv sync` (it is a declared project dependency).'
        ) from error

    options: dict[str, object] = {}
    if cache_dir is not None:
        options['cache_dir'] = cache_dir

    try:
        # Prefer the local cache: this avoids a Hub metadata request entirely,
        # which matters on offline compute nodes.
        vae = AutoencoderKL.from_pretrained(vae_name, local_files_only=True, **options)
    except (OSError, ValueError):
        # Not (fully) cached -- OSError covers huggingface_hub's
        # LocalEntryNotFoundError/FileNotFoundError, ValueError covers a stale or
        # partial cache entry. Fall back to a remote fetch.
        vae = AutoencoderKL.from_pretrained(vae_name, **options)

    return freeze_vae(vae).to(device)


def freeze_vae(vae: nn.Module) -> nn.Module:
    """Put a VAE into inference mode in place and return it.

    Parameters
    ----------
    vae : torch.nn.Module
        The autoencoder to freeze.

    Returns
    -------
    torch.nn.Module
        The same module, in ``eval()`` mode with ``requires_grad=False`` on every
        parameter.
    """
    return vae.eval().requires_grad_(False)


def _encode_batch_moments(vae: nn.Module, batch: Tensor) -> Tensor:
    """Encode one normalized micro-batch into ``[mean, std]`` moments.

    Parameters
    ----------
    vae : torch.nn.Module
        A frozen ``AutoencoderKL``-compatible module whose ``encode`` returns a
        mapping containing a ``'latent_dist'`` with ``.mean`` and ``.std``.
    batch : torch.Tensor
        Normalized image batch of shape ``(b, 3, H, W)`` in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        Moment tensor of shape ``(b, 8, H / 8, W / 8)``.
    """
    latent_dist = vae.encode(batch)['latent_dist']
    return torch.cat([latent_dist.mean, latent_dist.std], dim=1)


def encode_moments(
    images_uint8: Tensor,
    vae: nn.Module,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode ``uint8`` images into frozen-VAE posterior moments.

    Pixels are normalized with ``x / 127.5 - 1``, split into micro-batches, and
    pushed through the VAE encoder under ``torch.no_grad()``. Micro-batching is
    purely a memory-management concern: the result is bit-identical for any
    ``batch_size`` because samples are neither reordered nor mixed.

    Parameters
    ----------
    images_uint8 : torch.Tensor
        Raw pixel batch of shape ``(B, 3, H, W)``, dtype ``uint8`` in ``[0, 255]``
        and NCHW/RGB order (see :func:`src.data.components.adm_crop.to_rgb_uint8`).
    vae : torch.nn.Module
        Frozen ``AutoencoderKL``-compatible encoder. It is forced into eval mode
        with ``requires_grad=False`` before use, so a trainable module passed in
        by mistake cannot leak gradients.
    batch_size : int, optional
        Micro-batch size for the encoder forward pass, by default ``8``.
    device : torch.device or str or None, optional
        Device to run the encoder on. When ``None`` (the default) the inputs are
        left on their current device.

    Returns
    -------
    torch.Tensor
        Posterior moments of shape ``(B, 8, H / 8, W / 8)``, laid out as
        ``torch.cat([mean, std], dim=1)``: channels ``[:4]`` are the posterior
        mean and channels ``[4:]`` the strictly positive standard deviation.

    Raises
    ------
    ValueError
        If ``images_uint8`` is not a 4D NCHW tensor, or ``batch_size`` is not
        positive.
    """
    if images_uint8.ndim != 4:
        raise ValueError(
            f'images_uint8 must be a 4D NCHW tensor, got shape {tuple(images_uint8.shape)}.'
        )
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}.')

    freeze_vae(vae)

    images = images_uint8 if device is None else images_uint8.to(device)
    normalized = images.to(torch.float32) / _UINT8_HALF_RANGE - 1

    with torch.no_grad():
        moments = [_encode_batch_moments(vae, batch) for batch in normalized.split(batch_size)]

    return torch.cat(moments, dim=0)
