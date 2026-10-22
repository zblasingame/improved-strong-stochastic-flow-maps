"""Exact ADM/Dhariwal ImageNet center crop.

This module reproduces, bit-for-bit, the ``center-crop-dhariwal`` transform used
by the DMF preprocessing reference (``dataset_tools.py``, ``center_crop_imagenet``),
which itself is transcribed from ADM/guided-diffusion::

    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

The algorithm is deliberately *not* a torchvision ``Resize``/``CenterCrop``
pipeline: the reference repeatedly halves the image with PIL's ``BOX`` filter
while the shorter edge is at least ``2 * image_size``, then does a single
``BICUBIC`` resize so the shorter edge lands exactly on ``image_size``, and only
then takes the center slice. Substituting a different resampling path (e.g. a
direct ``BICUBIC`` resize to ``256 x 256``) produces different pixel values and
would silently shift the VAE latent statistics away from the DMF checkpoint.

Notes
-----
Both helpers here are pure functions: the input array is never mutated, and the
same input always yields the same output.
"""

import numpy as np
import PIL.Image


def to_rgb_uint8(image: PIL.Image.Image) -> np.ndarray:
    """Convert a PIL image to an HWC ``uint8`` RGB array.

    This is the boundary conversion applied before :func:`center_crop_imagenet`.
    It mirrors the reference loaders, which call ``np.array(PIL.Image.open(...).convert('RGB'))``
    so that palette, grayscale, CMYK, and RGBA source images all collapse to
    three 8-bit channels before cropping.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image in any PIL mode.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W, 3)`` and dtype ``uint8`` in RGB channel order.
    """
    return np.array(image.convert('RGB'))


def center_crop_imagenet(image_size: int, arr: np.ndarray) -> np.ndarray:
    """Apply the ADM/Dhariwal center crop to a HWC ``uint8`` image array.

    The three stages are executed exactly as in the reference implementation:

    1. While the shorter edge is at least ``2 * image_size``, halve both edges
       with ``Resampling.BOX`` (integer floor division on each edge).
    2. Resize once with ``Resampling.BICUBIC`` to ``round(x * scale)`` per edge,
       where ``scale = image_size / min(size)``.
    3. Take the center ``image_size x image_size`` slice, with the offsets
       computed by floor division.

    Parameters
    ----------
    image_size : int
        Target square output resolution (e.g. ``256``). Must be positive.
    arr : np.ndarray
        Input image array as accepted by ``PIL.Image.fromarray``, typically
        shape ``(H, W, 3)`` with dtype ``uint8`` (see :func:`to_rgb_uint8`).

    Returns
    -------
    numpy.ndarray
        Cropped array of shape ``(image_size, image_size, 3)`` with the input
        dtype (``uint8`` for RGB input).

    Raises
    ------
    ValueError
        If ``image_size`` is not positive.
    """
    if image_size <= 0:
        raise ValueError(f'image_size must be positive, got {image_size}')

    pil_image = PIL.Image.fromarray(arr)

    # Stage 1: successive BOX halving. PIL's `size` is (width, height), and the
    # loop condition uses the *shorter* edge so that the following BICUBIC step
    # never has to downscale by more than 2x.
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    # Stage 2: single BICUBIC resize placing the shorter edge on `image_size`.
    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    # Stage 3: center slice. Rounding above can leave the longer edge one pixel
    # off, so the floor-division offsets bias the crop toward the top/left,
    # matching the reference byte-for-byte.
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
