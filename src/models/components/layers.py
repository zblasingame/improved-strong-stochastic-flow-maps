"""DiT transformer layers (adaLN-Zero conditioning).

Ported and simplified from the DMF reference (``~/sfm/dmf/models/layers.py``). The
flash-attention JVP kernels and REPA projectors are intentionally omitted; the
``base`` and ``torch_sdpa`` attention paths are retained because both are
``torch.func.jvp``-friendly, which the Eulerian/Lagrangian self-distillation
losses will rely on.

References
----------
- GLIDE: https://github.com/openai/glide-text2im
- MAE: https://github.com/facebookresearch/mae
- timm: https://github.com/huggingface/pytorch-image-models
"""

import math
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaLN-Zero modulation ``x * (1 + scale) + shift``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` along its last dimension."""
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True, dtype=torch.float32) + self.eps)
        return x * norm_x.to(dtype=x.dtype) * self.scale.to(dtype=x.dtype)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal timestep embeddings of size ``dim`` for times ``t``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Embed a scalar timestep into a ``hidden_size`` conditioning vector."""

    def __init__(self, hidden_size: int, emb_size: int = 256):
        super().__init__()
        self.emb_size = emb_size
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps ``t`` (shape ``(B,)``) into shape ``(B, hidden_size)``."""
        t_freq = timestep_embedding(t, dim=self.emb_size).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


def attn_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    op: str = 'base',
    mask: Tensor | None = None,
) -> Tensor:
    """Compute attention over ``q, k, v`` of shape ``(B, L, H, D)``.

    :param op: One of ``"base"`` (explicit softmax), ``"torch_sdpa"`` (fused SDPA),
        or ``"triton"`` (Triton fused kernel with a forward-mode JVP path).
        ``"base"`` and ``"torch_sdpa"`` support ``torch.func.jvp`` natively; the
        ``"triton"`` path requires Triton and a CUDA device.
    :param mask: Optional boolean key/value mask of shape ``(B, M)`` where ``M`` is
        the key/value sequence length; ``True`` marks key positions that take part
        in attention. Only the ``"base"`` and ``"torch_sdpa"`` backends honor it
        (with identical boolean semantics); any other backend raises rather than
        silently ignoring the mask.
    """
    if mask is not None:
        if op not in ('base', 'torch_sdpa'):
            raise RuntimeError(
                f"attn_func={op!r} does not support key/value masks; use 'base' or 'torch_sdpa'."
            )
        if mask.dtype != torch.bool:
            raise ValueError(
                f'mask must be a boolean tensor (True = valid key position); got {mask.dtype}.'
            )
        mask = rearrange(mask, 'b m -> b 1 1 m')  # broadcast over heads and queries

    if op == 'triton':
        from .attention_jvp import triton_attn_func

        if triton_attn_func is None:
            raise RuntimeError(
                "attn_func='triton' requires Triton (and a CUDA device). It is "
                "unavailable in this environment; use 'base' or 'torch_sdpa'."
            )
        return triton_attn_func(q, k, v)  # operates on (B, L, H, D)

    q, k, v = (rearrange(t, 'b l h d -> b h l d') for t in (q, k, v))  # (B, H, L, D)
    if op == 'torch_sdpa':
        # Boolean attn_mask: True = takes part in attention (matches `base`).
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
    else:  # base
        scale = q.shape[-1] ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        x = attn @ v
    return rearrange(x, 'b h l d -> b l h d')


class Attention(nn.Module):
    """Multi-head self-attention with optional QK RMS-normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_func: str = 'base',
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('dim should be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn = partial(attn_op, op=attn_func)

    def forward(self, x: Tensor) -> Tensor:
        """Self-attend over the token dimension of ``x`` (shape ``(B, L, C)``)."""
        qkv = rearrange(
            self.qkv(x),
            'b l (three h d) -> three b l h d',
            three=3,
            h=self.num_heads,
        )
        q, k, v = qkv.unbind(0)  # each (B, L, H, D)
        q, k = self.q_norm(q), self.k_norm(k)
        x = rearrange(self.attn(q, k, v), 'b l h d -> b l (h d)')
        return self.proj(x)


class MLP(nn.Module):
    """Two-layer feed-forward block with a configurable activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward transformation to ``x``."""
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer-norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs.get('qk_norm', False),
            attn_func=block_kwargs.get('attn_func', 'base'),
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate='tanh'),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Apply conditioned attention and MLP sub-blocks to tokens ``x`` given context ``c``."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final adaLN-Zero layer projecting tokens back to patch pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Modulate and project tokens ``x`` given conditioning ``c``."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Build a fixed 2D sine-cosine positional embedding of shape ``(grid_size**2, embed_dim)``."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)  # w goes first
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Combine height/width 1D embeddings into a 2D positional embedding."""
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even')
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Build a 1D sine-cosine positional embedding for positions ``pos``."""
    if embed_dim % 2 != 0:
        raise ValueError('embed_dim must be even')
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
