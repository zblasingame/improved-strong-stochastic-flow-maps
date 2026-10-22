"""DiT diffusion transformer backbone (unconditional, time-only conditioning).

Ported and simplified from the DMF reference (``~/sfm/dmf/models/sit.py``). Class
conditioning (label embeddings / classifier-free-guidance) is intentionally
omitted: the backbone conditions only on the diffusion time ``t``. The REPA
projectors and feature-alignment heads are likewise omitted; instead a
lightweight ``return_features`` hook exposes intermediate block activations at a
configurable depth (REPA can be layered on later as a config-gated branch).

References
----------
- DiT: https://github.com/facebookresearch/DiT
- SiT: https://github.com/willisma/SiT
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .layers import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
)


class PatchEmbed(nn.Module):
    """Image-to-patch embedding via a strided convolution.

    A self-contained replacement for ``timm``'s ``PatchEmbed`` exposing the same
    ``.proj`` Conv2d attribute, so weight initialization mirrors the DMF backbone.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """Patchify ``x`` (shape ``(B, C, H, W)``) into tokens ``(B, num_patches, embed_dim)``."""
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        return rearrange(x, 'b d h w -> b (h w) d')  # (B, L, embed_dim)


class DiT(nn.Module):
    """Diffusion transformer with adaLN-Zero, time-only conditioning."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        **block_kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_patches = (input_size // patch_size) ** 2
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Fixed (non-trainable) 2D sin-cos positional embedding.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights: xavier Linear, sin-cos pos-embed, zero-init adaLN + final."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed proj like nn.Linear.
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in every DiT block.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """Reassemble patch tokens ``x`` (shape ``(N, L, p*p*C)``) into images ``(N, C, H, W)``."""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        return rearrange(
            x,
            'n (h w) (p q c) -> n c (h p) (w q)',
            h=h,
            w=w,
            p=p,
            q=p,
            c=c,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        return_features: bool = False,
        feature_depth: int = -1,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predict the velocity field for noisy inputs ``x`` at times ``t``.

        Parameters
        ----------
        x : Tensor
            Noisy inputs of shape ``(B, C, H, W)``.
        t : Tensor
            Times of shape ``(B,)`` (or broadcastable to it).
        return_features : bool, optional
            If ``True``, also return intermediate block activations.
        feature_depth : int, optional
            Index of the block whose output activations are returned when
            ``return_features`` is set. Negative indices count from the end.

        Returns
        -------
        Tensor or tuple of Tensor
            The predicted field ``(B, C, H, W)``, optionally paired with the
            requested intermediate features ``(B, num_patches, hidden_size)``.
        """
        t = t.reshape(-1)
        x = self.x_embedder(x) + self.pos_embed  # (B, L, D)
        b = x.shape[0]
        d = x.shape[-1]
        c = self.t_embedder(t).reshape(b, -1, d)  # (B, 1, D)

        num_blocks = len(self.blocks)
        target_depth = feature_depth % num_blocks
        features = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if return_features and i == target_depth:
                features = x

        x = self.final_layer(x, c)  # (B, L, p*p*C)
        x = self.unpatchify(x)  # (B, C, H, W)

        if return_features:
            return x, features
        return x
