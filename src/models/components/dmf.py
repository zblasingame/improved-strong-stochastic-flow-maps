"""Project-native DMFT architecture matching upstream checkpoint keys."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .dit import PatchEmbed
from .layers import (
    DiTBlock,
    FinalLayer,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    timestep_embedding,
)


class LabelEmbedder(nn.Module):
    """Embed ImageNet labels plus one classifier-free null label."""

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: Tensor) -> Tensor:
        """Return label embeddings."""
        return self.embedding_table(labels)


class DMFT(nn.Module):
    """Decoupled MeanFlow Transformer compatible with upstream DMFT-XL/2."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        dmf_depth: int = 20,
        qk_norm: bool = False,
        attn_func: str = 'torch_sdpa',
        use_logvar: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dmf_depth = dmf_depth
        self.use_logvar = use_logvar
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    attn_func=attn_func,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        if use_logvar:
            self.logvar_t_embedder = partial(timestep_embedding, dim=128)
            self.logvar_linear = nn.Linear(256, 1)
        position = get_2d_sincos_pos_embed(hidden_size, int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(position).float().unsqueeze(0))

    def unpatchify(self, tokens: Tensor) -> Tensor:
        """Reassemble patch tokens into latent feature maps."""
        return rearrange(
            tokens,
            'n (h w) (p q c) -> n c (h p) (w q)',
            h=self.input_size // self.patch_size,
            w=self.input_size // self.patch_size,
            p=self.patch_size,
            q=self.patch_size,
            c=self.out_channels,
        )

    def forward(self, x: Tensor, t: Tensor, r: Tensor, labels: Tensor) -> Tensor:
        """Predict DMF-oriented average velocity from noise time ``t`` to data time ``r``."""
        tokens = self.x_embedder(x)
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        condition_t = self.t_embedder(t.reshape(-1))
        condition_r = self.t_embedder(r.reshape(-1))
        label = self.y_embedder(labels).to(dtype=condition_t.dtype)
        condition_t = condition_t + label
        condition_r = condition_r + label
        for block in self.blocks[: self.dmf_depth]:
            tokens = block(tokens, condition_t)
        for block in self.blocks[self.dmf_depth :]:
            tokens = block(tokens, condition_r)
        return self.unpatchify(self.final_layer(tokens, condition_r))


def load_dmf_xl2(checkpoint_path: str, *, attn_func: str = 'torch_sdpa') -> DMFT:
    """Load a SiT-XL/2 or DMFT-XL/2 state into the shared XL/2 backbone.

    SiT-REPA checkpoints share the DiT-XL/2 keys and also carry unused
    ``projectors.*`` REPA heads. Those extras are dropped. Optional DMF-only
    log-variance keys may stay at initialization.
    """
    model = DMFT(attn_func=attn_func)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict):
        raise TypeError('DMF checkpoint must contain a state dictionary.')
    if any(not hasattr(value, 'shape') for value in state.values()):
        raise TypeError('DMF checkpoint must be a flat tensor state dictionary.')
    filtered = {key: value for key, value in state.items() if not key.startswith('projectors.')}
    result = model.load_state_dict(filtered, strict=False)
    required_missing = [key for key in result.missing_keys if not key.startswith('logvar_')]
    if required_missing:
        raise RuntimeError(
            'SiT/DMF checkpoint is missing backbone keys: ' + ', '.join(required_missing[:8])
        )
    unexpected = [key for key in result.unexpected_keys if not key.startswith('projectors.')]
    if unexpected:
        raise RuntimeError(
            'SiT/DMF checkpoint has unexpected backbone keys: ' + ', '.join(unexpected[:8])
        )
    return model
