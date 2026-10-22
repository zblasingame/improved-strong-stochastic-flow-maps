"""Original-DMF-style fixed-coefficient SSFM adapters."""

from __future__ import annotations


import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .compile import compile_adapter_backbone
from .dmf import DMFT
from .fixed_coefficient import FixedCoefficientResidual
from .layers import FinalLayer
from .lora import apply_lora, set_lora_enabled
from .ssfm import SSFMNet, _diagonal_velocity

DMF_LORA_TARGETS = (
    't_embedder.mlp.0',
    't_embedder.mlp.2',
    'blocks.{i}.attn.qkv',
    'blocks.{i}.attn.proj',
    'blocks.{i}.mlp.fc1',
    'blocks.{i}.mlp.fc2',
    'blocks.{i}.adaLN_modulation.1',
    'final_layer.adaLN_modulation.1',
    'final_layer.linear',
)


def _build_modulation(hidden_size: int, modulation: str, *, degree: int = 3) -> nn.Module:
    """Build the spatial levy residual modulation."""
    if modulation == 'residual':
        return FixedCoefficientResidual(hidden_size, degree=degree)
    raise ValueError(f"modulation must be 'residual'; got {modulation!r}.")


DIFFUSION_INITS = ('zero', 'random')


def _build_diffusion_layer(base: DMFT, init: str = 'zero') -> FinalLayer:
    """Build an adaLN head mirroring the DMF drift read-out for the noise ``g``.

    ``'zero'`` zeroes the output projection so ``g = 0`` at the start;
    ``'random'`` keeps PyTorch's default initialization, matching the
    reference SSFM, whose diffusion net keeps its library-default final conv.
    """
    if init not in DIFFUSION_INITS:
        raise ValueError(f'diffusion_init must be one of {DIFFUSION_INITS}; got {init!r}.')
    layer = FinalLayer(base.hidden_size, base.patch_size, base.out_channels)
    if init == 'zero':
        nn.init.zeros_(layer.linear.weight)
        nn.init.zeros_(layer.linear.bias)
    return layer


class _FixedCoefficientMixin:
    """Shared label and coefficient-token helpers for the DMF adapter."""

    def _active_labels(self, x: Tensor) -> Tensor:
        if self._labels is None:
            raise RuntimeError('No labels assigned; call set_labels(...) before evaluation.')
        labels = self._labels.reshape(-1).to(device=x.device, dtype=torch.long)
        if labels.shape[0] != x.shape[0]:
            raise ValueError(
                f'label batch size {labels.shape[0]} does not match input {x.shape[0]}.'
            )
        return labels

    def _fixed_coefficient_tokens(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 5 or coefficients.shape[1] != self.degree:
            raise ValueError(
                f'fixed-coefficient adapters require degree={self.degree} features with '
                f'shape (B, {self.degree}, C, H, W); got {tuple(coefficients.shape)}.'
            )
        fields = rearrange(coefficients, 'b n c h w -> (b n) c h w')
        tokens = self.coefficient_embedder(fields)
        return rearrange(tokens, '(b n) l d -> b n l d', b=coefficients.shape[0])


class FixedCoefficientDMFEIEulerSSFM(_FixedCoefficientMixin, SSFMNet):
    """Latent DMF EI-Euler adapter using additive fixed-coefficient conditioning.

    Parameters
    ----------
    base : DMFT
        Frozen pretrained DMFT backbone (LoRA adapters attached in place).
    degree : int, optional
        Fixed coefficient count the adapter consumes (``3`` reproduces the
        original SDMFT ``W, H, K`` design bit-for-bit, including checkpoint
        key layout). ``model.loss.degree`` must match.
    normalize_coefficients : bool, optional
        Whether callers feed unit-variance coefficient features (each
        ``I^{(n)}`` divided by its standard deviation). ``False`` exposes the
        raw coefficients at their natural scale ``sqrt((t-s)/(2n+1))``, the
        hypothesis test for whether unit-variance conditioning amplifies
        high-order channels into low-SNR inputs (the degree-6 underperformance).
    diffusion_head : bool, optional
        Ablation switch (default ``False``). When set, an adaLN head
        (initialized per ``diffusion_init``) reads the final tokens as a
        learned diffusion field ``g`` and the adapter becomes
        ``'euler_maruyama'``: the noise term is ``g * W_{s,t}`` (see
        :mod:`src.models.components.ablation_map`) instead of the closed-form
        scaled martingale.
    diffusion_init : {'zero', 'random'}, optional
        Initialization of the learned diffusion head's output projection
        (default ``'zero'``, i.e. ``g = 0`` at the start; ``'random'`` keeps
        PyTorch's default, as in the reference SSFM).
    lift : bool, optional
        Ablation switch (default ``True``). ``False`` drops the exponential
        lift: the drift head is read as the raw SDE drift ``b_st`` and each
        step is the plain Euler step ``x + h b_st`` plus the learned
        ``g_st W_{s,t}`` or the closed-form ``int_s^t nu_r dW_r``
        (see :mod:`src.models.components.ablation_map`).
    """

    def __init__(
        self,
        base: DMFT,
        *,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        degree: int = 3,
        modulation: str = 'residual',
        normalize_coefficients: bool = True,
        diffusion_head: bool = False,
        diffusion_init: str = 'zero',
        lift: bool = True,
    ) -> None:
        super().__init__(
            parameterization='euler_maruyama' if diffusion_head else 'euler',
            normalize_coefficients=normalize_coefficients,
        )
        self.lift = lift
        if degree < 1:
            raise ValueError(f'degree must be >= 1; got {degree}.')
        self.base = base
        self.num_classes = base.num_classes
        self.degree = degree
        self._labels: Tensor | None = None
        self.coefficient_embedder = base.x_embedder
        self.fixed_residual = _build_modulation(base.hidden_size, modulation, degree=degree)
        apply_lora(base, DMF_LORA_TARGETS, rank=lora_rank, alpha=lora_alpha)
        # Built after LoRA so it lives outside the frozen base and stays trainable.
        self.diffusion_layer = (
            _build_diffusion_layer(base, diffusion_init) if diffusion_head else None
        )

    def set_labels(self, labels: Tensor | None) -> None:
        """Assign transient labels."""
        self._labels = labels

    def pretrained_prediction(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        """Evaluate the original DMFT with project-time conversion."""
        set_lora_enabled(self.base, False)
        try:
            return -self.base(x, 1.0 - t, 1.0 - t, labels)
        finally:
            set_lora_enabled(self.base, True)

    def compile_backbone(
        self, *, pretrained: bool = True, diagonal: bool = True, field: bool = True
    ) -> None:
        """Compile the shared block stack backing every prediction path."""
        compile_adapter_backbone(self, pretrained=pretrained, field=field or diagonal)

    def anchor_prediction(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        """Evaluate the frozen pretrained path (its blocks are compiled in place)."""
        return self.pretrained_prediction(x, t, labels)

    def anchor_student_prediction(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluate the diagonal path (its blocks are compiled in place)."""
        return self.diagonal_prediction(x, t)

    def diagonal_prediction(self, x: Tensor, t: Tensor) -> Tensor:
        """Evaluate the h -> 0 field with silenced Brownian coefficients."""
        return _diagonal_velocity(self, x, t, degree=self.degree)

    def diagonal_fields(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor | None]:
        """Evaluate ``(drift, diffusion)`` on ``s = t`` with silenced coefficients.

        One forward serves both heads; ``diffusion`` is ``None`` without a head.
        """
        times = t.reshape(-1)
        coefficients = x.new_zeros(x.shape[0], self.degree, *x.shape[1:])
        return self._field(x, times, times, coefficients)

    def drift(self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor) -> Tensor:
        """Evaluate the fixed-coefficient EI-Euler field."""
        return self._field(x, s.reshape(-1), t.reshape(-1), coefficients)[0]

    def _field(
        self, x: Tensor, s: Tensor, t: Tensor, coefficients: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Build the additive fixed-coefficient latent-token stream."""
        labels = self._active_labels(x)
        condition_s = self.base.t_embedder(1.0 - s)
        condition_t = self.base.t_embedder(1.0 - t)
        label = self.base.y_embedder(labels).to(dtype=condition_s.dtype)
        condition_s = condition_s + label
        condition_t = condition_t + label
        image_tokens = self.base.x_embedder(x)
        coefficient_tokens = self._fixed_coefficient_tokens(coefficients)
        image_tokens = self.fixed_residual(image_tokens, coefficient_tokens, s, t)
        tokens = image_tokens + self.base.pos_embed.to(dtype=image_tokens.dtype)
        for index, block in enumerate(self.base.blocks):
            condition = condition_s if index < self.base.dmf_depth else condition_t
            tokens = block(tokens, condition)
        drift = -self.base.unpatchify(self.base.final_layer(tokens, condition_t))
        if self.diffusion_layer is None:
            return drift, None
        return drift, self.base.unpatchify(self.diffusion_layer(tokens, condition_t))
