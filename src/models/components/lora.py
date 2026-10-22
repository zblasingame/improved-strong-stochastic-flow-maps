"""Model-agnostic low-rank adapters for linear layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _validate_lora_config(rank: int, alpha: float) -> None:
    """Validate low-rank adapter dimensions and scaling."""
    if rank < 1:
        raise ValueError(f'rank must be at least 1; got {rank}.')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f'alpha must be finite and positive; got {alpha}.')


class LoraLinear(nn.Module):
    """Add a trainable low-rank update to a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        _validate_lora_config(rank, alpha)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = self._clone_bias(base.bias)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @staticmethod
    def _clone_bias(bias: nn.Parameter | None) -> nn.Parameter | None:
        if bias is None:
            return None
        return nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the frozen linear map and, when enabled, its low-rank update."""
        output = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return output
        # The low-rank factors stay float32 for optimizer precision even when
        # the frozen stack runs in its storage dtype with autocast disabled
        # (autocast's per-op casts do not compose with forward-mode AD, so the
        # Sana JVP region runs uniform bfloat16 instead). Cast on the fly;
        # gradients flow back to the float32 factors through the cast.
        lora_a = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
        lora_b = self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
        update = F.linear(F.linear(x, lora_a), lora_b)
        return output + self.scale * update

    def merge(self) -> nn.Linear:
        """Return a plain linear layer with the low-rank update absorbed.

        The update ``scale * (B @ A)`` is small relative to the frozen weight, so
        both the accumulation *and* the exported layer are float32 even when the
        frozen weight is stored in bfloat16. Accumulating in float32 and then
        storing to bfloat16 would discard the extra precision at the final round:
        at the observed magnitudes the bfloat16 grid spacing near ``|W|`` is a
        sizeable fraction of the update, so the exported checkpoint would lose
        part of the adapter it is supposed to bake in.
        """
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=torch.float32,
        )
        update = self.lora_B.float() @ self.lora_A.float()
        merged.weight.data.copy_(self.weight.float() + self.scale * update)
        if self.bias is not None:
            merged.bias.data.copy_(self.bias.float())
        return merged


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated module path, including numeric list indices."""
    module = model
    for part in path.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    """Replace the module at a dot-separated path."""
    parent_path, _, leaf = path.rpartition('.')
    parent = _get_submodule(model, parent_path) if parent_path else model
    if leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _expand_targets(model: nn.Module, targets: tuple[str, ...]) -> list[str]:
    """Expand block-index placeholders in caller-supplied target paths."""
    paths: list[str] = []
    for target in targets:
        if '{i}' not in target:
            paths.append(target)
            continue
        if not hasattr(model, 'blocks'):
            raise ValueError("target contains '{i}', but model has no 'blocks' module.")
        paths.extend(target.replace('{i}', str(index)) for index in range(len(model.blocks)))
    return paths


def freeze_except_lora(model: nn.Module, trainable_prefixes: tuple[str, ...] = ()) -> None:
    """Freeze inherited parameters while retaining adapters and declared new modules."""
    for name, parameter in model.named_parameters():
        is_lora = name.endswith(('lora_A', 'lora_B'))
        is_new = name.startswith(trainable_prefixes)
        parameter.requires_grad_(is_lora or is_new)


def apply_lora(
    model: nn.Module,
    targets: tuple[str, ...],
    rank: int = 16,
    alpha: float = 16.0,
    trainable_prefixes: tuple[str, ...] = (),
) -> nn.Module:
    """Wrap selected linear layers and freeze all inherited parameters."""
    for path in _expand_targets(model, targets):
        try:
            module = _get_submodule(model, path)
        except (AttributeError, IndexError, KeyError) as error:
            raise ValueError(f'Unable to resolve LoRA target {path!r}.') from error
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f'LoRA target {path!r} must resolve to nn.Linear; got {type(module).__name__}.'
            )
        _set_submodule(model, path, LoraLinear(module, rank=rank, alpha=alpha))
    freeze_except_lora(model, trainable_prefixes)
    return model


def cast_frozen_to(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Store every frozen floating-point parameter of ``model`` in ``dtype``.

    Under ``bf16-mixed`` autocast already rounds each frozen weight to bfloat16
    before every matmul, so materializing that cast once is bitwise identical to
    performing it on each of the (many) forward passes per step. It halves the
    resident size of the frozen backbone and removes the repeated casts.

    Trainable parameters (the low-rank factors) are left in float32: they receive
    optimizer updates and must keep full precision.

    :param model: The module whose frozen parameters should be cast.
    :param dtype: Target floating-point dtype.
    :return: The same module, mutated in place.
    """
    for name, parameter in list(model.named_parameters()):
        if parameter.requires_grad or not parameter.is_floating_point():
            continue
        if parameter.dtype == dtype:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._parameters[leaf] = nn.Parameter(
            parameter.detach().to(dtype=dtype), requires_grad=False
        )
    return model


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Enable or disable every low-rank adapter in a model."""
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.enabled = enabled


