"""Exponential moving average (EMA) helpers for the training models.

A single constant-decay shadow copy updated by :func:`update_ema`;
:class:`build_ema` constructs it.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


def _to_float32_ema(module: nn.Module) -> None:
    """Keep EMA copies in float32 so small decays are not quantized away."""
    for name, parameter in list(module.named_parameters()):
        if not parameter.is_floating_point() or parameter.dtype == torch.float32:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._parameters[leaf] = nn.Parameter(parameter.detach().float(), requires_grad=False)
    for name, buffer in list(module.named_buffers()):
        if not buffer.is_floating_point() or buffer.dtype == torch.float32:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = module.get_submodule(parent_name) if parent_name else module
        parent._buffers[leaf] = buffer.detach().float()


def _strip_prefix(name: str) -> str:
    """Strip DDP / ``torch.compile`` wrapper prefixes from a parameter name.

    :param name: Possibly-prefixed parameter/buffer name.
    :return: The name with a leading wrapper prefix removed, if present.
    """
    for pre in ('module.', 'orig_mod.', '_orig_mod.'):
        if name.startswith(pre):
            return name[len(pre) :]
    return name


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.9999,
    only_trainable: bool = True,
    update_buffers: bool = True,
) -> None:
    """Update a vanilla constant-decay EMA model in place.

    ``p_ema <- decay * p_ema + (1 - decay) * p_model`` for floating-point
    parameters; non-float parameters and (optionally) buffers are copied
    directly. Wrapper prefixes (``module.`` / ``orig_mod.``) are stripped so DDP
    and ``torch.compile`` wrapped models align with the EMA copy.

    :param ema_model: The EMA shadow model to update in place.
    :param model: The live training model.
    :param decay: Constant EMA decay in ``[0, 1)``.
    :param only_trainable: Skip parameters with ``requires_grad=False``.
    :param update_buffers: Copy buffers directly (no decay).
    """
    trainable = {_strip_prefix(n) for n, p in model.named_parameters() if p.requires_grad}
    ema_params = {_strip_prefix(n): p for n, p in ema_model.named_parameters()}

    for raw_name, p_model in model.named_parameters():
        name = _strip_prefix(raw_name)
        if only_trainable and name not in trainable:
            continue
        p_ema = ema_params.get(name)
        if p_ema is None:
            continue
        if p_ema.is_floating_point():
            p_ema.mul_(decay).add_(p_model.data.float().to(device=p_ema.device), alpha=1.0 - decay)
        else:
            p_ema.copy_(p_model.data)

    if update_buffers:
        ema_buffers = {_strip_prefix(n): b for n, b in ema_model.named_buffers()}
        for raw_name, b_model in model.named_buffers():
            b_ema = ema_buffers.get(_strip_prefix(raw_name))
            if b_ema is not None:
                b_ema.copy_(b_model)


class VanillaEMA:
    """A single constant-decay EMA shadow model.

    Provides :meth:`update` and :meth:`get_model`; the decay is a fixed
    constant and there is exactly one profile.

    Parameters
    ----------
    model : nn.Module
        The live training model (cloned to seed the shadow copy).
    decay : float, optional
        Constant EMA decay. Defaults to ``0.9999``.
    device : torch.device, optional
        Device for the shadow copy (defaults to the model's device).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device | None = None,
    ) -> None:
        self.decay = decay
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = device
        self.ema_model = deepcopy(model).to(device)
        _to_float32_ema(self.ema_model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def init_from_model(self, model: nn.Module) -> None:
        """Copy the live model weights into the shadow copy (decay-free).

        :param model: The live training model to seed from.
        """
        update_ema(self.ema_model, model, decay=0.0, only_trainable=False)

    def get_model(self, index: int = 0) -> nn.Module:
        """Return the EMA shadow model.

        :param index: Ignored (vanilla EMA has a single profile).
        :return: The EMA shadow model.
        """
        return self.ema_model

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        t_next: float | None = None,
        t_delta: float | None = None,
        only_trainable: bool = True,
        update_buffers: bool = True,
    ) -> None:
        """Apply one constant-decay EMA update.

        :param model: The live training model.
        :param t_next: Ignored.
        :param t_delta: Ignored.
        :param only_trainable: Skip parameters with ``requires_grad=False``.
        :param update_buffers: Copy buffers directly (no decay).
        """
        update_ema(
            self.ema_model,
            model,
            decay=self.decay,
            only_trainable=only_trainable,
            update_buffers=update_buffers,
        )

    def state_dict(self) -> dict:
        """Return a serialisable dict of the shadow model's weights."""
        return {'ema_model': self.ema_model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the shadow model from a :meth:`state_dict` payload."""
        self.ema_model.load_state_dict(state['ema_model'])


def build_ema(
    model: nn.Module,
    *,
    decay: float = 0.9999,
    device: torch.device | None = None,
) -> VanillaEMA:
    """Construct the constant-decay EMA tracker.

    :param model: The live training model to track.
    :param decay: Constant decay.
    :param device: Device for the EMA copy.
    :return: An EMA tracker exposing ``update`` / ``get_model``.
    """
    return VanillaEMA(model, decay=decay, device=device)
