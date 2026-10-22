"""Load live or EMA student weights from Lightning SSFM checkpoints.

``SSFMLitModule.on_save_checkpoint`` stores the EMA shadow under
``checkpoint['ema_state']`` (``{'ema_model': sd}`` for vanilla EMA, or
``{'stds': [...], 'profiles': [sd, ...]}`` for post-hoc EMA) with net-relative
keys, and strips the frozen ``net.base`` weights from both the live and vanilla
EMA payloads, recording their names in ``stripped_frozen_base``. A freshly
instantiated net already carries those frozen weights, so exactly those keys
may be missing; anything else missing or unexpected is a hard error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from src.models.components.metrics import profile_name

LIVE_WEIGHTS = 'non_ema'
VANILLA_EMA = 'ema'


def live_state(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    """Net-relative live (non-EMA) weights from the Lightning ``state_dict``."""
    return {
        key.removeprefix('net.'): value
        for key, value in checkpoint['state_dict'].items()
        if key.startswith('net.')
    }


def available_weights(checkpoint: Mapping[str, Any]) -> tuple[str, ...]:
    """Every weight selector the checkpoint supports (EMA profiles first)."""
    ema = checkpoint.get('ema_state') or {}
    if 'profiles' in ema:
        return (*[profile_name(float(std)) for std in ema['stds']], LIVE_WEIGHTS)
    if 'ema_model' in ema:
        return (VANILLA_EMA, LIVE_WEIGHTS)
    return (LIVE_WEIGHTS,)


def default_ema_weights(checkpoint: Mapping[str, Any]) -> str:
    """Primary EMA selector (``'ema'`` or the first post-hoc profile)."""
    choice = available_weights(checkpoint)[0]
    if choice == LIVE_WEIGHTS:
        raise KeyError('checkpoint carries no ema_state')
    return choice


def select_state(checkpoint: Mapping[str, Any], weights: str) -> dict[str, Tensor]:
    """Net-relative state dict for ``weights`` (``'ema'``, ``'sigma_x.xxx'``, ``'non_ema'``)."""
    if weights == LIVE_WEIGHTS:
        return live_state(checkpoint)
    ema = checkpoint.get('ema_state') or {}
    if weights == VANILLA_EMA and 'ema_model' in ema:
        return dict(ema['ema_model'])
    if 'profiles' in ema:
        by_name = {
            profile_name(float(std)): state
            for std, state in zip(ema['stds'], ema['profiles'], strict=True)
        }
        if weights in by_name:
            return dict(by_name[weights])
    raise KeyError(
        f'weights {weights!r} not in checkpoint; available: {available_weights(checkpoint)}'
    )


def load_net_state(net: nn.Module, state: Mapping[str, Tensor], stripped_base: list[str]) -> None:
    """Load ``state`` into ``net``, tolerating only the stripped frozen-base keys."""
    result = net.load_state_dict(dict(state), strict=False)
    allowed = {f'base.{name}' for name in stripped_base}
    missing = sorted(key for key in result.missing_keys if key not in allowed)
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f'checkpoint/net mismatch: missing {missing[:8]} (of {len(missing)}), '
            f'unexpected {sorted(result.unexpected_keys)[:8]}'
        )


def load_student(config_path, checkpoint_path, device, weights: str | None = None):  # noqa: ANN001, ANN201
    """Instantiate a trained student from its run config and load checkpoint weights.

    :param config_path: The run's ``.hydra/config.yaml``.
    :param checkpoint_path: A Lightning checkpoint written by the run.
    :param device: Device to place the net on.
    :param weights: Weight selector (defaults to the EMA weights).
    :return: ``(net, config, checkpoint_metadata)`` with the net in eval mode;
        the metadata holds ``global_step``, ``epoch`` and the selected ``weights``.
    """
    import hydra
    import torch
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    weights = weights or default_ema_weights(checkpoint)
    net = hydra.utils.instantiate(config.model.net)
    load_net_state(
        net, select_state(checkpoint, weights), checkpoint.get('stripped_frozen_base', [])
    )
    meta = {
        'global_step': int(checkpoint.get('global_step', -1)),
        'epoch': int(checkpoint['epoch']) if 'epoch' in checkpoint else None,
        'weights': weights,
    }
    return net.to(device).eval(), config, meta
