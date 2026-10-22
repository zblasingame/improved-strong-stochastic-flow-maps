"""Load released iSSFM ImageNet-256 models.

A release directory holds ``model.safetensors`` (EMA weights of the trainable
parameters) and ``config.yaml`` (the model config). The frozen DMF-XL/2+
backbone is rebuilt from ``DMF_CHECKPOINT_PATH``, which must point at
``dmf_xl_2_256.pt`` from https://huggingface.co/kyungmnlee/DMF.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

HF_REPO = 'zblasingame/improved-strong-stochastic-flow-maps'


def resolve_release(model: str) -> Path:
    """Return a local release directory for a path or a model name in ``HF_REPO``."""
    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    root = snapshot_download(HF_REPO, allow_patterns=[f'{model}/*'])
    return Path(root) / model


def load_pretrained(model: str, device: torch.device) -> tuple[torch.nn.Module, DictConfig]:
    """Build a released student on top of DMF-XL/2+ and load its weights.

    :param model: A release directory, or a model name in ``HF_REPO`` (e.g. ``issfm``).
    :param device: Device for the net.
    :return: ``(net, model_config)`` with the net in eval mode.
    """
    if not os.environ.get('DMF_CHECKPOINT_PATH'):
        data_dir = os.environ.get('DATA_DIR', 'data')
        default = Path(data_dir) / 'checkpoints' / 'dmf_xl_2_256.pt'
        if not default.is_file():
            raise RuntimeError('Set DMF_CHECKPOINT_PATH to dmf_xl_2_256.pt (kyungmnlee/DMF).')
        os.environ['DMF_CHECKPOINT_PATH'] = str(default)
    release = resolve_release(model)
    config = OmegaConf.load(release / 'config.yaml')
    net = hydra.utils.instantiate(config.net)
    result = net.load_state_dict(load_file(release / 'model.safetensors'), strict=False)
    missing = [k for k in result.missing_keys if not k.startswith('base.')]
    if missing or result.unexpected_keys:
        raise RuntimeError(f'weights mismatch: missing {missing[:5]}, unexpected '
                           f'{result.unexpected_keys[:5]}')
    return net.to(device).eval(), config
