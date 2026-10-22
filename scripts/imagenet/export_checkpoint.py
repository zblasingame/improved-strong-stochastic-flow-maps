"""Export a trained ImageNet run to the release format.

Writes ``OUT_DIR/{model.safetensors, config.yaml}``: the EMA weights of the
trainable parameters only (LoRA adapters, coefficient injection, and the learned
diffusion head of the SSFM ablation rungs), and the model
section of the run config. The frozen DMF-XL/2+ backbone is not included; it is
rebuilt from ``DMF_CHECKPOINT_PATH`` at load time.

Usage::

    uv run python -m scripts.imagenet.export_checkpoint \
        --run-dir RUNS/imagenet256_wz_imf --checkpoint current.ckpt --out-dir release/issfm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from src.evaluation.checkpoints import default_ema_weights, select_state

# Keys of the model config needed to rebuild the student and its sampler.
MODEL_KEYS = ('_target_', 'net', 'schedule')
SAMPLER_KEYS = ('degree', 'epsilon')


# Ablation-ladder switches of the DMF adapter; omitted from the release config
# when they hold the iSSFM defaults.
LADDER_DEFAULTS = {'lift': True, 'diffusion_head': False, 'diffusion_init': 'zero'}


def release_net_config(net: DictConfig) -> DictConfig:
    """Normalize a run's net config to the release form (configs/model/net/dmf_xl2.yaml).

    Every ImageNet model shares the same student; the backbone path is supplied at
    load time and the Triton attention path is used for all models. The ablation
    ladder's adapter switches (learned diffusion head, no lift) are kept when set.
    """
    ladder = {
        key: net[key]
        for key, default in LADDER_DEFAULTS.items()
        if net.get(key) is not None and net[key] != default
    }
    return OmegaConf.create(
        {
            '_target_': 'src.models.components.fixed_coefficient_ssfm.'
            'FixedCoefficientDMFEIEulerSSFM',
            'base': {
                '_target_': 'src.models.components.dmf.load_dmf_xl2',
                'checkpoint_path': '${oc.env:DMF_CHECKPOINT_PATH}',
                'attn_func': 'triton',
            },
            'lora_rank': net.lora_rank,
            'lora_alpha': net.lora_alpha,
            'degree': net.degree,
            'modulation': net.get('modulation') or 'residual',
            'normalize_coefficients': net.get('normalize_coefficients', True),
            **ladder,
        }
    )


def main() -> None:
    """Export one run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='current.ckpt')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    ckpt = torch.load(
        args.run_dir / 'checkpoints' / args.checkpoint, map_location='cpu', weights_only=False
    )
    state = select_state(ckpt, default_ema_weights(ckpt))
    stripped = {f'base.{name}' for name in ckpt.get('stripped_frozen_base', [])}
    state = {k: v.float().contiguous() for k, v in state.items() if k not in stripped}

    config = OmegaConf.load(args.run_dir / '.hydra' / 'config.yaml')
    model = OmegaConf.masked_copy(config.model, list(MODEL_KEYS))
    model.loss = OmegaConf.masked_copy(config.model.loss, list(SAMPLER_KEYS))
    model.net = release_net_config(config.model.net)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {'global_step': str(ckpt.get('global_step', -1)), 'format': 'issfm-v1'}
    save_file(state, args.out_dir / 'model.safetensors', metadata=meta)
    OmegaConf.save(model, args.out_dir / 'config.yaml', resolve=False)
    n_params = sum(v.numel() for v in state.values())
    print(f'{args.out_dir}: {n_params / 1e6:.1f}M params, step {meta["global_step"]}')


if __name__ == '__main__':
    main()
