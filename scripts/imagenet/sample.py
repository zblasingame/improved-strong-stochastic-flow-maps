"""Sample ImageNet-256 images from a released model (iSSFM or an SSFM ablation rung).

Every NFE uses the SAME initial noise and the SAME Brownian path (composed with
the exact Chen relations), so each row of the output grid approximates the same
strong (pathwise) solution with increasing accuracy as the step count grows.

Usage::

    export DMF_CHECKPOINT_PATH=/path/to/dmf_xl_2_256.pt   # kyungmnlee/DMF
    uv run python -m scripts.imagenet.sample --model issfm --labels 339 207 \
        --nfe 1 2 4 8 16 --out samples.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import rootutils
import torch
from torchvision.utils import save_image

# Load .env (DATA_DIR, DMF_CHECKPOINT_PATH, ...) like the other entry points;
# variables already set in the environment take precedence.
rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.evaluation.pretrained import load_pretrained  # noqa: E402
from src.models.components.samplers import (  # noqa: E402
    chen_matched_ei_doss_rows,
    multistep_ei_euler_sampler,
)
from src.models.components.vae_decoder import StabilityVAEDecoder  # noqa: E402


@torch.no_grad()
def main() -> None:
    """Sample one row per label and save a (labels x NFE) image grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='issfm', help='Release name or local directory.')
    parser.add_argument('--labels', type=int, nargs='+', default=[339])
    parser.add_argument('--nfe', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('samples.png'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, config = load_pretrained(args.model, device)
    schedule = hydra.utils.instantiate(config.schedule)
    gen = torch.Generator(device).manual_seed(args.seed)
    labels = torch.tensor(args.labels, device=device)
    noise = torch.randn(len(labels), 4, 32, 32, device=device, generator=gen)

    degree, t0 = int(config.loss.degree), float(config.loss.epsilon)
    net.set_labels(labels)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        if 'wz_module' in config._target_:  # iSSFM / WZ models
            rows = chen_matched_ei_doss_rows(
                net, schedule, noise, args.nfe, degree=degree, t0=t0, generator=gen
            )
        else:  # semigroup (SSFM ladder) models: every dyadic NFE up to max(--nfe)
            rows, counts = multistep_ei_euler_sampler(
                net, schedule, noise, max_steps=max(args.nfe), degree=degree, t0=t0,
                generator=gen,
            )
            rows = torch.stack([rows[counts.index(n)] for n in args.nfe])
    net.set_labels(None)

    decoder = StabilityVAEDecoder('stabilityai/sd-vae-ft-mse').to(device)
    images = decoder(rows.transpose(0, 1).flatten(0, 1).float())  # (label, nfe) order
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.out, nrow=len(args.nfe))
    print(f'Saved {args.out} ({len(labels)} labels x NFE {args.nfe})')


if __name__ == '__main__':
    main()
