"""Export point clouds and contour lines for the pgfplots 40-Gaussian figure.

Layout (2 x 7), all in the reference notebook's unnormalized coordinates (x50):

    row 1: starting distribution | SSFM NFE 4 | diffusion NFE 4 | SSFM 8 | diffusion 8 | SSFM 16 | diffusion 16
    row 2: ground truth          | SSFM+FKC 4 | diffusion+FKC 4 | ...                                   (K=8, clip)

Every rollout starts from the same 10k particles (seed ``--seed``) and uses the
same Brownian RNG stream as the metric tables. FKC rows resample within
independent groups of K particles after every step with ``--clip`` TOTAL
log-weight winsorization (0.25 = 12.5th/87.5th percentiles). A fixed random
subset of ``--n-plot`` particles is written per panel.

Points outside the plotting window are clamped to it (as in the reference
``plot_marginal_pair``) and flagged ``clamped=1``; nonfinite points are dropped.
Per-panel counts are recorded in ``metadata.json`` so diverged panels can be
styled or left empty.

Contours follow the reference ``plot_contours``: 20 levels of
``-sqrt(-log p)`` (log p clamped at -1000) on a square grid, written as
``x,y,level`` with ``nan,nan`` rows separating line segments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import contourpy
import hydra
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf

from scripts.fkc.sampling import _load, run_sampler  # noqa: E402

SCALE = 50.0
BOUND = 1.4 * 40.0


def log_prob_grid(log_prob, points: int, chunk: int = 512):
    axis = torch.linspace(-BOUND, BOUND, points, dtype=torch.float64)
    xx, yy = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).float()
    with torch.no_grad():
        values = torch.cat([log_prob(grid[i : i + chunk] / SCALE) for i in range(0, grid.shape[0], chunk)])
    values = (values.double() - 2.0 * math.log(SCALE)).clamp_min(-1000.0)
    return xx.numpy(), yy.numpy(), values.reshape(points, points).numpy()


def write_contours(path: Path, log_prob, points: int, levels: int = 20) -> int:
    xx, yy, logp = log_prob_grid(log_prob, points)
    z = -np.sqrt(-logp)
    # matplotlib's contour(levels=int) uses MaxNLocator(levels + 1) and keeps
    # only the levels strictly inside the data range for line contours.
    values = MaxNLocator(levels + 1).tick_values(z.min(), z.max())
    values = values[(values > z.min()) & (values < z.max())]
    generator = contourpy.contour_generator(xx, yy, z)
    rows = 0
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'level'])
        for index, value in enumerate(values):
            for segment in generator.lines(float(value)):
                for x, y in segment:
                    writer.writerow([f'{x:.3f}', f'{y:.3f}', index])
                    rows += 1
                writer.writerow(['nan', 'nan', index])
    return rows


def write_cloud(path: Path, cloud: torch.Tensor, subset: torch.Tensor) -> dict:
    cloud = cloud.detach().cpu().double()
    finite = torch.isfinite(cloud).all(-1)
    points = (cloud[subset] * SCALE)
    keep = finite[subset]
    points = points[keep]
    clamped = (points.abs() > BOUND).any(-1)
    points = points.clamp(-BOUND, BOUND)
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'clamped'])
        for (x, y), flag in zip(points.tolist(), clamped.tolist(), strict=True):
            writer.writerow([f'{x:.3f}', f'{y:.3f}', int(flag)])
    return {
        'written': int(points.shape[0]),
        'nonfinite_dropped': int((~keep).sum()),
        'clamped': int(clamped.sum()),
        'finite_fraction_all_particles': float(finite.double().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--diffusion-run-dir', type=Path, required=True)
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--clip', type=float, default=0.25, help='TOTAL log-weight clip fraction')
    parser.add_argument('--particles', type=int, default=10000)
    parser.add_argument('--n-plot', type=int, default=2000)
    parser.add_argument('--grid', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=Path('results/fkc/panels'))
    args = parser.parse_args()
    if args.particles % args.k:
        raise ValueError('particles must be divisible by K.')

    module, map_ckpt = _load(args.run_dir, None)
    config = OmegaConf.load(args.diffusion_run_dir / '.hydra' / 'config.yaml')
    diffusion = hydra.utils.instantiate(config.model)
    diffusion_ckpt = args.diffusion_run_dir / 'checkpoints' / 'last.ckpt'
    diffusion.load_state_dict(torch.load(diffusion_ckpt, map_location='cpu', weights_only=False)['state_dict'])
    problem, loss = module.problem, module.loss
    if diffusion.problem.powers != (1,) or not torch.equal(
        diffusion.problem.experts[0].means, problem.experts[0].means
    ):
        raise ValueError('Diffusion baseline must use beta=1 and identical modes.')
    diffusion.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subset = torch.randperm(args.particles, generator=torch.Generator().manual_seed(12345))[: args.n_plot]
    meta = {
        'flow_map_checkpoint': str(map_ckpt),
        'diffusion_checkpoint': str(diffusion_ckpt),
        'seed': args.seed, 'particles': args.particles, 'n_plot': args.n_plot,
        'k_fkc': args.k, 'clip_total_fkc': args.clip, 'resample_every': 1,
        'plot_bound': BOUND, 'coordinates': 'unnormalized (x50)', 'panels': {},
    }

    gen = torch.Generator().manual_seed(1000 + args.seed)
    start = problem.experts[0].sample(args.particles, 1.0, generator=gen)
    truth = problem.sample_target(args.particles, 1.0, generator=gen)
    meta['panels']['start'] = write_cloud(args.out_dir / 'start.csv', start, subset)
    meta['panels']['truth'] = write_cloud(args.out_dir / 'truth.csv', truth, subset)

    x0 = problem.sample_target(args.particles, loss.t0, generator=torch.Generator().manual_seed(args.seed))
    for steps in args.nfe:
        for name, sampler in (('ssfm', 'learned'), ('diffusion', 'diffusion')):
            for fkc, k, clip in ((False, 1, 0.0), (True, args.k, args.clip)):
                with torch.no_grad():
                    cloud, _ = run_sampler(
                        sampler, steps, x0, k, 1, problem=problem, net=module.ema_net,
                        schedule=loss.schedule, t0=loss.t0, degree=loss.degree,
                        generator=torch.Generator().manual_seed(7919 * args.seed + 13),
                        diffusion_net=diffusion.ema_net, clip_total_fraction=clip,
                    )
                panel = f'{name}{"_fkc" if fkc else ""}_nfe{steps}'
                meta['panels'][panel] = write_cloud(args.out_dir / f'{panel}.csv', cloud, subset)
                print(f'[panels] {panel}: {meta["panels"][panel]}', flush=True)

    print('[panels] contours ...', flush=True)
    meta['contour_rows'] = {
        'start': write_contours(
            args.out_dir / 'contours_start.csv', lambda x: problem.experts[0].log_prob(x, 1.0), args.grid
        ),
        'target': write_contours(
            args.out_dir / 'contours_target.csv', lambda x: problem.log_target(x, 1.0), args.grid
        ),
    }
    (args.out_dir / 'metadata.json').write_text(json.dumps(meta, indent=1))
    print(f'[panels] wrote {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
