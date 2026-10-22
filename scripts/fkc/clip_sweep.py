"""Paper-style metric sweep over inference-time total log-weight clipping.

``--clip-total 0.10`` winsorizes log weights at the 5th and 95th
GLOBAL-cloud quantiles at *every* SMC step before resampling (K=8/50
independent groups); 0.25/0.50/0.75 correspond to per-tail rates
0.125/0.25/0.375. The legacy ``--weight-clip 0.10`` means 20% total,
included by default. Neither sample positions nor evaluation energies are
ever clipped. Existing unclipped results can be used for the zero control.
"""

import argparse
import json
import math
from pathlib import Path

import hydra
import torch
from scripts.fkc.metrics_table import METRICS, SCALE, mmd, total_variation, wasserstein
from omegaconf import OmegaConf
from scripts.fkc.sampling import _load, run_sampler

from src.models.components.fkc.evaluation import energy_values, energy_wasserstein


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--diffusion-run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 12, 16, 20, 24, 32, 64])
    parser.add_argument('--k', type=int, nargs='+', default=[8, 50])
    parser.add_argument('--clip-total', type=float, nargs='+', default=[.10, .20, .25, .50, .75])
    parser.add_argument('--particles', type=int, default=10000)
    parser.add_argument('--out-dir', type=Path, default=Path('results/fkc/clip_sweep'))
    args = parser.parse_args()
    if any(not 0.0 < c < 1.0 for c in args.clip_total):
        raise ValueError('all clip-total fractions must lie in (0, 1).')
    if any(args.particles % k != 0 for k in args.k):
        raise ValueError('number of particles must be divisible by every K.')

    module, map_ckpt = _load(args.run_dir, None)
    config = OmegaConf.load(args.diffusion_run_dir / '.hydra' / 'config.yaml')
    diffusion = hydra.utils.instantiate(config.model)
    diffusion_ckpt = args.diffusion_run_dir / 'checkpoints' / 'last.ckpt'
    diffusion.load_state_dict(torch.load(diffusion_ckpt, map_location='cpu', weights_only=False)['state_dict'])
    if diffusion.problem.powers != (1,) or not torch.equal(
        diffusion.problem.experts[0].means, module.problem.experts[0].means
    ):
        raise ValueError('Diffusion must use beta=1 and the same Gaussian modes.')
    diffusion.eval()
    n, seed = args.particles, args.seed
    problem = module.problem
    x0 = problem.sample_target(n, module.loss.t0, generator=torch.Generator().manual_seed(seed))
    reference = problem.sample_target(n, 1.0, generator=torch.Generator().manual_seed(1000 + seed))
    ref_energy, ref_u = energy_values(problem, reference).cpu(), reference.cpu() * SCALE
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f'{args.run_dir.name}_seed{seed}.json'
    rows = []
    metadata = dict(map_checkpoint=str(map_ckpt), diffusion_checkpoint=str(diffusion_ckpt),
                    seed=seed, nfe=args.nfe, k=args.k, clip_total=args.clip_total,
                    particles=n, resample_every=1)

    for steps in args.nfe:
        for k in args.k:
            for clip in args.clip_total:
                for sampler in ('learned', 'diffusion'):
                    trace: list[float] = []
                    with torch.no_grad():
                        cloud, _ = run_sampler(
                            sampler, steps, x0, k, 1,
                            problem=problem, net=module.ema_net, diffusion_net=diffusion.ema_net,
                            schedule=module.loss.schedule, t0=module.loss.t0,
                            degree=module.loss.degree,
                            generator=torch.Generator().manual_seed(7919 * seed + 13),
                            clip_total_fraction=clip, ess_trace=trace,
                        )
                    row = dict(sampler=sampler, nfe=steps, k=k, clip_total=clip, seed=seed,
                               mean_pre_resampling_ess_fraction=sum(trace) / len(trace),
                               terminal_pre_resampling_ess_fraction=trace[-1])
                    if not bool(torch.isfinite(cloud).all()) or float(cloud.abs().max()) > 1e3:
                        row['diverged'] = True
                    else:
                        with torch.no_grad():
                            cu = cloud.cpu() * SCALE
                            row.update(energy_w2=energy_wasserstein(
                                ref_energy, energy_values(problem, cloud).cpu(), order=2),
                                mmd=mmd(cu, ref_u), total_variation=total_variation(cu, ref_u),
                                w1=wasserstein(cu, ref_u, 1), w2=wasserstein(cu, ref_u, 2))
                        if any(not math.isfinite(row[m]) for m in METRICS):
                            row['diverged'] = True
                    rows.append(row)
                    # Checkpoint progress, so cancelled jobs still leave usable data.
                    out.write_text(json.dumps({'metadata': metadata, 'rows': rows}, indent=1))
                    print(json.dumps(row), flush=True)
    print(f'[clip sweep] wrote {len(rows)} rows to {out}', flush=True)


if __name__ == '__main__':
    main()
