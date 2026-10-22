"""Reproduce paper-table rollouts and record ESS just before each resampling.

Runs both the learned beta=1 diffusion Euler solver and the beta=3 flow map
with identical initial particles, Brownian polynomial RNG seeds and K-group
systematic resampling. Does not recompute the expensive distribution metrics.
"""

import argparse
import json
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from scripts.fkc.sampling import _load, run_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--diffusion-run-dir', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    parser.add_argument('--k', type=int, nargs='+', default=[1, 8, 50])
    parser.add_argument('--particles', type=int, default=10000)
    parser.add_argument('--out-dir', type=Path, default=Path('results/fkc/diffusion_comparison'))
    parser.add_argument('--file-tag', default='',
                        help='filename suffix for additional NFE rows; avoids replacing previous ESS traces')
    args = parser.parse_args()
    module, checkpoint = _load(args.run_dir, None)
    config = OmegaConf.load(args.diffusion_run_dir / '.hydra' / 'config.yaml')
    diffusion = hydra.utils.instantiate(config.model)
    diffusion_checkpoint = args.diffusion_run_dir / 'checkpoints' / 'last.ckpt'
    diffusion.load_state_dict(torch.load(diffusion_checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    if diffusion.problem.powers != (1,) or not torch.equal(
        diffusion.problem.experts[0].means, module.problem.experts[0].means
    ):
        raise ValueError('Diffusion baseline must use beta=1 and identical modes.')
    diffusion.eval()
    module.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for seed in args.seeds:
            x0 = module.problem.sample_target(
                args.particles, module.loss.t0, generator=torch.Generator().manual_seed(seed)
            )
            rows = []
            for nfe in args.nfe:
                for k in args.k:
                    if args.particles % k:
                        raise ValueError(f'{args.particles} particles not divisible by K={k}.')
                    for sampler in ('learned', 'diffusion'):
                        trace: list[float] = []
                        cloud, _ = run_sampler(
                            sampler, nfe, x0, k, 1, problem=module.problem,
                            net=module.ema_net, diffusion_net=diffusion.ema_net,
                            schedule=module.loss.schedule, t0=module.loss.t0,
                            degree=module.loss.degree,
                            generator=torch.Generator().manual_seed(7919 * seed + 13),
                            ess_trace=trace,
                        )
                        row = {
                            'run': args.run_dir.name, 'sampler': sampler, 'seed': seed,
                            'nfe': nfe, 'k': k,
                            'mean_pre_resampling_ess_fraction': sum(trace) / len(trace),
                            'terminal_pre_resampling_ess_fraction': trace[-1],
                            'min_pre_resampling_ess_fraction': min(trace),
                            'finite_cloud': bool(torch.isfinite(cloud).all()),
                            'trace': trace,
                        }
                        rows.append(row)
                        print(json.dumps({field: value for field, value in row.items() if field != 'trace'}), flush=True)
            out = args.out_dir / f'ess_{args.run_dir.name}_seed{seed}{args.file_tag}.json'
            out.write_text(json.dumps({'checkpoint': str(checkpoint), 'diffusion_checkpoint': str(diffusion_checkpoint),
                                       'rows': rows}, indent=1))
            print(f'[ess] wrote {out}', flush=True)


if __name__ == '__main__':
    main()
