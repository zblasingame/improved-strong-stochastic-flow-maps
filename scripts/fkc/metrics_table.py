"""Table-A1-style metrics (Skreta et al. 2025) for a trained FKC flow map.

Protocol follows the reference notebook ``gmm_temp_annealed_birth_death.ipynb``:
10k samples per run, metrics in UNNORMALIZED coordinates (x50), total
variation on a 200x200 grid set by the test sample, MMD = biased
``mix_rbf_mmd2`` with ``sigma_list = 10**linspace(-2, 0, 10)``, energy W2 =
1D W2 between ``log pi_1`` values of generated and test samples.

Deviations (stated in the output):
  * W1/W2 use exact assignment on 3 x 2500 random subsamples instead of POT's
    full 10k ``emd2`` (POT is not installed); this biases W1/W2 slightly UP.
  * MMD uses a 3000-point subsample of each cloud (kernel cost).

Rows: ``Target Score`` = learned map, no resampling; ``Target Score + FKC
(K=k)`` = systematic resampling within independent groups of k particles after
every step, terminal draw by weight (``scripts.fkc.sampling.run_sampler``).

Run mode writes one JSON per (run, seed); ``--aggregate`` merges JSONs and
prints mean +- std tables.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

from scripts.fkc.sampling import _load, run_sampler  # noqa: E402

from src.models.components.fkc.evaluation import energy_values, energy_wasserstein  # noqa: E402

SCALE = 50.0
METRICS = ('energy_w2', 'mmd', 'total_variation', 'w1', 'w2')


def mmd(a: torch.Tensor, b: torch.Tensor, n: int = 3000) -> float:
    g = torch.Generator().manual_seed(0)
    a = a[torch.randperm(a.shape[0], generator=g)[:n]].double()
    b = b[torch.randperm(b.shape[0], generator=g)[:n]].double()
    sigmas = 10 ** np.linspace(-2, 0, 10)

    def kernel(x, y):
        d = torch.cdist(x, y).pow(2)
        return sum(torch.exp(-d / (2 * s * s)) for s in sigmas)

    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean())


def total_variation(a: torch.Tensor, b: torch.Tensor) -> float:
    h_b, xb, yb = np.histogram2d(b[:, 0].numpy(), b[:, 1].numpy(), bins=200)
    h_a, _, _ = np.histogram2d(a[:, 0].numpy(), a[:, 1].numpy(), bins=(xb, yb))
    return float(0.5 * np.abs(h_a / h_a.sum() - h_b / h_b.sum()).sum())


def wasserstein(a: torch.Tensor, b: torch.Tensor, p: int, n: int = 2500, repeats: int = 3) -> float:
    g = torch.Generator().manual_seed(1)
    values = []
    for _ in range(repeats):
        ia = torch.randperm(a.shape[0], generator=g)[:n]
        ib = torch.randperm(b.shape[0], generator=g)[:n]
        cost = torch.cdist(a[ia].double(), b[ib].double())
        cost = cost.pow(2) if p == 2 else cost
        r, c = linear_sum_assignment(cost.numpy())
        v = float(cost[r, c].mean())
        values.append(math.sqrt(v) if p == 2 else v)
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> None:
    module, ckpt = _load(args.run_dir, args.checkpoint)
    problem, loss, net = module.problem, module.loss, module.ema_net
    diffusion_net = None
    if args.diffusion_run_dir is not None:
        config = OmegaConf.load(args.diffusion_run_dir / '.hydra' / 'config.yaml')
        diffusion = hydra.utils.instantiate(config.model)
        diffusion_ckpt = args.diffusion_run_dir / 'checkpoints' / 'last.ckpt'
        diffusion.load_state_dict(
            torch.load(diffusion_ckpt, map_location='cpu', weights_only=False)['state_dict']
        )
        if diffusion.problem.powers != (1,):
            raise ValueError('Diffusion baseline must be trained at beta=1.')
        if not torch.equal(diffusion.problem.experts[0].means, problem.experts[0].means):
            raise ValueError('Diffusion baseline and flow map must use identical Gaussian modes.')
        diffusion_net = diffusion.eval().ema_net
        print(f'[table] learned diffusion checkpoint: {diffusion_ckpt}', flush=True)
    n, seed = args.particles, args.seed
    reference = problem.sample_target(n, 1.0, generator=torch.Generator().manual_seed(1000 + seed))
    ref_energy = energy_values(problem, reference).cpu()
    ref_u = reference.cpu() * SCALE
    x0 = problem.sample_target(n, loss.t0, generator=torch.Generator().manual_seed(seed))
    rows = []
    for steps in args.nfe:
        for sampler in (
            ['diffusion'] if args.diffusion_only else
            ['learned', 'diffusion'] if diffusion_net is not None else ['learned']
        ):
          for k in [1, *args.k]:
            with torch.no_grad():
                cloud, _ = run_sampler(
                    sampler, steps, x0, k, 1, problem=problem, net=net,
                    schedule=loss.schedule, t0=loss.t0, degree=loss.degree,
                    generator=torch.Generator().manual_seed(7919 * seed + 13),
                    diffusion_net=diffusion_net,
                )
            row = {'run': args.run_dir.name, 'sampler': sampler, 'seed': seed, 'nfe': steps, 'k': k}
            if not bool(torch.isfinite(cloud).all()) or float(cloud.abs().max()) > 1e3:
                row['diverged'] = True
            else:
                cu = cloud.cpu() * SCALE
                row.update(
                    energy_w2=energy_wasserstein(ref_energy, energy_values(problem, cloud).cpu(), order=2),
                    mmd=mmd(cu, ref_u),
                    total_variation=total_variation(cu, ref_u),
                    w1=wasserstein(cu, ref_u, 1),
                    w2=wasserstein(cu, ref_u, 2),
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = args.out_dir / f'{args.run_dir.name}_seed{seed}{args.file_tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'checkpoint': str(ckpt), 'rows': rows}, indent=1))
    print(f'[table] wrote {out}')


def aggregate(args: argparse.Namespace) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    diverged: dict[tuple, int] = defaultdict(int)
    ess_rows: dict[tuple, list[dict]] = defaultdict(list)
    if args.ess_dir is not None:
        for path in sorted(args.ess_dir.glob('ess_*.json')):
            for row in json.loads(path.read_text())['rows']:
                if args.include_seeds is not None and row['seed'] not in args.include_seeds:
                    continue
                if args.include_runs is not None and row['run'] not in args.include_runs:
                    continue
                key = (row['run'], row['nfe'], row['k'], row['sampler'])
                ess_rows[key].append(row)
    paths = sorted(path for path in args.out_dir.glob('*.json') if not path.name.startswith('ess_'))
    if args.extra_dir is not None:
        paths.extend(sorted(args.extra_dir.glob('*.json')))
    for path in paths:
        for row in json.loads(path.read_text())['rows']:
            if args.include_seeds is not None and row['seed'] not in args.include_seeds:
                continue
            if args.include_runs is not None and row['run'] not in args.include_runs:
                continue
            key = (row['run'], row['nfe'], row['k'], row.get('sampler', 'learned'))
            if row.get('diverged') or any(not math.isfinite(row[m]) for m in METRICS):
                diverged[key] += 1
            else:
                groups[key].append(row)
    header = f'{"run":14s} {"NFE":>4s}  {"SDE":34s} {"n":>2s}  ' + '  '.join(
        f'{m:>17s}' for m in ('Energy-W2', 'MMD', 'Total Var', 'W1', 'W2')
    )
    if args.ess_dir is not None:
        header += f'  {"mean ESS/K":>15s}  {"last ESS/K":>15s}'
    lines = [header]
    for key in sorted(set(groups) | set(diverged)):
        run_name, nfe, k, sampler = key
        name = 'diffusion EM' if sampler == 'diffusion' else 'flow map'
        label = f'{name} target score' if k == 1 else f'{name} + FKC K={k}'
        rows = groups.get(key, [])
        cells = []
        for m in METRICS:
            if rows:
                v = torch.tensor([r[m] for r in rows], dtype=torch.float64)
                if m == 'mmd':
                    cells.append(f'{float(v.mean()):8.4f} ± {float(v.std(unbiased=False)):6.4f}')
                else:
                    cells.append(f'{float(v.mean()):8.3f} ± {float(v.std(unbiased=False)):6.3f}')
            else:
                cells.append(f'{"diverged":>17s}')
        if args.ess_dir is not None:
            valid_seeds = {row['seed'] for row in rows}
            for m in ('mean_pre_resampling_ess_fraction', 'terminal_pre_resampling_ess_fraction'):
                v = torch.tensor(
                    [r[m] for r in ess_rows.get(key, []) if r['seed'] in valid_seeds],
                    dtype=torch.float64,
                )
                if v.numel() and bool(torch.isfinite(v).all()):
                    cells.append(f'{float(v.mean()):7.3f} ± {float(v.std(unbiased=False)):5.3f}')
                else:
                    cells.append(f'{"unavailable":>15s}')
        note = f'  ({diverged[key]} diverged)' if diverged.get(key) else ''
        lines.append(f'{run_name:14s} {nfe:4d}  {label:34s} {len(rows):2d}  ' + '  '.join(cells) + note)
    output = '\n'.join(lines) + '\n'
    print(output, end='')
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--diffusion-run-dir', type=Path, default=None)
    parser.add_argument('--diffusion-only', action='store_true',
                        help='reuse existing flow-map seed metrics via --extra-dir when aggregating')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nfe', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    parser.add_argument('--k', type=int, nargs='+', default=[8, 50])
    parser.add_argument('--particles', type=int, default=10000)
    parser.add_argument('--out-dir', type=Path, default=Path('results/fkc/table'))
    parser.add_argument('--file-tag', default='',
                        help='filename suffix for additional NFE rows; avoids replacing previous per-seed results')
    parser.add_argument('--extra-dir', type=Path, default=None,
                        help='additional per-seed JSONs for --aggregate (e.g. previous flow-map runs)')
    parser.add_argument('--ess-dir', type=Path, default=None,
                        help='directory with pre-resampling ESS traces from scripts/fkc/ess.py')
    parser.add_argument('--include-seeds', type=int, nargs='+', default=None,
                        help='only aggregate these seeds, e.g. 0 1 2 3 to omit seed 4 for both arms')
    parser.add_argument('--include-runs', nargs='+', default=None,
                        help='only show the specified flow-map run(s) and their matched diffusion comparisons')
    parser.add_argument('--summary-out', type=Path, default=None,
                        help='write the same aggregated table to this file')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
