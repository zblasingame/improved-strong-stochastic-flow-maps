"""Aggregate the four-seed log-weight clipping sweep with the existing unclipped control.

Writes one CSV row per sampler/NFE/K/total clipped fraction. ``valid`` counts
the runs for which ALL five distribution metrics are finite and the cloud was
not marked divergent; ESS is averaged over those same seeds only. Unclipped
control rows come from the previous paper-table and pre-resampling ESS JSONs.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.fkc.metrics_table import METRICS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-dir', type=Path, default=Path('results/fkc/clip_sweep'))
    parser.add_argument('--baseline-dir', type=Path, default=Path('results/fkc/diffusion_comparison'))
    parser.add_argument('--flow-map-dir', type=Path, default=Path('results/fkc/table'))
    parser.add_argument('--run', default='fkc40_issfm', help='Name of the iSSFM run directory.')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--out', type=Path, default=Path('results/fkc/clip_sweep/summary.csv'))
    args = parser.parse_args()

    metric_rows = {}
    ess_rows = {}
    for directory in (args.baseline_dir, args.flow_map_dir):
        for path in directory.glob(f'{args.run}_seed*.json'):
            for row in json.loads(path.read_text())['rows']:
                if row['seed'] in args.seeds:
                    key = (row.get('sampler', 'learned'), row['nfe'], row['k'], 0.0, row['seed'])
                    if key in metric_rows:
                        raise ValueError(f'duplicate unclipped result {key}')
                    metric_rows[key] = row
    for path in args.baseline_dir.glob(f'ess_{args.run}_seed*.json'):
        for row in json.loads(path.read_text())['rows']:
            if row['seed'] in args.seeds:
                key = (row['sampler'], row['nfe'], row['k'], 0.0, row['seed'])
                if key in ess_rows:
                    raise ValueError(f'duplicate unclipped ESS {key}')
                ess_rows[key] = row
    for path in args.sweep_dir.glob(f'{args.run}_seed*.json'):
        for row in json.loads(path.read_text())['rows']:
            if row['seed'] in args.seeds:
                key = (row['sampler'], row['nfe'], row['k'], row['clip_total'], row['seed'])
                if key in metric_rows:
                    raise ValueError(f'duplicate clipped result {key}')
                metric_rows[key] = row
                ess_rows[key] = row

    groups = defaultdict(dict)
    for (sampler, nfe, k, clip, seed), row in metric_rows.items():
        groups[(nfe, k, clip, sampler)][seed] = row

    stats = []
    for (nfe, k, clip, sampler), seeds in sorted(groups.items()):
        seen = set(seeds)
        complete = seen == set(args.seeds)
        valid = {seed: row for seed, row in seeds.items() if not row.get('diverged')
                 and all(m in row and math.isfinite(row[m]) for m in METRICS)}
        out = {'nfe': nfe, 'k': k, 'clip_total': clip, 'sampler': sampler,
               'valid': len(valid), 'total_seeds': len(seen), 'complete': complete}
        for metric in (*METRICS, 'mean_pre_resampling_ess_fraction',
                       'terminal_pre_resampling_ess_fraction'):
            values = [
                ess_rows[(sampler, nfe, k, clip, seed)][metric]
                if 'ess_fraction' in metric else row[metric]
                for seed, row in valid.items()
                if 'ess_fraction' not in metric or
                ((sampler, nfe, k, clip, seed) in ess_rows and
                 math.isfinite(ess_rows[(sampler, nfe, k, clip, seed)][metric]))
            ]
            out[f'{metric}_mean'] = sum(values) / len(values) if values else float('nan')
            out[f'{metric}_std'] = (
                (sum((v - out[f'{metric}_mean']) ** 2 for v in values) / len(values)) ** .5
                if values else float('nan')
            )
        stats.append(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)
    print(f'{len(stats)} aggregate rows -> {args.out}; incomplete groups: '
          f'{sum(not row["complete"] for row in stats)}')
    for row in stats:
        if row['k'] == 8 and row['nfe'] in (12, 20, 24, 32) and row['complete']:
            print(f'NFE {row["nfe"]:2d} {row["sampler"]:9s} clip {row["clip_total"]:.2f} '
                  f'valid {row["valid"]}/{row["total_seeds"]} '
                  f'eW2={row["energy_w2_mean"]:.3g} TV={row["total_variation_mean"]:.3f} '
                  f'ESS/K={row["mean_pre_resampling_ess_fraction_mean"]:.3f}')


if __name__ == '__main__':
    main()
