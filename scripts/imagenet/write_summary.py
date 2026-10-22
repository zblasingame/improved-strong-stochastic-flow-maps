"""Write a local JSON summary of cached ImageNet-256 evaluation metrics.

No wandb run is created; the per-NFE metric files and this summary remain
alongside the stored images for later comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import rootutils

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.evaluation.results import summarize_sample_set  # noqa: E402
from src.evaluation.sample_store import (  # noqa: E402
    find_sample_sets,
    read_manifest,
    write_json_atomic,
)


def collect(eval_dir: Path) -> dict[str, Any]:
    """Collect the sample-set metrics and provenance under ``eval_dir``."""
    sets = find_sample_sets(eval_dir)
    if not sets:
        raise FileNotFoundError(f'no sample sets under {eval_dir}')
    manifest = read_manifest(sets[0])
    rows = [summarize_sample_set(s) | {'sampler': read_manifest(s)['sampler']} for s in sets]
    source = manifest['source']
    epoch = source.get('checkpoint_epoch')
    return {
        'source': source,
        'best_epoch': epoch + 1 if epoch is not None else None,
        'plan': manifest['plan'],
        'git_commit': manifest['git_commit'],
        'results': sorted(rows, key=lambda row: (row['weights'], row['rollout'], row['nfe'])),
    }


def main() -> None:
    """Write ``summary.json`` without contacting any remote logging service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True, type=Path)
    parser.add_argument('--durable-summary', type=Path, default=None)
    args = parser.parse_args()
    summary = collect(args.eval_dir)
    path = args.eval_dir / 'summary.json'
    write_json_atomic(path, summary)
    if args.durable_summary is not None:
        write_json_atomic(args.durable_summary, summary)
    print(f'Wrote {len(summary["results"])} sample sets to {path}', flush=True)


if __name__ == '__main__':
    main()
