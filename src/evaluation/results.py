"""Per-metric result cache stored beside each sample set.

Each scorer writes ``{sample_set}/metrics/{metric}.json`` with at least a
``value`` field; the wandb aggregator collects them. Adding a metric later only
requires a new scorer writing another file -- no regeneration.

Stdlib-only: imported by the isolated preference-scoring environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.sample_store import read_manifest, write_json_atomic

METRICS_DIR = 'metrics'


def metric_path(sample_set: Path, metric: str) -> Path:
    """Location of one metric's cached result."""
    return Path(sample_set) / METRICS_DIR / f'{metric}.json'


def has_metric(sample_set: Path, metric: str) -> bool:
    """Whether ``metric`` has already been computed for ``sample_set``."""
    return metric_path(sample_set, metric).exists()


def write_metric(sample_set: Path, metric: str, value: float, **details: Any) -> None:
    """Atomically cache one metric result (``value`` plus free-form details)."""
    write_json_atomic(
        metric_path(sample_set, metric), {'metric': metric, 'value': value, **details}
    )


def read_metrics(sample_set: Path) -> dict[str, dict[str, Any]]:
    """Load every cached metric for a sample set, keyed by metric name."""
    directory = Path(sample_set) / METRICS_DIR
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob('*.json'))}


def summarize_sample_set(sample_set: Path) -> dict[str, Any]:
    """Flat summary: identifying manifest fields plus every metric value."""
    manifest = read_manifest(sample_set)
    return {
        'weights': manifest['weights'],
        'rollout': manifest['rollout'],
        'nfe': manifest['nfe'],
        'steps': manifest.get('steps', manifest['nfe']),
        'actual_nfe': manifest.get('actual_nfe', manifest['nfe']),
        'metrics': {name: payload['value'] for name, payload in read_metrics(sample_set).items()},
    }
