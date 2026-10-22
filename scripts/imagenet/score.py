"""Score every metric on stored ImageNet-256 sample sets in one pass.

Metrics (each image is read from disk once and fed to every model):

* ``fid50k``        -- torchmetrics / torch-fidelity Inception FID against
  ``imgnet256_torchmetrics.pt`` (the project FID protocol).
* ``fd_dinov2_50k`` -- EDM2-style FD-DINOv2 against the ImageNet-256 train
  reference from ``scripts.imagenet.compute_dinov2_ref``.
* ``imagereward``, ``pickscore``, ``hpsv2.1`` -- prompt-conditioned preference
  models on the caption ``"A photo of a/an {class name}"``.

Runs in the isolated ``scripts/imagenet/scoring_env`` environment (Python 3.11; old
transformers for ImageReward's vendored BLIP), which also pins the main env's
torchmetrics / torch-fidelity versions so FID is protocol-identical. Results are
cached per sample set under ``metrics/``; sets already carrying every requested
metric are skipped unless ``--overwrite``.

Example (from the repo root)::

    $SCORING_ENV/bin/python scripts/imagenet/score.py --eval-dir OUT/RUN/step_200000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


from scripts.imagenet.preference_models import BUILDERS, MODEL_IDS, PROMPT_NOTE, Scorer, class_means  # noqa: E402

from src.evaluation.frechet import (  # noqa: E402
    DINOV2_FEATURE_DIM,
    DINOV2_PROTOCOL,
    DINOv2Features,
    MomentAccumulator,
    frechet_distance,
    load_reference,
)
from src.evaluation.results import has_metric, metric_path, write_metric  # noqa: E402
from src.evaluation.sample_store import find_sample_sets, iter_shards  # noqa: E402

DEFAULT_FID_REF = Path(os.environ.get('FID_REF_STATS', ''))
DEFAULT_DINO_REF = Path(os.environ.get('DINO_REF_STATS', ''))
FID_PROTOCOL = 'tf-fidelity-native-uint8-v1'  # must match src.models.components.metrics
FID, FD_DINO = 'fid50k', 'fd_dinov2_50k'
PREFERENCE = tuple(BUILDERS)
ALL_METRICS = (FID, FD_DINO, *PREFERENCE)


class FrechetMetrics:
    """FID (torchmetrics) + FD-DINOv2 accumulators sharing one image stream."""

    def __init__(self, device: torch.device, fid_ref: Path, dino_ref: Path) -> None:
        from torchmetrics.image.fid import FrechetInceptionDistance

        stats = torch.load(fid_ref, map_location=device, weights_only=False)
        if stats.get('metadata', {}).get('protocol') != FID_PROTOCOL:
            raise ValueError(f'{fid_ref} is not a {FID_PROTOCOL} reference')
        self.fid = FrechetInceptionDistance(
            feature=2048, normalize=False, reset_real_features=False, antialias=False
        ).to(device)
        for key in ('real_features_sum', 'real_features_cov_sum', 'real_features_num_samples'):
            buffer = getattr(self.fid, key)
            buffer.copy_(stats[key].to(device=buffer.device, dtype=buffer.dtype))
        self.dino = DINOv2Features().to(device)
        self.dino_ref = load_reference(dino_ref, DINOV2_PROTOCOL)
        self.acc = MomentAccumulator(DINOV2_FEATURE_DIM)

    def reset(self) -> None:
        """Clear fake statistics (real references are kept)."""
        self.fid.reset()
        self.acc = MomentAccumulator(DINOV2_FEATURE_DIM)

    @torch.inference_mode()
    def update(self, images: torch.Tensor) -> None:
        """Accumulate one uint8 NCHW batch into both feature spaces."""
        self.fid.update(images, real=False)
        self.acc.update(self.dino(images))

    def compute(self) -> dict[str, float]:
        """FID and FD-DINOv2 of the accumulated set."""
        mean, cov = self.acc.moments()
        return {
            FID: float(self.fid.compute().item()),
            FD_DINO: frechet_distance(mean, cov, *self.dino_ref),
        }


@torch.inference_mode()
def stream_set(
    sample_set: Path,
    frechet: FrechetMetrics | None,
    scorers: dict[str, Scorer],
    device: torch.device,
    batch: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Read each shard once, feeding every batch to all requested metrics."""
    scores: dict[str, list[np.ndarray]] = {name: [] for name in scorers}
    labels, indices = [], []
    for shard in iter_shards(sample_set):
        images = torch.from_numpy(shard.images).permute(0, 3, 1, 2)
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch].to(device, non_blocking=True)
            if frechet is not None:
                frechet.update(chunk)
            for name, scorer in scorers.items():
                scores[name].append(
                    scorer(chunk, shard.labels[start : start + batch]).float().cpu().numpy()
                )
        labels.append(shard.labels)
        indices.append(shard.indices)
    per_image = {name: np.concatenate(values) for name, values in scores.items()}
    return per_image, np.concatenate(labels), np.concatenate(indices)


def write_results(
    sample_set: Path,
    frechet_values: dict[str, float],
    per_image: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    refs: dict[str, str],
) -> None:
    """Cache every metric JSON (and per-image preference scores) for one set."""
    count = int(len(labels))
    for name, value in frechet_values.items():
        protocol = FID_PROTOCOL if name == FID else DINOV2_PROTOCOL
        write_metric(
            sample_set, name, value, num_samples=count, protocol=protocol, reference=refs[name]
        )
    if per_image:
        np.savez(
            metric_path(sample_set, 'preference_scores').with_suffix('.npz'),
            labels=labels,
            indices=indices,
            **{k.replace('.', '_'): v for k, v in per_image.items()},
        )
    for name, values in per_image.items():
        write_metric(
            sample_set,
            name,
            float(values.mean()),
            std=float(values.std()),
            num_samples=int(values.size),
            model=MODEL_IDS[name],
            prompt_template=PROMPT_NOTE,
            per_class_mean=class_means(values, labels),
        )


def main() -> None:
    """Score every sample set under ``--eval-dir`` that lacks a requested metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', required=True, type=Path)
    parser.add_argument('--metrics', nargs='+', default=list(ALL_METRICS), choices=ALL_METRICS)
    parser.add_argument('--fid-ref', type=Path, default=DEFAULT_FID_REF)
    parser.add_argument('--dino-ref', type=Path, default=DEFAULT_DINO_REF)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    sets = [
        s
        for s in find_sample_sets(args.eval_dir)
        if args.overwrite or not all(has_metric(s, m) for m in args.metrics)
    ]
    print(f'{len(sets)} sample sets to score under {args.eval_dir}: {args.metrics}', flush=True)
    if not sets:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wants_frechet = bool({FID, FD_DINO} & set(args.metrics))
    frechet = FrechetMetrics(device, args.fid_ref, args.dino_ref) if wants_frechet else None
    scorers = {name: BUILDERS[name](device) for name in args.metrics if name in PREFERENCE}
    refs = {FID: str(args.fid_ref), FD_DINO: str(args.dino_ref)}

    for sample_set in sets:
        if frechet is not None:
            frechet.reset()
        per_image, labels, indices = stream_set(
            sample_set, frechet, scorers, device, args.batch_size
        )
        frechet_values = frechet.compute() if frechet is not None else {}
        frechet_values = {k: v for k, v in frechet_values.items() if k in args.metrics}
        write_results(sample_set, frechet_values, per_image, labels, indices, refs)
        shown = {**frechet_values, **{k: float(v.mean()) for k, v in per_image.items()}}
        print(f'{sample_set}: ' + ' '.join(f'{k}={v:.4f}' for k, v in shown.items()), flush=True)


if __name__ == '__main__':
    main()
