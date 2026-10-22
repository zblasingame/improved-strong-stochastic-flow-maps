"""ImageNet DataModule backed by DMF-compatible VAE posterior moments."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from lightning import LightningDataModule
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.components.latent_store import LATENTS_SCALE, Manifest, load_manifest
from src.models.components.gaussian import randn_like


def sample_posterior(
    moments: Tensor, latents_scale: float = LATENTS_SCALE, latents_bias: float = 0.0
) -> Tensor:
    """Sample float32 VAE latents from concatenated posterior mean and std.

    Parameters
    ----------
    moments : torch.Tensor
        Posterior moments with mean and std concatenated along the channel
        dimension (``dim=-3``).
    latents_scale : float, optional
        Checkpoint-matched scale applied to the sampled latents, by default
        :data:`~src.data.components.latent_store.LATENTS_SCALE`.
    latents_bias : float, optional
        Shift applied after scaling, by default ``0.0``.

    Returns
    -------
    torch.Tensor
        Float32 latents with half the input channel count.
    """
    mean, std = moments.float().chunk(2, dim=-3)
    return (mean + std * randn_like(mean)) * latents_scale + latents_bias


class LatentShardDataset(Dataset[tuple[Tensor, Tensor]]):
    """Read validated safetensors shards lazily, caching one shard per worker."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: Manifest = load_manifest(self.manifest_path)
        missing = [
            shard.filename
            for shard in self.manifest.shards
            if not (self.manifest_path.parent / shard.filename).is_file()
        ]
        if missing:
            raise ValueError(f'manifest references missing shard: {missing[0]}')

        # Cumulative shard end offsets enable O(log n) global-index lookup.
        self._ends: list[int] = []
        total = 0
        for shard in self.manifest.shards:
            total += shard.count
            self._ends.append(total)

        self._cached_id: int | None = None
        self._cached_tensors: dict[str, Tensor] | None = None

    def __len__(self) -> int:
        """Return the total sample count across every shard.

        Returns
        -------
        int
            Number of samples declared by the manifest.
        """
        return self.manifest.total_count

    def __getstate__(self) -> dict[str, Any]:
        """Drop the shard cache so the dataset pickles cheaply to workers.

        Returns
        -------
        dict
            Instance state with the cached shard cleared.
        """
        state = self.__dict__.copy()
        state['_cached_id'] = None
        state['_cached_tensors'] = None
        return state

    def _load_shard(self, shard_id: int) -> dict[str, Tensor]:
        cached = self._cached_tensors
        if self._cached_id != shard_id or cached is None:
            filename = self.manifest.shards[shard_id].filename
            cached = load_file(str(self.manifest_path.parent / filename))
            self._cached_tensors = cached
            self._cached_id = shard_id
        return cached

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return the sampled latent and label for a global sample index.

        Parameters
        ----------
        index : int
            Global sample index in ``[0, len(self))``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The sampled latent and its class label.

        Raises
        ------
        IndexError
            If ``index`` is outside the dataset range.
        """
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_id = bisect_right(self._ends, index)
        start = 0 if shard_id == 0 else self._ends[shard_id - 1]
        tensors = self._load_shard(shard_id)
        local_index = index - start
        return sample_posterior(tensors['moments'][local_index]), tensors['labels'][local_index]


class ShardBatchSampler(Sampler[list[int]]):
    """Form batches within shards so each worker loads at most one shard per batch."""

    def __init__(
        self,
        dataset: LatentShardDataset,
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not 0 <= rank < world_size:
            raise ValueError(f'rank must be in [0, {world_size}); got {rank}')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def _batches(self, epoch: int) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        shard_ids = list(range(len(self.dataset.manifest.shards)))
        if self.shuffle:
            order = torch.randperm(len(shard_ids), generator=generator).tolist()
            shard_ids = [shard_ids[index] for index in order]

        batches: list[list[int]] = []
        starts = [0, *self.dataset._ends[:-1]]
        for shard_id in shard_ids:
            count = self.dataset.manifest.shards[shard_id].count
            local = (
                torch.randperm(count, generator=generator).tolist()
                if self.shuffle
                else list(range(count))
            )
            indices = [starts[shard_id] + index for index in local]
            for offset in range(0, count, self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        usable = len(batches) - len(batches) % self.world_size
        return batches[:usable][self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's shard-local batches for the next epoch."""
        batches = self._batches(self.epoch)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        """Return the number of complete distributed batches on this rank."""
        count = sum(
            shard.count // self.batch_size
            if self.drop_last
            else (shard.count + self.batch_size - 1) // self.batch_size
            for shard in self.dataset.manifest.shards
        )
        return count // self.world_size


class LatentImageNetDataModule(LightningDataModule):
    """Serve latent ImageNet with project time convention: zero noise, one data."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 2,
        multiprocessing_context: str | None = 'spawn',
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of ImageNet classes.

        Returns
        -------
        int
            ``1000``.
        """
        return 1000

    def setup(self, stage: str | None = None) -> None:
        """Build the train and validation shard datasets.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage. Unused; both splits are built eagerly.

        Raises
        ------
        RuntimeError
            If the global batch size is not divisible by the world size.
        """
        del stage
        world_size = self.trainer.world_size if self.trainer is not None else 1
        if self.hparams.batch_size % world_size:
            raise RuntimeError(
                f'batch size {self.hparams.batch_size} is not divisible by {world_size} devices.'
            )
        self.batch_size_per_device = self.hparams.batch_size // world_size
        root = Path(self.hparams.data_dir)
        self.data_train = LatentShardDataset(root / 'train' / 'manifest.json')
        validation = root / 'validation' / 'manifest.json'
        # Latent dumps may ship train-only; reuse train as val to keep hooks working.
        self.data_val = LatentShardDataset(validation) if validation.exists() else self.data_train

    def _loader(self, dataset: Dataset, *, shuffle: bool) -> DataLoader:
        workers = self.hparams.num_workers
        options: dict[str, Any] = {}
        # prefetch_factor is only accepted by DataLoader when workers are used.
        if workers and self.hparams.prefetch_factor is not None:
            options['prefetch_factor'] = self.hparams.prefetch_factor
        batch_sampler = ShardBatchSampler(
            dataset,
            self.batch_size_per_device,
            shuffle=shuffle,
            drop_last=shuffle,
            rank=self.trainer.global_rank if self.trainer is not None else 0,
            world_size=self.trainer.world_size if self.trainer is not None else 1,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and workers > 0,
            multiprocessing_context=self.hparams.multiprocessing_context if workers > 0 else None,
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        """Create and return the shuffled training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        """
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader, which mirrors validation.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation dataloader.
        """
        return self.val_dataloader()
