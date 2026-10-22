"""Placeholder datamodule for the analytic FKC toy experiments.

FKC training samples states directly from the exact analytic marginals of the
:class:`FKCProblem`, so no data distribution is needed: the datamodule only
defines how many optimizer steps run and what batch size they use. Each item is
a single dummy scalar; the Lightning module reads ``batch[0].shape[0]``.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset


class FKCToy2DDataModule(LightningDataModule):
    """Length-only datamodule driving analytic FKC training."""

    def __init__(
        self,
        batch_size: int = 4096,
        train_size: int = 409_600,
        val_size: int = 4_096,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialize the datamodule.

        :param batch_size: Global batch size (split across devices).
        :param train_size: Number of dummy training items.
        :param val_size: Number of dummy validation items.
        :param num_workers: DataLoader workers.
        :param pin_memory: Whether to pin memory.
        :param seed: Unused; kept for config symmetry with the toy datamodule.
        """
        super().__init__()
        if batch_size < 1:
            raise ValueError(f'batch_size must be positive; got {batch_size}.')
        if train_size < 1 or val_size < 1:
            raise ValueError('train_size and val_size must be positive.')
        self.save_hyperparameters(logger=False)
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """No download; the placeholders are built in :meth:`setup`."""

    def setup(self, stage: str | None = None) -> None:
        """Build the dummy index tensors once."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f'Batch size ({self.hparams.batch_size}) is not divisible by '
                    f'the number of devices ({self.trainer.world_size}).'
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if self.data_train is not None or self.data_val is not None:
            return
        self.data_train = TensorDataset(torch.zeros(self.hparams.train_size, 1))
        self.data_val = TensorDataset(torch.zeros(self.hparams.val_size, 1))

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the shuffled placeholder training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the placeholder validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the placeholder test dataloader (reuses the validation split)."""
        return self.val_dataloader()


if __name__ == '__main__':
    _ = FKCToy2DDataModule()
