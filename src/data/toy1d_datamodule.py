"""Synthetic 1D noise `LightningDataModule` for the non-linear SDE toy.

The 1D toy of arXiv:2606.01086 Section 4.1 has no dataset: training states
``X_s ~ p_s`` are simulated on the fly from the known SDE inside the loss.
The datamodule therefore only supplies batches of standard-normal "data"
of shape ``(1,)`` (plus a dummy label) so that the shared
:class:`~src.models.ssfm_module.SFMLitModule` batch contract ``(x_1, labels)``
holds; the loss re-derives ``X_0`` from the ``x_0`` noise draw.
"""

from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset


class Toy1DNoiseDataModule(LightningDataModule):
    """On-the-fly 1D standard-normal datamodule (shape template only)."""

    def __init__(
        self,
        train_val_test_split: tuple[int, int, int] = (100_000, 4_096, 4_096),
        batch_size: int = 4096,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialize a `Toy1DNoiseDataModule`.

        Parameters
        ----------
        train_val_test_split : tuple[int, int, int], optional
            Sizes of the train/val/test noise draws.
        batch_size : int, optional
            Global batch size (divided across devices), by default 4096.
        num_workers : int, optional
            Dataloader workers, by default 0.
        pin_memory : bool, optional
            Whether to pin memory in dataloaders, by default False.
        seed : int, optional
            Generator seed used when drawing the three splits.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Return ``1`` (unconditional dummy label)."""
        return 1

    @property
    def dim(self) -> int:
        """Return the data dimensionality."""
        return 1

    def prepare_data(self) -> None:
        """No download; synthetic noise is generated in :meth:`setup`."""

    def setup(self, stage: str | None = None) -> None:
        """Draw the train/val/test noise splits if not already present."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f'Batch size ({self.hparams.batch_size}) is not divisible by '
                    f'the number of devices ({self.trainer.world_size}).'
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train or self.data_val or self.data_test:
            return
        generator = torch.Generator().manual_seed(self.hparams.seed)
        n_train, n_val, n_test = self.hparams.train_val_test_split
        self.data_train = self._split(n_train, generator)
        self.data_val = self._split(n_val, generator)
        self.data_test = self._split(n_test, generator)

    def _split(self, n: int, generator: torch.Generator) -> TensorDataset:
        """Draw ``n`` standard-normal 1D samples with a dummy zero label."""
        x = torch.randn(n, 1, generator=generator)
        return TensorDataset(x, torch.zeros(n, dtype=torch.long))

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
