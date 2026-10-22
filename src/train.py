from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

# PyTorch 2.6+ defaults `torch.load(weights_only=True)`, which blocks unpickling
# of omegaconf containers and their metadata embedded in Lightning checkpoint
# hyperparameters. The allowlist approach is a losing game of whack-a-mole
# (ListConfig, DictConfig, ContainerMetadata, Metadata, typing.Any, list, ...)
# because omegaconf serializes a deep tree of nested Python/typing builtins.
# Training only loads checkpoints it wrote itself (resume) and the pretrained
# backbones you point it at, so we restore the pre-2.6 default of
# `weights_only=False`. Only load checkpoints from sources you trust.
_original_torch_load = torch.load


def _torch_load_no_weights_only(*args, **kwargs):
    # Lightning's `cloud_io._load` passes `weights_only=None` (its own default)
    # which torch.load treats as the 2.6 default `True`. Treat `None` and
    # "unset" the same way: fall back to `weights_only=False` for trusted local
    # checkpoints. An explicit `True`/`False` from the caller is respected.
    if kwargs.get('weights_only', None) is None:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_no_weights_only
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _configure_torch_performance() -> None:
    """Enable Tensor Core and cuDNN fast paths for fixed-shape training."""
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get('seed'):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f'Instantiating datamodule <{cfg.data._target_}>')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f'Instantiating model <{cfg.model._target_}>')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info('Instantiating callbacks...')
    callbacks: list[Callback] = instantiate_callbacks(cfg.get('callbacks'))

    log.info('Instantiating loggers...')
    logger: list[Logger] = instantiate_loggers(cfg.get('logger'))

    log.info(f'Instantiating trainer <{cfg.trainer._target_}>')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        'cfg': cfg,
        'datamodule': datamodule,
        'model': model,
        'callbacks': callbacks,
        'logger': logger,
        'trainer': trainer,
    }

    if logger:
        log.info('Logging hyperparameters!')
        log_hyperparameters(object_dict)

    if cfg.get('train'):
        log.info('Starting training!')
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))

    train_metrics = trainer.callback_metrics

    if cfg.get('test'):
        log.info('Starting testing!')
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == '':
            log.warning('Best ckpt not found! Using current weights for testing...')
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f'Best ckpt path: {ckpt_path}')

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base='1.3', config_path='../configs', config_name='train.yaml')
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: float | None with optimized metric value.
    """
    _configure_torch_performance()

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get('optimized_metric')
    )

    # return optimized metric
    return metric_value


if __name__ == '__main__':
    main()
