from typing import List, Optional, Tuple

import hydra
import torch
import pyrootutils
import numpy as np
import pytorch_lightning as pl
from tqdm.auto import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from modules.models import ModelsModule
from modules.xai_methods import XAIMethodsModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def explain(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()

    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    log.info(f"Instantiating models for <{cfg.data.modality}> data")
    models = ModelsModule(cfg)
    attr_total = []

    log.info(f"Starting Attribution computation over each Model and XAI Method")
    for model in tqdm(
        models.models,
        desc=f"Attribution for {cfg.data.modality} Models",
        colour="BLUE",
        position=0,
        leave=True,
    ):

        xai_methods = XAIMethodsModule(cfg, model, x_batch)

        attr = xai_methods.attribute(x_batch, y_batch)

        attr_total.append(attr)  # obs , XAI, c, w, h

    np.savez(
        str(cfg.paths.root_dir)
        + "/data/attribution_maps/"
        + cfg.data.modality
        + "/attr_"
        + str(datamodule.__name__)
        + "_dataset_"
        + str(attr_total[0].shape[1])
        + "_methods_"
        + cfg.time
        + ".npz",
        attr_total[0],
        attr_total[1],
        attr_total[2],
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # explain the model
    explain(cfg)


if __name__ == "__main__":
    main()
