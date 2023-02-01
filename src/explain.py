from typing import List, Optional, Tuple

import hydra
import torch
import pyrootutils
import numpy as np
import pytorch_lightning as pl
from rich.progress import track
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

    for model in (models.model_1,models.model_2,models.model_3):
        log.info(f"Applying attribution methods for <{model.__class__.__name__}> model")

        xai_methods = XAIMethodsModule(cfg, model, x_batch)
        attr_single = None

        log.info(f"Batch-based attribution methods")
        attr_batch = xai_methods.attribute_batch(x_batch,y_batch)

        log.info(f"Single-observation-based attribution methods")

        for x,y in track(zip(x_batch, y_batch)):
            attr = xai_methods.attribute_single(x.unsqueeze(0),y)

            if attr_single is not None:
                attr_single = np.vstack((attr_single,attr))
            else:
                attr_single = attr

        if attr_batch is not None:
            attr_total.append(np.hstack((attr_batch, attr_single)))
        else:
            attr_total.append(attr_single)

    attr_total = np.swapaxes(np.array(attr_total),0,1) # obs, models, XAI, c, w, h
    # Insert time anyhow
    np.savez(str(cfg.paths.root_dir) + "/data/attribution_maps/" + cfg.data.modality + "/attr_" + str(datamodule.__name__) + "_dataset_" + str(attr_total.shape[2]) + "_methods_.npz", attr_total)




@hydra.main(version_base="1.3", config_path="../configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # explain the model
    explain(cfg)


if __name__ == "__main__":
    main()
