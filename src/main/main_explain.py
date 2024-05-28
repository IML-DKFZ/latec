from typing import List, Optional, Tuple
import os

import hydra
import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.modules.models import ModelsModule
from src.modules.xai_methods import XAIMethodsModule
from src import utils

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def explain(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load dataloader for selected dataset
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()

    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    # Load pretrained models for dataset
    log.info(f"Instantiating models for <{cfg.data.modality}> data")
    models = ModelsModule(cfg)
    explain_data = []

    # Evaluation loop over model and chunk of obseravtions
    log.info(f"Starting saliency map computation over each Model and XAI Method")
    for model in tqdm(
        models.models,
        desc=f"Attribution for {cfg.data.modality} Models",
        colour="BLUE",
        position=0,
        leave=True,
    ):
        # Load XAI methods module
        xai_methods = XAIMethodsModule(cfg, model, x_batch)

        explain_data_model = []
        for idx_chunk in tqdm(
            range(0, x_batch.size(0), cfg.chunk_size),
            desc=f"Chunkwise (n={cfg.chunk_size}) Computation",
            colour="CYAN",
            position=1,
            leave=True,
        ):
            # Explain chunk
            explain_data_model.append(
                xai_methods.attribute(
                    x_batch[idx_chunk : idx_chunk + cfg.chunk_size],
                    y_batch[idx_chunk : idx_chunk + cfg.chunk_size],
                )
            )

        explain_data.append(np.vstack(explain_data_model))  # obs, XAI, c, w, h

    np.savez(
        str(cfg.paths.data_dir)
        + "/saliency_maps/"
        + cfg.data.modality
        + "/explain_"
        + str(datamodule.__name__)
        + "_"
        + str(explain_data[0].shape[1])
        + "_methods_"
        + cfg.time
        + ".npz",
        explain_data[0],
        explain_data[1],
        explain_data[2],
    )

@hydra.main(version_base="1.3", config_path= os.getcwd() + "/configs", config_name="explain.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    explain(cfg)


if __name__ == "__main__":
    main()
