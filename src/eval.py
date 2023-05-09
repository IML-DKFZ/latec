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
from modules.eval_methods import EvalModule
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
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(
        f"Loading Attributions <{cfg.attr_path}> for modality <{cfg.data.modality}>"
    )
    attr_data = np.load("data/attribution_maps/" + cfg.data.modality + cfg.attr_path)
    attr_data = [
        attr_data["arr_0"],
        attr_data["arr_1"],
        attr_data["arr_2"],
    ]  # obs, xaimethods, c , w, h

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()

    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    x_batch = x_batch[0 : attr_data[0].shape[0], :]
    y_batch = y_batch[0 : attr_data[0].shape[0]]

    log.info(f"Instantiating models for <{cfg.data.modality}> data")
    models = ModelsModule(cfg)

    eval_scores_total = []

    log.info(f"Starting Evaluation over each Model")
    for count_model, model in tqdm(
        enumerate(models.models),
        total=3,
        desc=f"Eval for {datamodule.__name__}",
        colour="BLUE",
        position=0,
        leave=True,
    ):
        eval_scores_model = []

        model = model.to(cfg.eval_method.device)

        for count_xai in tqdm(
            range(attr_data[count_model].shape[1]),
            total=attr_data[count_model].shape[1],
            desc=f"{model.__class__.__name__}",
            colour="CYAN",
            position=1,
            leave=True,
        ):
            eval_methods = EvalModule(cfg, model)

            xai_methods = XAIMethodsModule(
                cfg, model, x_batch.to(cfg.eval_method.device)
            )

            a_batch = attr_data[count_model][:, count_xai, :]

            results = eval_methods.evaluate(
                model,
                x_batch.cpu().numpy()
                if cfg.data.modality == "Image"
                else x_batch.squeeze().cpu().numpy(),
                y_batch.cpu().numpy(),
                a_batch if cfg.data.modality == "Image" else a_batch.squeeze(),
                xai_methods,
                count_xai,
            )

            eval_scores_model.append(results)

        eval_scores_total.append(np.array(eval_scores_model))

    eval_scores_total = np.array(eval_scores_total)

    np.savez(
        str(cfg.paths.root_dir)
        + "/data/evaluation/"
        + cfg.data.modality
        + "/attr_"
        + str(datamodule.__name__)
        + "_dataset_"
        + str(eval_scores_total.shape[0])
        + "_methods"
        + ".npz",
        eval_scores_total,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # eval the xai
    eval(cfg)


if __name__ == "__main__":
    main()
