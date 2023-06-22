from typing import List, Optional, Tuple

import hydra
import torch
import pyrootutils
import numpy as np
import pytorch_lightning as pl
import gc
from tqdm.auto import tqdm
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from modules.models import ModelsModule
from modules.eval_methods import EvalModule
from modules.xai_methods import XAIMethodsModule
from copy import deepcopy

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
    attr_data = np.load(
        str(cfg.paths.data_dir)
        + "/attribution_maps/"
        + cfg.data.modality
        + cfg.attr_path
    )
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

    assert attr_data[0].shape[0] >= cfg.chunk_size, "chuncksize larger than n obs"

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

        for count_xai in tqdm(
            range(attr_data[count_model].shape[1]),
            total=attr_data[count_model].shape[1],
            desc=f"{model.__class__.__name__}",
            colour="CYAN",
            position=1,
            leave=True,
        ):
            results = []
            for i in tqdm(
                range(0, x_batch.shape[0], cfg.chunk_size),
                desc=f"Chunkwise (n={cfg.chunk_size}) Computation",
                colour="GREEN",
                position=2,
                leave=True,
            ):
                model = model.to(cfg.eval_method.device)

                if torch.is_tensor(x_batch) == False:
                    x_batch = torch.from_numpy(x_batch).to(cfg.eval_method.device)
                    if cfg.data.modality == "Voxel":
                        x_batch = x_batch.unsqueeze(1)
                else:
                    x_batch = x_batch.to(cfg.eval_method.device)

                xai_methods = XAIMethodsModule(cfg, model, x_batch)

                a_batch = attr_data[count_model][:, count_xai, :]

                if np.all((a_batch[i : i + cfg.chunk_size] == 0)):
                    for j in range(cfg.chunk_size):
                        a_batch[i : i + cfg.chunk_size][j,0,0,0] = 0.0000000001

                if cfg.data.modality == "Image" or cfg.data.modality == "Point_Cloud":
                    x_batch = x_batch.cpu().numpy()
                elif cfg.data.modality == "Voxel":
                    x_batch = x_batch.squeeze().cpu().numpy()
                    a_batch = a_batch.squeeze()

                eval_methods = EvalModule(cfg, model)

                scores = eval_methods.evaluate(
                    model,
                    x_batch[i : i + cfg.chunk_size],
                    y_batch.cpu().numpy()[i : i + cfg.chunk_size],
                    a_batch[i : i + cfg.chunk_size],
                    xai_methods,
                    count_xai,
                )
                results.append(deepcopy(scores))

                del xai_methods
                del eval_methods
                del scores

                gc.collect()

            eval_scores_model.append(np.hstack(results))

        del model

        gc.collect()

        eval_scores_total.append(np.array(eval_scores_model)) # xai, eval, obs

    eval_scores_total = np.array(eval_scores_total) # model, xai, eval, obs

    np.savez(
        str(cfg.paths.data_dir)
        + "/evaluation/"
        + cfg.data.modality
        + "/attr_"
        + str(datamodule.__name__)
        + "_dataset_"
        + str(eval_scores_total.shape[2])
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
