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

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load selected saliency maps from ./data/explanation_maps/*modality*
    log.info(
        f"Loading saliency maps <{cfg.attr_path}> for modality <{cfg.data.modality}>"
    )
    explain_data = np.load(
        str(cfg.paths.data_dir)
        + "/explanation_maps/"
        + cfg.data.modality
        + "/"
        + cfg.attr_path
    )
    explain_data = [
        explain_data["arr_0"],
        explain_data["arr_1"],
        explain_data["arr_2"],
    ]  # obs, xaimethods, c , w, h

    # Load dataloader for selected dataset
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()

    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    x_batch = x_batch[0 : explain_data[0].shape[0], :]
    y_batch = y_batch[0 : explain_data[0].shape[0]]

    # Assert that number selected observations is larger than chunk size
    assert explain_data[0].shape[0] >= cfg.chunk_size, "chuncksize larger than n obs"

    # Load pretrained models for dataset
    log.info(f"Instantiating models for <{cfg.data.modality}> data")
    models = ModelsModule(cfg)

    eval_data = []

    # Evaluation loop over model, XAI method and chunk of obseravtions
    log.info(f"Starting Evaluation over each Model")
    for idx_model, model in tqdm(
        enumerate(models.models),
        total=3,
        desc=f"Eval for {datamodule.__name__}",
        colour="BLUE",
        position=0,
        leave=True,
    ):
        eval_data_model = []

        for idx_xai in tqdm(
            range(explain_data[idx_model].shape[1]),
            total=explain_data[idx_model].shape[1],
            desc=f"{model.__class__.__name__}",
            colour="CYAN",
            position=1,
            leave=True,
        ):
            results = []
            for idx_chunk in tqdm(
                range(0, x_batch.shape[0], cfg.chunk_size),
                desc=f"Chunkwise (n={cfg.chunk_size}) Computation",
                colour="GREEN",
                position=2,
                leave=True,
            ):
                model = model.to(cfg.eval_method.device)

                # Check if observations are tensor
                if torch.is_tensor(x_batch) == False:
                    x_batch = torch.from_numpy(x_batch).to(cfg.eval_method.device)
                    if cfg.data.modality == "Volume":
                        x_batch = x_batch.unsqueeze(1)
                else:
                    x_batch = x_batch.to(cfg.eval_method.device)

                # Load XAI methods module
                xai_methods = XAIMethodsModule(cfg, model, x_batch)

                a_batch = explain_data[idx_model][
                    :, idx_xai, :
                ]  # Select saliency maps for observations

                if np.all((a_batch[idx_chunk : idx_chunk + cfg.chunk_size] == 0)):
                    a_batch[idx_chunk : idx_chunk + cfg.chunk_size][
                        :, 0, 0
                    ] = 0.0000000001  # for numerical stability if all zero
                    log.info(
                        "Saliency all zero in chunk: "
                        + str(idx_chunk)
                        + " to "
                        + str(idx_chunk + cfg.chunk_size)
                    )

                if cfg.data.modality == "Image" or cfg.data.modality == "Point_Cloud":
                    x_batch = x_batch.cpu().numpy()
                elif cfg.data.modality == "Volume":
                    x_batch = x_batch.squeeze().cpu().numpy()
                    a_batch = a_batch.squeeze()

                # Load evaluation module
                eval_methods = EvalModule(cfg, model)

                # Evaluate chunk
                scores = eval_methods.evaluate(
                    model,
                    x_batch[idx_chunk : idx_chunk + cfg.chunk_size],
                    y_batch.cpu().numpy()[idx_chunk : idx_chunk + cfg.chunk_size],
                    a_batch[idx_chunk : idx_chunk + cfg.chunk_size],
                    xai_methods,
                    idx_xai,
                    custom_batch=[
                        x_batch,
                        y_batch,
                        a_batch,
                        list(range(idx_chunk, idx_chunk + cfg.chunk_size)),
                    ],
                )
                results.append(deepcopy(scores))

                # tidy up RAM and VRAM
                del xai_methods
                del eval_methods
                del scores

                torch.cuda.empty_cache()
                gc.collect()

            eval_data_model.append(np.hstack(results))

        del model

        torch.cuda.empty_cache()
        gc.collect()

        eval_data.append(np.array(eval_data_model))

    np.savez(
        str(cfg.paths.data_dir)
        + "/evaluation/"
        + cfg.data.modality
        + "/eval_"
        + str(datamodule.__name__)
        + "_dataset"
        + ".npz",
        eval_data[0],
        eval_data[1],
        eval_data[2],
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    eval(cfg)


if __name__ == "__main__":
    main()
