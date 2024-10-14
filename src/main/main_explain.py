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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@utils.task_wrapper
def explain(cfg: DictConfig) -> Tuple[dict, dict]:
    """Main explanation loop."""
    # Set seed for random number generators
    _set_random_seed(cfg)

    # Load data and extract a batch
    datamodule, x_batch, y_batch = _load_data(cfg)

    # Load pretrained models
    models = ModelsModule(cfg)
    explain_data = []

    # Loop over models to compute saliency maps
    log.info(f"Starting saliency map computation for each Model and XAI Method")
    for model in tqdm(
        models.models, desc=f"Attribution for {cfg.data.modality} Models", colour="BLUE"
    ):
        explain_data_model = _compute_explain_data_for_model(
            cfg, model, x_batch, y_batch
        )
        explain_data.append(np.vstack(explain_data_model))

    # Save the saliency maps
    _save_saliency_maps(cfg, datamodule, explain_data)


def _set_random_seed(cfg: DictConfig) -> None:
    """Set seed for random number generators in PyTorch, NumPy, and Python's random module."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)


def _load_data(
    cfg: DictConfig,
) -> Tuple[LightningDataModule, torch.Tensor, torch.Tensor]:
    """Instantiate the datamodule and return the first batch of data."""
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()

    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    return datamodule, x_batch, y_batch


def _compute_explain_data_for_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> List[np.ndarray]:
    """Compute the explain data for a given model using the configured XAI methods."""
    model = model.to(device)
    xai_methods = XAIMethodsModule(cfg, model, x_batch.to(device))
    explain_data_model = []

    for idx_chunk in tqdm(
        range(0, x_batch.size(0), cfg.chunk_size),
        desc=f"Chunkwise (n={cfg.chunk_size}) Computation",
        colour="CYAN",
    ):
        x_chunk = x_batch[idx_chunk : idx_chunk + cfg.chunk_size].to(device)
        y_chunk = y_batch[idx_chunk : idx_chunk + cfg.chunk_size].to(device)
        explain_data_model.append(xai_methods.attribute(x_chunk, y_chunk))

    return explain_data_model


def _save_saliency_maps(
    cfg: DictConfig, datamodule: LightningDataModule, explain_data: List[np.ndarray]
) -> None:
    """Save the saliency maps to disk."""
    file_name = (
        f"{cfg.paths.data_dir}/saliency_maps/{cfg.data.modality}/explain_"
        f"{datamodule.__class__.__name__}_{explain_data[0].shape[1]}_methods_{cfg.time}.npz"
    )

    np.savez(file_name, *explain_data)
    log.info(f"Saliency maps saved to {file_name}")


@hydra.main(
    version_base="1.3", config_path=os.getcwd() + "/configs", config_name="explain.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    explain(cfg)


if __name__ == "__main__":
    main()
