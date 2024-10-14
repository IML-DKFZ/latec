import gc
from copy import deepcopy
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

from src.modules.eval_metrics import EvalMetricsModule
from src.modules.models import ModelsModule
from src.modules.xai_methods import XAIMethodsModule
from src import utils

log = utils.get_pylogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@utils.task_wrapper
def eval(cfg: DictConfig) -> Tuple[dict, dict]:
    """Main evaluation loop."""
    _set_random_seed(cfg)
    explain_data = _load_saliency_maps(cfg)
    datamodule, dataloader = _instantiate_datamodule(cfg)
    x_batch, y_batch = _get_data_batch(dataloader, explain_data)

    models = ModelsModule(cfg)
    eval_data = _evaluate_models(
        cfg, models, explain_data, x_batch, y_batch, datamodule
    )

    _save_evaluation_scores(cfg, datamodule, eval_data)


def _set_random_seed(cfg: DictConfig) -> None:
    """Set seed for random number generators in PyTorch, NumPy, and Python."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)


def _load_saliency_maps(cfg: DictConfig) -> List[np.ndarray]:
    """Load the saliency maps from the specified path."""
    log.info(
        f"Loading saliency maps <{cfg.attr_path}> for modality <{cfg.data.modality}>"
    )
    explain_data = np.load(
        os.path.join(
            cfg.paths.data_dir, "saliency_maps", cfg.data.modality, cfg.attr_path
        )
    )
    return [explain_data["arr_0"], explain_data["arr_1"], explain_data["arr_2"]]


def _instantiate_datamodule(
    cfg: DictConfig,
) -> Tuple[LightningDataModule, torch.utils.data.DataLoader]:
    """Instantiate the LightningDataModule and return the dataloader."""
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    dataloader = datamodule.dataloader()
    return datamodule, dataloader


def _get_data_batch(
    dataloader, explain_data: List[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve a batch of data and adjust it based on the number of observations."""
    with torch.no_grad():
        x_batch, y_batch = next(iter(dataloader))

    x_batch = x_batch[: explain_data[0].shape[0], :]
    y_batch = y_batch[: explain_data[0].shape[0]]
    return x_batch, y_batch


def _evaluate_models(
    cfg: DictConfig,
    models: ModelsModule,
    explain_data: List[np.ndarray],
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    datamodule: LightningDataModule,
) -> List[np.ndarray]:
    """Perform the evaluation for each model using the specified explainability methods."""
    eval_data = []
    log.info(f"Starting evaluation over each model")

    for idx_model, model in tqdm(
        enumerate(models.models),
        total=len(models.models),
        desc=f"Eval for {datamodule.__class__.__name__}",
        colour="BLUE",
    ):
        eval_data_model = _evaluate_single_model(
            cfg, model, idx_model, explain_data, x_batch, y_batch
        )
        eval_data.append(np.array(eval_data_model))
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return eval_data


def _evaluate_single_model(
    cfg: DictConfig,
    model: torch.nn.Module,
    idx_model: int,
    explain_data: List[np.ndarray],
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> List[np.ndarray]:
    """Evaluate a single model on the given explainability methods and data."""
    eval_data_model = []

    for idx_xai in tqdm(
        range(explain_data[idx_model].shape[1]),
        desc=f"{model.__class__.__name__}",
        colour="CYAN",
    ):
        results = _evaluate_chunks(
            cfg, model, idx_model, idx_xai, explain_data, x_batch, y_batch
        )
        eval_data_model.append(np.hstack(results))

    return eval_data_model


def _evaluate_chunks(
    cfg: DictConfig,
    model: torch.nn.Module,
    idx_model: int,
    idx_xai: int,
    explain_data: List[np.ndarray],
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
) -> List[np.ndarray]:
    """Evaluate the model in chunks to handle large data efficiently."""
    results = []
    model = model.to(device)
    x_batch = _prepare_tensor(x_batch, cfg)

    for idx_chunk in tqdm(
        range(0, x_batch.shape[0], cfg.chunk_size),
        desc=f"Chunkwise (n={cfg.chunk_size}) Computation",
        colour="GREEN",
    ):
        a_batch = _get_saliency_chunk(explain_data, idx_model, idx_xai, idx_chunk, cfg)
        xai_methods = XAIMethodsModule(cfg, model, x_batch)
        eval_methods = EvalMetricsModule(cfg, model)

        if cfg.data.modality == "volume":
            x_batch = x_batch.squeeze()
            a_batch = a_batch.squeeze()

        scores = eval_methods.evaluate(
            model,
            x_batch.cpu().numpy()[idx_chunk : idx_chunk + cfg.chunk_size],
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

        del xai_methods, eval_methods, scores
        torch.cuda.empty_cache()
        gc.collect()

    return results


def _prepare_tensor(x_batch: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Prepare the tensor by moving it to the appropriate device and adjusting for modality."""
    if not torch.is_tensor(x_batch):
        x_batch = torch.from_numpy(x_batch).to(device)
        if cfg.data.modality == "volume":
            x_batch = x_batch.unsqueeze(1)
    else:
        x_batch = x_batch.to(device)
    return x_batch


def _get_saliency_chunk(
    explain_data: List[np.ndarray],
    idx_model: int,
    idx_xai: int,
    idx_chunk: int,
    cfg: DictConfig,
) -> np.ndarray:
    """Retrieve a chunk of saliency maps and adjust for numerical stability if necessary."""
    a_batch = explain_data[idx_model][:, idx_xai, :]
    if np.all(a_batch[idx_chunk : idx_chunk + cfg.chunk_size] == 0):
        a_batch[idx_chunk : idx_chunk + cfg.chunk_size][:, 0, 0] = 1e-10
        log.info(
            f"Saliency all zero in chunk: {idx_chunk} to {idx_chunk + cfg.chunk_size}"
        )
    return a_batch


def _save_evaluation_scores(
    cfg: DictConfig, datamodule: LightningDataModule, eval_data: List[np.ndarray]
) -> None:
    """Save the evaluation scores to disk."""
    file_name = os.path.join(
        cfg.paths.data_dir,
        "evaluation_scores",
        cfg.data.modality,
        f"eval_{datamodule.__class__.__name__}_dataset{cfg.time}.npz",
    )
    np.savez(file_name, *eval_data)
    log.info(f"Evaluation scores saved to {file_name}")


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="eval.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    eval(cfg)


if __name__ == "__main__":
    main()
