import os
from typing import List, Optional, Tuple
import hydra
import numpy as np
import pandas as pd
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from scipy.stats import sem
from tqdm.auto import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def normalize_data(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min + 1e-10)


def load_evaluation_scores(
    file_loc: str, filenames: List[str]
) -> List[List[np.ndarray]]:
    """Load evaluation scores for the given list of filenames."""
    return [
        [
            np.load(os.path.join(file_loc, filename), allow_pickle=True)[f"arr_{i}"]
            for i in range(3)
        ]
        for filename in filenames
    ]


def prepare_modalities(cfg: DictConfig) -> List[List[List[np.ndarray]]]:
    """Prepare data arrays for image, volume, and point cloud modalities."""
    file_loc = "./data/evaluation_scores"

    arr_image_files = [cfg.file_image_inet, cfg.file_image_oct, cfg.file_image_r45]
    arr_volume_files = [cfg.file_volume_adr, cfg.file_volume_org, cfg.file_volume_ves]
    arr_pc_files = [cfg.file_pc_coma, cfg.file_pc_m40, cfg.file_pc_shpn]

    arr_image = load_evaluation_scores(file_loc, arr_image_files)
    arr_volume = load_evaluation_scores(file_loc, arr_volume_files)
    arr_pc = load_evaluation_scores(file_loc, arr_pc_files)

    return [arr_image, arr_volume, arr_pc]


def compute_full_ranking(
    arr_modalities: List[List[List[np.ndarray]]], bup_order: List[int]
) -> np.ndarray:
    """Compute the full ranking across models, datasets, and modalities."""
    arr_ranking = np.full(
        [3, 3, 3, 17, 20], np.nan
    )  # modality, dataset, model, xai, eval

    for modality in range(3):
        for dataset in range(3):
            for model in range(3):
                for xai in range(arr_modalities[modality][dataset][model].shape[0]):
                    for eval_idx in range(20):
                        ranking = np.median(
                            arr_modalities[modality][dataset][model][:, eval_idx, :],
                            axis=-1,
                        ).argsort()
                        if eval_idx in bup_order:
                            ranking = ranking[
                                ::-1
                            ]  # reverse ranking if larger is better
                        pos = ranking.argsort()[xai] + 1  # start ranking from 1
                        arr_ranking[modality, dataset, model, xai, eval_idx] = pos

    return arr_ranking


def compute_model_ranking(
    arr_modalities: List[List[List[np.ndarray]]], bup_order: List[int]
) -> np.ndarray:
    """Compute ranking across models for each modality and dataset."""
    arr_ranking = np.full([3, 3, 17, 20], np.nan)  # modality, dataset, xai, eval

    for modality in range(3):
        for dataset in range(3):
            for eval_idx in range(20):
                arr_models = []
                for model in range(3):
                    data = arr_modalities[modality][dataset][model][:, eval_idx, :]
                    q_h, q_l = np.quantile(data, [0.975, 0.025])
                    data = np.clip(data, q_l, q_h)
                    data_max, data_min = data.max(), data.min()
                    arr_models.append(normalize_data(data, data_min, data_max))

                combined_data = np.concatenate(
                    [
                        np.median(
                            np.hstack(
                                [arr_models[0], arr_models[1], arr_models[2][:-3]]
                            ),
                            -1,
                        ),
                        np.median(arr_models[2][-3:], -1),
                    ]
                )
                ranking = combined_data.argsort()
                if eval_idx in bup_order:
                    ranking = ranking[::-1]

                for xai in range(ranking.shape[0]):
                    pos = ranking.argsort()[xai] + 1  # start ranking from 1
                    arr_ranking[modality, dataset, xai, eval_idx] = pos

    return arr_ranking


def create_ranking_table(arr_ranking: np.ndarray, cfg: DictConfig) -> pd.DataFrame:
    """Create a ranking table from the computed rankings."""
    arr_table = []
    for eval_range in [(0, 9), (9, 16), (16, 19)]:
        for modality in range(3):
            for dataset in range(3):
                arr_col_val = []
                for xai in range(17):
                    if modality == 2 and xai == 6:
                        arr_col_val = arr_col_val + ["-", "-", "-"]
                    if modality == 2 and xai == 14:
                        break
                    x = (
                        arr_ranking[
                            modality,
                            dataset,
                            cfg.idx_model,
                            xai,
                            eval_range[0] : eval_range[1],
                        ]
                        if cfg.full_ranking
                        else arr_ranking[
                            modality, dataset, xai, eval_range[0] : eval_range[1]
                        ]
                    )
                    val = (
                        int(np.round(np.mean(x[~np.isnan(x)])))
                        if not np.isnan(np.mean(x))
                        else "-"
                    )
                    arr_col_val.append(val)
                arr_table.append(arr_col_val)

    df_table = pd.DataFrame(arr_table).transpose()
    df_table.index = [
        "OC",
        "LI",
        "KS",
        "VG",
        "IxG",
        "GB",
        "GC",
        "SC",
        "C+",
        "IG",
        "EG",
        "DL",
        "DLS",
        "LRP",
        "RA",
        "RoA",
        "LA",
    ]
    return df_table


def save_ranking_table(df_table: pd.DataFrame, cfg: DictConfig) -> None:
    """Save the ranking table to a CSV file."""
    filename = f"./data/figures/table_{'across' if not cfg.full_ranking else 'model_' + str(cfg.idx_model)}.csv"
    df_table.to_csv(filename, encoding="utf-8", index=True, header=False)


def rank(cfg: DictConfig) -> None:
    """Main ranking function."""
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    arr_modalities = prepare_modalities(cfg)
    bup_order = [0, 1, 2, 4, 5, 7, 9, 12, 17]

    arr_ranking = (
        compute_full_ranking(arr_modalities, bup_order)
        if cfg.full_ranking
        else compute_model_ranking(arr_modalities, bup_order)
    )
    df_table = create_ranking_table(arr_ranking, cfg)
    save_ranking_table(df_table, cfg)

    print("Ranking process completed!")


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="rank.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    rank(cfg)


if __name__ == "__main__":
    main()
