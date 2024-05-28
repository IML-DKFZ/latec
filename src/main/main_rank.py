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

def NormalizeData(data, min, max):
    return (data - min) / ((max - min) + 0.00000000001)

def rank(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load all evaluation scores
    file_loc = "./data/evaluation_scores"

    file = np.load(file_loc + cfg.file_image_inet, allow_pickle=True)
    arr_image_inet = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_image_oct, allow_pickle=True)
    arr_image_oct = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_image_r45, allow_pickle=True)
    arr_image_r45 = [file["arr_0"], file["arr_1"], file["arr_2"]]

    file = np.load(file_loc + cfg.file_volume_adr, allow_pickle=True)
    arr_volume_adr = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_volume_org, allow_pickle=True)
    arr_volume_org = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_volume_ves, allow_pickle=True)
    arr_volume_ves = [file["arr_0"], file["arr_1"], file["arr_2"]]

    file = np.load(file_loc + cfg.file_pc_coma, allow_pickle=True)
    arr_pc_coma = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_pc_m40, allow_pickle=True)
    arr_pc_m40 = [file["arr_0"], file["arr_1"], file["arr_2"]]
    file = np.load(file_loc + cfg.file_pc_shpn, allow_pickle=True)
    arr_pc_shpn = [file["arr_0"], file["arr_1"], file["arr_2"]]

    arr_image = [arr_image_inet, arr_image_oct, arr_image_r45]
    arr_volume = [arr_volume_adr, arr_volume_org, arr_volume_ves]
    arr_pc = [arr_pc_coma, arr_pc_m40, arr_pc_shpn]
    arr_modalities = [arr_image, arr_volume, arr_pc]

    # Compute either full ranking or ranking across models
    if cfg.full_ranking == True:
        arr_ranking = np.empty(
            [3, 3, 3, 17, 20], dtype=float
        )  # modality, dataset, model, xai, eval
        arr_ranking[:] = np.nan

        bup_order = [0, 1, 2, 4, 5, 7, 9, 12, 17]

        for modality in range(3):
            for dataset in range(3):
                for model in range(3):
                    for xai in range(arr_modalities[modality][dataset][model].shape[0]):
                        for eval in range(20):
                            ranking = np.median(
                                arr_modalities[modality][dataset][model][:, eval, :], -1
                            ).argsort()  # compute ranking based on median obs score
                            if eval in bup_order:
                                ranking = ranking[
                                    ::-1
                                ]  # reverse ranking to bottom up if larger is better

                            pos = (
                                ranking.argsort()[xai] + 1
                            )  # get rankin position of xai method (+1 so ranking starts at 1 and not 0)
                            arr_ranking[modality, dataset, model, xai, eval] = pos

    else:
        arr_ranking = np.empty(
            [3, 3, 17, 20], dtype=float
        )  # modality, dataset, xai, eval
        arr_ranking[:] = np.nan

        bup_order = [0, 1, 2, 4, 5, 7, 9, 12, 17]

        for modality in range(3):
            for dataset in range(3):
                for eval in range(20):
                    arr_models = []
                    for i in range(3):
                        d = arr_modalities[modality][dataset][i][:, eval, :]
                        q_h = np.quantile(d, 0.975)
                        q_l = np.quantile(d, 0.025)

                        d = np.clip(d, q_l, q_h)
                        d_max = d.max()
                        d_min = d.min()
                        arr_models.append(NormalizeData(d, d_min, d_max))

                    ranking = np.concatenate(
                        [
                            np.median(
                                np.hstack(
                                    [arr_models[0], arr_models[1], arr_models[2][:-3]]
                                ),
                                -1,
                            ),
                            np.median(arr_models[2][-3:], -1),
                        ]
                    ).argsort()
                    # compute ranking based on median obs score
                    if eval in bup_order:
                        ranking = ranking[
                            ::-1
                        ]  # reverse ranking to bottom up if larger is better

                    for xai in range(ranking.shape[0]):
                        pos = (
                            ranking.argsort()[xai] + 1
                        )  # get rankin position of xai method (+1 so ranking starts at 1 and not 0)
                        arr_ranking[modality, dataset, xai, eval] = pos

    # Compute table either across models or for model "idx_model"
    arr_table = []
    for eval in [(0, 9), (9, 16), (16, 19)]:
        for modality in range(3):
            for dataset in range(3):
                arr_col_val = []
                arr_col_std = []
                for xai in range(17):
                    if modality == 2 and xai == 6:
                        arr_col_val = arr_col_val + ["-", "-", "-"]
                        arr_col_std = arr_col_std + [" ", " ", " "]
                    if modality == 2 and xai == 14:
                        break
                    x = (
                        arr_ranking[
                            modality, dataset, cfg.idx_model, xai, eval[0] : eval[1]
                        ]
                        if cfg.full_ranking == True
                else arr_ranking[modality, dataset, xai, eval[0] : eval[1]]
                    )
                    val = np.round(np.mean(x[~np.isnan(x)]))
                    std = np.round(sem(x[~np.isnan(x)]), 2)
                    if not np.isnan(val):
                        val = int(val)
                    else:
                        val = "-"
                        std = "-"
                    arr_col_val.append(val)
                    #arr_col_std.append("±" + str(std))
                arr_table.append(arr_col_val)
                #arr_table.append(arr_col_std)

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
    df_table.to_csv(
        "./data/figures/table"
        + ("_across" if cfg.full_ranking == False else "_model_" + str(cfg.idx_model))
        + ".csv",
        encoding="utf-8",
        index=True,
        header=False,
    )

    print("Done!")


@hydra.main(version_base="1.3", config_path= os.getcwd() + "/configs", config_name="rank.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    rank(cfg)


if __name__ == "__main__":
    main()
