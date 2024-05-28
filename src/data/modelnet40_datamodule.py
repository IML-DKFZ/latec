import torch
import numpy as np
import random

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import NormalizeScale
from torch_geometric.datasets import ModelNet

from src.utils.point_sampling import *


def collate(list_of_examples):
    data_list = [x.pos for x in list_of_examples]
    tensors = [x.y for x in list_of_examples]
    return (
        torch.stack(data_list, dim=0).transpose(1, 2),
        torch.stack(tensors, dim=0).squeeze(),
    )


class ModelNet40DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/datasets/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes=45,
        resize=224,
        resize_mode="bilinear",
        modality: str = "point_cloud",
        weights_pointnet="data/model_weights/ModelNet40/PointNet-epoch=199.ckpt",  # PointNet2-epoch=199.ckpt
        weights_dgcnn="data/model_weights/ModelNet40/DGCNN-epoch=249.ckpt",
        weights_pct="data/model_weights/ModelNet40/PCT-epoch=249.ckpt",
    ):
        super().__init__()
        self.__name__ = "modelnet40"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        pretransform = NormalizeScale()
        self.transforms = SamplePoints(1024)

        self.data = ModelNet(
            root=data_dir + "/ModelNet40/",
            name="40",
            train=False,
            pre_transform=pretransform,
            transform=self.transforms,
        )

        self.g = torch.Generator()
        self.g.manual_seed(5)

    @property
    def num_classes(self):
        return 40

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def dataloader(self):
        return DataLoader(
            dataset=self.data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            collate_fn=collate,
        )


if __name__ == "__main__":
    _ = ModelNet40DataModule()
