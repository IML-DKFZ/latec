import torch
import numpy as np
import random

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.transforms import NormalizeScale
from torch_geometric.datasets import ShapeNet


def collate(list_of_examples):
    data_list = [
        x.pos[
            torch.randint(
                x.pos.shape[0], (1024,), generator=torch.Generator().manual_seed(1)
            ),
            :,
        ]
        for x in list_of_examples
    ]
    tensors = [x.category for x in list_of_examples]
    return (
        torch.stack(data_list, dim=0).transpose(1, 2),
        torch.stack(tensors, dim=0).squeeze(),
    )


class ShapeNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/datasets/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        categories: str = None,
        num_classes=16,
        modality: str = "point_cloud",
        weights_pointnet="data/model_weights/ShapeNet/PointNet-epoch=199-val_F1=0.9553-val_Accuracy=0.9920.ckpt",  # PointNet2-epoch=99-val_F1=0.9665-val_Accuracy=0.9941.ckpt
        weights_dgcnn="data/model_weights/ShapeNet/DGCNN-epoch=199-val_F1=0.9605-val_Accuracy=0.9941.ckpt",
        weights_pct="data/model_weights/ShapeNet/PCT-epoch=199-val_F1=0.9730-val_Accuracy=0.9963.ckpt",
    ):
        super().__init__()
        self.__name__ = "shapenet"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        pretransform = NormalizeScale()
        self.transforms = None

        self.data = ShapeNet(
            root=data_dir + "/ShapeNet/",
            split="test",
            pre_transform=pretransform,
            transform=self.transforms,
            # categories=categories
        )

        self.g = torch.Generator()
        self.g.manual_seed(0)

    @property
    def num_classes(self):
        return 16

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
    _ = ShapeNetDataModule()
