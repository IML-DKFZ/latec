from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import numpy as np
import random
from torch_geometric.transforms import NormalizeScale
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.transforms import FixedPoints


class SamplePoints(BaseTransform):
    r"""Uniformly samples a fixed number of points on the mesh faces according
    to their face area (functional name: :obj:`sample_points`).

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """

    def __init__(
        self,
        num: int,
        remove_faces: bool = True,
        include_normals: bool = False,
    ):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def __call__(self, data: Data) -> Data:
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        face[0] = torch.where(face[0] < 112589990689, face[0], 1)
        face[1] = torch.where(face[1] < 112589990689, face[1], 1)
        face[2] = torch.where(face[2] < 112589990689, face[2], 1)

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = torch.nn.functional.normalize(vec1.cross(vec2), p=2)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num})"


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
        modality: str = "Point_Cloud",
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
