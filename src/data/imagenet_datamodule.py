from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
import numpy as np
import random


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/datasets/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        resize=256,
        resize_mode="bilinear",
        crop: int = 224,
        modality: str = "Image",
        weights_cnext="IMAGENET1K_V1",
        weights_effnet="IMAGENET1K_V1",
        weights_vit="IMAGENET1K_V1",
    ):
        super().__init__()
        self.__name__ = "imagenet"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.hparams.resize,
                    interpolation=transforms.InterpolationMode.BILINEAR
                    if self.hparams.resize_mode == "bilinear"
                    else transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.hparams.crop),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.data = ImageNet(root=data_dir, split="val", transform=self.transforms)

        self.g = torch.Generator()
        self.g.manual_seed(0)

    @property
    def num_classes(self):
        return 1000


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
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
        )


if __name__ == "__main__":
    _ = ImageNetDataModule()
