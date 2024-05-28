import torch
import os
import numpy as np
import random

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.utils.download_url import *


class Transform3D:
    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, volume):
        if self.mul == "0.5":
            volume = volume * 0.5
        elif self.mul == "random":
            volume = volume * np.random.uniform()

        return volume.astype(np.float32)


class MedMNIST(Dataset):
    "From: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/dataset.py"

    flag = ...

    def __init__(
        self, split, transform=None, target_transform=None, as_rgb=False, root=None
    ):
        """dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
        """

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == "train":
            self.imgs = npz_file["train_images"]
            self.labels = npz_file["train_labels"]
        elif self.split == "val":
            self.imgs = npz_file["val_images"]
            self.labels = npz_file["val_labels"]
        elif self.split == "test":
            self.imgs = npz_file["test_images"]
            self.labels = npz_file["test_labels"]
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision.ss"""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MedMNIST3D(MedMNIST):
    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img / 255.0] * (3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.Tensor(target)[0].type(torch.LongTensor)

        return img, target


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class OrganMNSIT3DDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/datasets/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_classes=11,
        modality: str = "volume",
        weights_3dresnet=None,
        weights_3deffnet=None,
        weights_s3dformer=None,
    ):
        super().__init__()
        self.__name__ = "OrganMNIST3D"

        if not os.path.exists(data_dir + "/OrganMNIST3D"):
            print("Downloading OrganMNIST3D data...")

            data_url = (
                "https://zenodo.org/record/6496656/files/organsmnist.npz?download=1"
            )
            save_path = data_dir + "/OrganMNIST3D/organmnist3d.npz"

            os.makedirs(data_dir + "/OrganMNIST3D/", exist_ok=True)

            download_url(data_url, save_path)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = Transform3D()

        self.data = OrganMNIST3D(
            root=data_dir + "/OrganMNIST3D/",
            split="test",
            transform=self.transforms,
        )

        self.g = torch.Generator()
        self.g.manual_seed(0)

    @property
    def num_classes(self):
        return 11

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
        )


if __name__ == "__main__":
    _ = OrganMNSIT3DDataModule()
