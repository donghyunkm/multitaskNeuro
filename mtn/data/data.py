import os
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from mtn.utils import get_paths


class FmriDataModule(L.LightningDataModule):
    """
    Data module using PyG functions to return graph patches.
    """

    def __init__(
        self,
        batch_size: int = 128,
        quantile: int = 31,
        rand_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.quantile = quantile
        self.rand_seed = rand_seed

    def prepare_data(self):
        data_dir = get_paths()["data_root"]
        features = np.load(data_dir + "triangle.npy")
        labels = np.load(data_dir + "age_labels_norm.npy")

        x = torch.from_numpy(features)
        x = x.to(torch.float32)
        self.x = x
        if self.quantile < 31:
            y = torch.tensor(pd.qcut(labels, q=self.quantile, labels=False))
        else:
            y = torch.tensor(labels)
        self.y = y

    def get_indices(self):
        # Get indices for train/val/test splits, stratified by y
        indices = np.arange(len(self.y))

        # First split into train+val vs test (80/20)
        train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=self.y, random_state=self.rand_seed)

        # Split train+val into train vs val (80/20 of remaining data)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, stratify=self.y[train_val_idx], random_state=self.rand_seed
        )

        return train_idx, val_idx, test_idx

    def setup(self, stage: str):
        self.prepare_data()
        self.train_idx, self.val_idx, self.test_idx = self.get_indices()

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.train_idx], self.y[self.train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.val_idx], self.y[self.val_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.test_idx], self.y[self.test_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.test_idx], self.y[self.test_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


def test_dataloader():
    import numpy as np

    datamodule = FmriDataModule(batch_size=4, quantile=-1, rand_seed=42)

    datamodule.setup(stage="fit")
    dataloader = iter(datamodule.train_dataloader())

    for i in range(3):
        batch = next(dataloader)

        print(batch)
        print(batch[0].shape, batch[1].shape)

    print("checks passed")

    return


if __name__ == "__main__":
    print("running dataloader")
    test_dataloader()
