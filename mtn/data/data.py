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
        y = np.load(data_dir + "y_age.npy")
        bc = np.load(data_dir + "y_bc.npy")
        y_aux = np.load(data_dir + "labels_aux.npy")

        x = torch.from_numpy(features)
        x = x.to(torch.float32)
        self.x = x

        self.y_aux = torch.from_numpy(y_aux)
        self.labels = torch.tensor(labels)
        self.y = torch.from_numpy(y)
        self.bc = torch.from_numpy(bc)

    def get_indices(self):
        # Get indices for train/val/test splits, stratified by y
        indices = np.arange(len(self.labels))

        # First split into train+val vs test (80/20)
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=self.labels, random_state=self.rand_seed
        )

        # Split train+val into train vs val (80/20 of remaining data)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.2, stratify=self.labels[train_val_idx], random_state=self.rand_seed
        )

        return train_idx, val_idx, test_idx

    def setup(self, stage: str):
        self.prepare_data()
        self.train_idx, self.val_idx, self.test_idx = self.get_indices()

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(
                self.x[self.train_idx],
                self.y[self.train_idx],
                self.bc[self.train_idx],
                self.labels[self.train_idx],
                self.y_aux[self.train_idx],
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(
                self.x[self.val_idx],
                self.y[self.val_idx],
                self.bc[self.val_idx],
                self.labels[self.val_idx],
                self.y_aux[self.val_idx],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(
                self.x[self.test_idx],
                self.y[self.test_idx],
                self.bc[self.test_idx],
                self.labels[self.test_idx],
                self.y_aux[self.test_idx],
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(
                self.x[self.test_idx],
                self.y[self.test_idx],
                self.bc[self.test_idx],
                self.labels[self.test_idx],
                self.y_aux[self.test_idx],
            ),
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
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3], batch[4])

    print("checks passed")

    return


if __name__ == "__main__":
    print("running dataloader")
    test_dataloader()
