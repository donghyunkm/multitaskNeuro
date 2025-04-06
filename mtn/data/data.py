import os
import random
import warnings
from itertools import islice

import lightning as L
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from mtn.utils import get_paths


class FmriDataModule(L.LightningDataModule):
    """
    Data module using PyG functions to return graph patches.
    """

    def __init__(
        self, batch_size: int = 128, quantile: int = 31, rand_seed: int = 42, trainval_size: int = -1, fold_ix: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.quantile = quantile
        self.rand_seed = rand_seed
        self.trainval_size = trainval_size
        self.fold_ix = fold_ix

    def prepare_data(self):
        data_dir = get_paths()["data_root"]
        self.data_dir = data_dir
        features = np.load(data_dir + "triangle.npy")
        labels = np.load(data_dir + "age_labels_norm.npy")
        labels_aux = np.load(data_dir + "labels_aux.npy")
        x = torch.from_numpy(features)
        x = x.to(torch.float32)
        self.x = x
        # if self.quantile < 31:
        #     y = torch.tensor(pd.qcut(labels, q=self.quantile, labels=False))
        # else:
        #     y = torch.tensor(labels)
        # self.y = y
        self.y = torch.tensor(labels)
        self.y_aux = torch.from_numpy(labels_aux)

    def get_indices(self):
        # indices = np.arange(len(self.y))
        # train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=self.y, random_state=self.rand_seed)
        # train_idx, val_idx = train_test_split(
        #     train_val_idx, test_size=0.2, stratify=self.y[train_val_idx], random_state=self.rand_seed
        # )

        self.trainvalidx = np.load(self.data_dir + "trainval.npy")
        print(f"Using the following trainval size: {self.trainval_size}")
        sss = StratifiedShuffleSplit(
            n_splits=5,
            test_size=int(self.trainval_size * 0.2),
            train_size=int(self.trainval_size * 0.8),
            random_state=0,
        )

        splits_generator = sss.split(np.zeros(len(self.trainvalidx)), self.y_aux[self.trainvalidx])
        print(f"Using the following fold index: {self.fold_ix}")
        train_idx, val_idx = next(islice(splits_generator, self.fold_ix, None))

        train_idx = self.trainvalidx[train_idx]
        val_idx = self.trainvalidx[val_idx]

        test_idx = np.load(self.data_dir + "unseen_test.npy")

        ood_idx = np.load(self.data_dir + "ood_test.npy")

        return train_idx, val_idx, test_idx, ood_idx

    def setup(self, stage: str):
        self.prepare_data()
        self.train_idx, self.val_idx, self.test_idx, self.ood_idx = self.get_indices()

    def train_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.train_idx], self.y[self.train_idx], self.y_aux[self.train_idx]),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.val_idx], self.y[self.val_idx], self.y_aux[self.val_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.test_idx], self.y[self.test_idx], self.y_aux[self.test_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )

    def ood_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.ood_idx], self.y[self.ood_idx], self.y_aux[self.ood_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=TensorDataset(self.x[self.test_idx], self.y[self.test_idx], self.y_aux[self.test_idx]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )


def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test_dataloader():
    import numpy as np

    datamodule = FmriDataModule(batch_size=4, quantile=-1, rand_seed=42, train_size=-1)

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
