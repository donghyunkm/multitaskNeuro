from itertools import islice
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset

from mtn.utils import *


def get_df(cnt, fold_ix):
    df = pd.read_csv("/data/users1/dkim195/multitaskNeuro/data/my_ukb_data_smri_age_bins.csv", low_memory=False)

    if cnt == -1:
        train_size = int(len(df) * 10 / 11)
        test_size = int(len(df) / 11)
    else:
        train_size = cnt
        test_size = cnt
    test_size = 2000

    # df = df.groupby("age_when_attended_assessment_centre_f21003_2_0").filter(lambda x: len(x) > 1)
    df = df[
        (df["age_when_attended_assessment_centre_f21003_2_0"] >= 50)
        & (df["age_when_attended_assessment_centre_f21003_2_0"] <= 75)
    ]
    X = df["eid"]
    y_age = df["age_when_attended_assessment_centre_f21003_2_0"]
    y_aux_bins = df["age_bins"]
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, train_size=train_size, random_state=0)

    splits_generator = sss.split(X, y_aux_bins)
    print(f"Using the following fold index: {fold_ix}")
    trainvalid_index, test_index = next(islice(splits_generator, fold_ix, None))

    X_test, y_test_age, y_test_aux_age = X.iloc[test_index], y_age.iloc[test_index], y_aux_bins.iloc[test_index]
    train_index, valid_index = train_test_split(
        trainvalid_index, train_size=0.8, shuffle=True, random_state=0, stratify=y_aux_bins.iloc[trainvalid_index]
    )
    X_train, y_train_age, y_train_aux_age = X.iloc[train_index], y_age.iloc[train_index], y_aux_bins.iloc[train_index]
    X_valid, y_valid_age, y_valid_aux_age = X.iloc[valid_index], y_age.iloc[valid_index], y_aux_bins.iloc[valid_index]

    train_df = pd.DataFrame({"X": X_train, "y_age": y_train_age, "y_aux_age": y_train_aux_age})
    train_df.reset_index(inplace=True, drop=True)

    valid_df = pd.DataFrame({"X": X_valid, "y_age": y_valid_age, "y_aux_age": y_valid_aux_age})
    valid_df.reset_index(inplace=True, drop=True)

    test_df = pd.DataFrame({"X": X_test, "y_age": y_test_age, "y_aux_age": y_test_aux_age})
    test_df.reset_index(inplace=True, drop=True)

    print("train, valid and test df", train_df.shape, valid_df.shape, test_df.shape)

    return train_df, valid_df, test_df


def get_df_stress():
    df = pd.read_csv(
        "/data/users1/dkim195/multitaskNeuro/data/my_ukb_data_smri_age_bins_unseen_full.csv", low_memory=False
    )
    X = df["eid"]
    y_age = df["age_when_attended_assessment_centre_f21003_2_0"]
    y_aux_bins = df["age_bins"]

    X_test, y_test_age, y_test_aux_age = X, y_age, y_aux_bins

    test_df = pd.DataFrame({"X": X_test, "y_age": y_test_age, "y_aux_age": y_test_aux_age})
    test_df.reset_index(inplace=True, drop=True)
    print("test df", test_df.shape)

    return test_df


class MriAux(Dataset):
    def __init__(self, data_type, cnt, fold_ix):
        super().__init__()
        self.data_type = data_type
        train_df, valid_df, test_df = get_df(cnt, fold_ix)
        data_p = Path("/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/")
        ext_p = "ses_01/anat/Sm6mwc1pT1.nii.nii"

        if data_type == "train":
            self.df = train_df.copy()
        elif data_type == "valid":
            self.df = valid_df.copy()
        elif data_type == "test":
            self.df = test_df.copy()

        self.data = np.empty((self.df.shape[0], 121, 145, 121), dtype=np.float32)
        print(f"Loading: {data_type} data")
        for i, row in self.df.iterrows():
            self.data[i] = nib.load(data_p / str(row["X"]) / ext_p).get_fdata().astype(np.float32)
            self.data[i] -= self.data[i].mean()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        bin_range = [50, 75]
        bin_step = 1
        sigma = 1
        x = self.data[i]
        x = torch.from_numpy(x).float()

        aug = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(0.75, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        if self.data_type == "train":
            x = aug(x)

        label_age = self.df.loc[i, "y_age"]
        y, bc = num2vect(label_age, bin_range, bin_step, sigma)

        auxiliary_age = self.df.loc[i, "y_aux_age"]

        return x.unsqueeze(0), [torch.from_numpy(y).float(), torch.from_numpy(bc).float(), label_age], auxiliary_age


class MriAuxStress(Dataset):
    def __init__(self):
        super().__init__()
        test_df = get_df_stress()
        data_p = Path("/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/")
        ext_p = "ses_01/anat/Sm6mwc1pT1.nii.nii"

        self.df = test_df.copy()

        self.data = np.empty((self.df.shape[0], 121, 145, 121), dtype=np.float32)
        print(f"Loading: test stress data")
        for i, row in self.df.iterrows():
            print(i / self.df.shape[0])
            self.data[i] = nib.load(data_p / str(row["X"]) / ext_p).get_fdata().astype(np.float32)
            self.data[i] -= self.data[i].mean()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        bin_range = [50, 75]
        bin_step = 1
        sigma = 1
        x = self.data[i]
        x = torch.from_numpy(x).float()

        label_age = self.df.loc[i, "y_age"]
        y, bc = num2vect(label_age, bin_range, bin_step, sigma)

        auxiliary_age = self.df.loc[i, "y_aux_age"]

        return x.unsqueeze(0), [torch.from_numpy(y).float(), torch.from_numpy(bc).float(), label_age], auxiliary_age
