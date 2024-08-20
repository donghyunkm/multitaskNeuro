from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset

from multitaskNeuro.utils import *


def get_df(seed, cnt):
    df = pd.read_csv("data/my_ukb_data_smri_only_relevant_col.csv", low_memory=False)

    if cnt == -1:
        cnt = len(df)

    df = df.groupby("age_when_attended_assessment_centre_f21003_2_0").filter(lambda x: len(x) > 1)
    X = df["eid"]
    y_age = df["age_when_attended_assessment_centre_f21003_2_0"]
    y_sex = df["sex_f31_0_0"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=int(cnt * 0.1), train_size=int(cnt * 0.9), random_state=0)

    for train, test in sss.split(X, y_age):
        X_train = X.iloc[train]
        y_train_age = y_age.iloc[train]
        y_train_sex = y_sex.iloc[train]
        X_test = X.iloc[test]
        y_test_age = y_age.iloc[test]
        y_test_sex = y_sex.iloc[test]

    train_df = pd.DataFrame({"X": X_train, "y_age": y_train_age, "y_sex": y_train_sex})
    train_df.reset_index(inplace=True, drop=True)

    test_df = pd.DataFrame({"X": X_test, "y_age": y_test_age, "y_sex": y_test_sex})
    test_df.reset_index(inplace=True, drop=True)

    return train_df, test_df


class Mri(Dataset):
    def __init__(self, data_type, seed, label_input, cnt):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        self.label_input = label_input
        train_df, test_df = get_df(seed, cnt)

        if data_type == "train":
            self.df = train_df.copy()
        elif data_type == "test":
            self.df = test_df.copy()

        # self.indices = self.df.index.values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        eid = self.df.loc[i, "X"]

        x = nib.load(
            "/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/" + str(eid) + "/ses_01/anat/Sm6mwc1pT1.nii.nii"
        ).get_fdata()
        x -= x.mean()

        x = torch.from_numpy(x).float()

        aug = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(0.75, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        if self.label_input == "age":
            label = self.df.loc[i, "y_age"]
            y, bc = num2vect(label, bin_range, bin_step, sigma)
            if self.data_type == "train":
                x = aug(x)
            return x.unsqueeze(0), torch.from_numpy(y).float(), bc, label

        if self.label_input == "sex":
            label2 = self.df.loc[i, "y_sex"]
            if self.data_type == "train":
                x = aug(x)
            return x.unsqueeze(0), label2


class MriMTL(Dataset):
    def __init__(self, data_type, seed, cnt):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        train_df, test_df = get_df(seed, cnt)

        if data_type == "train":
            self.df = train_df.copy()
        elif data_type == "test":
            self.df = test_df.copy()

        # self.indices = self.df.index.values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        eid = self.df.loc[i, "X"]

        x = nib.load(
            "/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/" + str(eid) + "/ses_01/anat/Sm6mwc1pT1.nii.nii"
        ).get_fdata()
        x -= x.mean()

        x = torch.from_numpy(x).float()

        aug = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(0.75, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        if self.data_type == "train":
            x = aug(x)
        sex = self.df.loc[i, "y_sex"]

        label_age = self.df.loc[i, "y_age"]
        y, bc = num2vect(label_age, bin_range, bin_step, sigma)

        return x.unsqueeze(0), [torch.from_numpy(y).float(), bc, label_age], [sex]


class MriAUX(Dataset):
    def __init__(self, data_type, seed, cnt):
        super().__init__()
        self.data_type = data_type
        self.seed = seed
        train_df, test_df = get_df(seed, cnt)

        if data_type == "train":
            self.df = train_df.copy()
        elif data_type == "test":
            self.df = test_df.copy()

        # self.indices = self.df.index.values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        bin_range = [42, 82]
        bin_step = 1
        sigma = 1
        eid = self.df.loc[i, "X"]

        x = nib.load(
            "/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/" + str(eid) + "/ses_01/anat/Sm6mwc1pT1.nii.nii"
        ).get_fdata()
        x -= x.mean()

        x = torch.from_numpy(x).float()

        aug = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(0.75, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        if self.data_type == "train":
            x = aug(x)
        sex = self.df.loc[i, "y_sex"]

        label_age = self.df.loc[i, "y_age"]
        y, bc = num2vect(label_age, bin_range, bin_step, sigma)

        auxiliary_age = 0

        if label_age < 56:
            auxiliary_age = 0
        elif label_age < 62:
            auxiliary_age = 1
        elif label_age < 66:
            auxiliary_age = 2
        elif label_age < 71:
            auxiliary_age = 3
        else:
            auxiliary_age = 4

        return x.unsqueeze(0), [torch.from_numpy(y).float(), bc, label_age], [auxiliary_age]
