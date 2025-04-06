# Calculate all pairwise similarities (linear CKA) between representations of the same subject from the same model but trained with different random initializations.
import argparse
import glob
import itertools
import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader


# https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py
class CudaCka(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def kernel_hsic(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_hsic(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_cka(self, X, Y):
        hsic = self.linear_hsic(X, Y)
        var1 = torch.sqrt(self.linear_hsic(X, X))
        var2 = torch.sqrt(self.linear_hsic(Y, Y))

        return hsic / (var1 * var2)

    def kernel_cka(self, X, Y, sigma=None):
        hsic = self.kernel_hsic(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_hsic(X, X, sigma))
        var2 = torch.sqrt(self.kernel_hsic(Y, Y, sigma))
        return hsic / (var1 * var2)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on", device, flush=True)
cuda_cka = CudaCka(device)
print("Variability in seeds")

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mode_list = [0, 1]
cnt_list = [1000, 2000, 5000, 10000, 20000]
fold_ix_list = [0, 1, 2, 3, 4]
mode_ls_cka = []
mode_ls_age = []

for mode in mode_list:
    cnt_ls_age = []
    cnt_ls_cka = []
    for cnt in cnt_list:
        seed_pairs_age = []
        seed_pairs_cka = []
        for fold_ix in fold_ix_list:
            for i, seed_i in enumerate(seed_list):
                model_name_i = f"m{mode}_s{seed_i}_size{cnt}_f{fold_ix}"
                save_file_i = glob.glob(f"/data/users1/dkim195/multitaskNeuro/data/checkpoints/*{model_name_i}/*.npz")[
                    0
                ]
                data_i = np.load(save_file_i, allow_pickle=True)
                embeddings_i = torch.from_numpy(data_i["embeddings"]).to(device)
                pred_age_i = data_i["pred_ages"]
                ages_i = data_i["ages"]
                for j, seed_j in enumerate(seed_list):
                    if j > i:
                        model_name_j = f"m{mode}_s{seed_j}_size{cnt}_f{fold_ix}"
                        save_file_j = glob.glob(
                            f"/data/users1/dkim195/multitaskNeuro/data/checkpoints/*{model_name_j}/*.npz"
                        )[0]
                        data_j = np.load(save_file_j, allow_pickle=True)
                        embeddings_j = torch.from_numpy(data_j["embeddings"]).to(device)
                        pred_age_j = data_j["pred_ages"]
                        ages_j = data_j["ages"]

                        cka = float(cuda_cka.linear_cka(embeddings_i, embeddings_j))
                        seed_pairs_cka.append(cka)
                        age_variance = np.mean(np.abs(pred_age_i - pred_age_j))
                        seed_pairs_age.append(age_variance)
        print(mode, cnt, " (cka) ", np.mean(seed_pairs_cka), " (age) ", np.mean(seed_pairs_age))
        cnt_ls_cka.append(np.asarray(seed_pairs_cka))
        cnt_ls_age.append(np.asarray(seed_pairs_age))

    mode_ls_cka.append(cnt_ls_cka)
    mode_ls_age.append(cnt_ls_age)

for i, cnt in enumerate(cnt_list):
    r_cka, p_cka = ttest_ind(mode_ls_cka[1][i], mode_ls_cka[0][i])
    r_age, p_age = ttest_ind(mode_ls_age[1][i], mode_ls_age[0][i])
    print("cka ", cnt, r_cka, p_cka)
    print("age ", cnt, r_age, p_age)
