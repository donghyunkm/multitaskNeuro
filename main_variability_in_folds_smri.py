# Calculate all pairwise similarities (linear CKA) between representations of the same subject from the same model but trained with different folds.
import pickle
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from scipy.stats import ttest_ind


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

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mode_list = [0, 1]
cnt_list = [100, 250, 500, 1000, 2000]
fold_ix_list = [0, 1, 2, 3, 4]
age_list = [i for i in range(26)]

mode_ls = []
for mode in mode_list:
    cnt_ls = []
    for cnt in cnt_list:
        seed_pairs = []
        for seed in seed_list:
            for i, fold_i in enumerate(fold_ix_list):
                model_name_i = f"Mode{mode}_seed{seed}_cnt{cnt}_fold{fold_i}"
                model_dir_i = Path("saved") / model_name_i
                save_file_i = model_dir_i / "embeddings.npz"
                data_i = np.load(save_file_i, allow_pickle=True)
                embeddings_i = torch.from_numpy(data_i["embeddings"]).to(device)
                pred_age_i = data_i["pred_ages"]
                ages_i = data_i["ages"]
                for j, fold_j in enumerate(fold_ix_list):
                    if j > i:
                        model_name_j = f"Mode{mode}_seed{seed}_cnt{cnt}_fold{fold_j}"
                        model_dir_j = Path("saved") / model_name_j
                        save_file_j = model_dir_j / "embeddings.npz"

                        data_j = np.load(save_file_j, allow_pickle=True)
                        embeddings_j = torch.from_numpy(data_j["embeddings"]).to(device)
                        pred_age_j = data_j["pred_ages"]
                        ages_j = data_j["ages"]

                        # cka = float(cuda_cka.linear_cka(embeddings_i, embeddings_j))
                        # cossim = float(F.cosine_similarity(embeddings_i, embeddings_j, dim=1).mean(0))
                        seed_pairs.append(np.abs(pred_age_i - pred_age_j).mean())
                        # seed_pairs.append(cka)
        print(mode, cnt, np.round(np.mean(seed_pairs), 4))
        cnt_ls.append(np.asarray(seed_pairs))
    mode_ls.append(cnt_ls)

for i, cnt in enumerate(cnt_list):
    r, p = ttest_ind(mode_ls[1][i], mode_ls[0][i])
    print(cnt, r, p)
