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
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on", device, flush=True)

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mode_list = [0, 1]
cnt_list = [1000, 2000, 5000, 10000, 20000]
fold_ix_list = [0, 1, 2, 3, 4]
mode_mae = [[[] for _ in range(5)] for _ in range(2)]

for i, mode in enumerate(mode_list):
    cnt_ls_age = []
    cnt_ls_cka = []
    for j, cnt in enumerate(cnt_list):
        for fold_ix in fold_ix_list:
            for seed_i in seed_list:
                model_name_i = f"m{mode}_s{seed_i}_size{cnt}_f{fold_ix}"
                save_file_i = glob.glob(f"/data/users1/dkim195/multitaskNeuro/data/checkpoints/*{model_name_i}/*.npz")[
                    0
                ]
                data_i = np.load(save_file_i, allow_pickle=True)
                pred_age_i = data_i["pred_ages"]
                ages_i = data_i["ages"]
                mae = mean_absolute_error(ages_i, pred_age_i)
                mode_mae[i][j].append(mae)
        print(mode, cnt, " (mae) ", np.mean(mode_mae[i][j]))

for i, cnt in enumerate(cnt_list):
    r_mae, p_mae = ttest_ind(mode_mae[1][i], mode_mae[0][i])
    print("mae ", cnt, r_mae, p_mae)
