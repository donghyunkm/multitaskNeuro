from datetime import datetime
from pathlib import Path

import numpy as np
import toml
import torch.nn as nn
from scipy.stats import norm


def clean_csv(seed):
    index_to_drop = []
    # df = pd.read_csv('/data/qneuromark/Data/UKBiobank/Data_info/New_rda/my_ukb_data_smri.csv', low_memory = False)
    df = pd.read_csv("/data/users2/zfu/Matlab/GSU/HCP_2023_Project/T1_UKB/my_ukb_data_smri.csv", low_memory=False)

    for i in df.index:
        eid = df.loc[i, "eid"]
        if not Path(
            "/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data/" + str(eid) + "/ses_01/anat/T1.nii.gz"
        ).exists():
            index_to_drop.append(i)
    df.drop(df.index[index_to_drop], inplace=True)
    # df.to_csv('my_ukb_data_smri_missing_dropped.csv')
    df.to_csv("my_ukb_data_smri.csv", index=False)

    trainval_index, test_index = train_test_split(df.index.values, train_size=0.8, random_state=seed)
    train_index, valid_index = train_test_split(trainval_index, train_size=0.9, random_state=seed)
    train_df = df.loc[train_index].copy()
    valid_df = df.loc[valid_index].copy()
    test_df = df.loc[test_index].copy()
    return train_df, valid_df, test_df


def num2vect(x, bin_range, bin_step, sigma):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.

    Function from (Peng et al., 2021)
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


def kldivloss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem

    Function from (Peng et al., 2021)
    """
    loss_func = nn.KLDivLoss(reduction="sum")
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    # print(loss)
    return loss


def get_datetime(expname: str = ""):
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if expname is None:
        expname = datetime_str
    else:
        expname = f"{datetime_str}_{expname}"
    return expname


# datareaders.py functionality from nichecompass
def get_paths(verbose: bool = False) -> dict:
    """
    Get custom paths from config.toml in the package root directory.
    """

    # get path of this file
    root_path = Path(__file__).parent.parent
    config_path = root_path / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config = toml.load(config_path)
    config["package_root"] = root_path
    if verbose:
        print(config)
    return config
