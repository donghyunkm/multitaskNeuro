# Save model activations (fMRI)
import argparse
import glob
import itertools
import random
import time

import numpy as np
import torch
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from mtn.data.data import FmriDataModule
from mtn.models.models import MLP, LitMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument("--hn", type=int, default=0)

    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(model, device, data_loader):
    model.eval()
    embeddings = torch.empty((len(data_loader.dataset), 32), device=device)
    ages = torch.empty((len(data_loader.dataset),), device=device)
    pred_ages = torch.empty(
        (
            len(
                data_loader.dataset,
            )
        ),
        device=device,
    )

    val_start_time = time.time()
    with torch.no_grad():
        start_ix = 0
        for _, (fnc, y, y_aux) in enumerate(data_loader):
            fnc = fnc.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            end_ix = start_ix + fnc.size(0)
            label, embedding = model.embed(fnc)

            embeddings[start_ix:end_ix] = embedding.squeeze()
            ages[start_ix:end_ix] = y
            pred_ages[start_ix:end_ix] = label
            start_ix = end_ix

    print("Time elasped for Emb: " + str(time.time() - val_start_time), flush=True)

    return embeddings.cpu().numpy(), pred_ages.cpu().numpy(), ages.cpu().numpy()


seed_everything(42, workers=False)
setup_seeds(42)

start_time = time.time()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on", device, flush=True)

args = parse_args()

ckpts = [f for f in glob.glob("/data/users1/dkim195/multitaskNeuro/data/checkpoints/*/*.ckpt") if "_0303_" in f]

model_file = ckpts[args.hn]
model_dir = model_file.rsplit("/", 1)[0] + "/"
datamodule = FmriDataModule(batch_size=128, quantile=31, rand_seed=42, trainval_size=1000, fold_ix=0)

datamodule.setup("test")
# test_loader = datamodule.test_dataloader()
test_loader = datamodule.ood_dataloader()

print(model_dir, flush=True)

# save_file = model_dir + "embeddings.npz"
save_file = model_dir + "embeddings_ood.npz"

model = LitMLP.load_from_checkpoint(model_file)

embeddings, pred_ages, ages = test(model, device, test_loader)
np.savez_compressed(save_file, embeddings=embeddings, pred_ages=pred_ages, ages=ages)
print("saved")
