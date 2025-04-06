# Save model activations
import argparse
import itertools
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mtn.data import MriAuxStress
from mtn.model import SFCNAUX


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
    embeddings = torch.empty((len(data_loader.dataset), 64), device=device)
    ages = torch.empty((len(data_loader.dataset),), device=device)
    pred_ages = torch.empty(
        (
            len(
                data_loader.dataset,
            )
        ),
        device=device,
    )
    final = torch.empty((len(data_loader.dataset), 25), device=device)
    val_start_time = time.time()
    with torch.no_grad():
        start_ix = 0
        for _, (x, [_, bc, y_label], _) in enumerate(data_loader):
            x = x.to(device, non_blocking=True)
            bc = bc.to(device, non_blocking=True)
            y_label = y_label.to(device, non_blocking=True)

            end_ix = start_ix + x.size(0)
            embedding, last_emb, output_age = model.embed(x)
            embeddings[start_ix:end_ix] = embedding.squeeze()
            ages[start_ix:end_ix] = y_label
            final[start_ix:end_ix] = last_emb
            prob = output_age.detach().exp()
            pred = torch.einsum("bj,bj->b", prob.squeeze(), bc)
            pred_ages[start_ix:end_ix] = pred
            start_ix = end_ix

    print("Time elasped for Val: " + str(time.time() - val_start_time), flush=True)

    return embeddings.cpu().numpy(), final.cpu().numpy(), ages.cpu().numpy(), pred_ages.cpu().numpy()


start_time = time.time()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on", device, flush=True)

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mode_list = [0, 1]
cnt_list = [100, 250, 500, 1000, 2000]
fold_ix_list = [0, 1, 2, 3, 4]
permutations = list(itertools.product(seed_list, mode_list, cnt_list, fold_ix_list))
print(permutations)
print(len(permutations))
args = parse_args()

seed, mode, cnt, fold_ix = permutations[args.hn]
print(seed, mode, cnt, fold_ix)

# batch size is 1 so that we get 1 activation per image
UKBiobank_test = MriAuxStress()
test_loader = DataLoader(
    UKBiobank_test,
    batch_size=16,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

setup_seeds(seed)
model_name = f"Mode{mode}_seed{seed}_cnt{cnt}_fold{fold_ix}"
print(model_name, flush=True)
model_dir_ = Path("saved") / model_name
model_file = model_dir_ / "best.pt"

save_file = model_dir_ / "embeddings.npz"

model = SFCNAUX().to(device)
model.load_state_dict(torch.load(model_file, weights_only=True))
embeddings, final_layer, ages, pred_ages = test(model, device, test_loader)
np.savez_compressed(save_file, embeddings=embeddings, ages=ages, pred_ages=pred_ages, final_layer=final_layer)
print("saved")
