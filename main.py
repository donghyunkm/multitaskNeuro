# Train, save, and evaluate models
import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from multitaskNeuro.data import MriAux
from multitaskNeuro.model import SFCNAUX
from multitaskNeuro.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="None", required=True, help="Output file.")
    parser.add_argument("--seed", type=int, default=10, required=True)
    parser.add_argument("--cnt", type=int, default=-1, required=True)
    parser.add_argument("--mode", type=int, default=0, required=True)
    parser.add_argument("--fold_ix", type=int, default=0, required=True)

    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, train_loader, optimizer, epoch, mode):
    model.train()
    train_loss = 0
    print("Epoch:", epoch, flush=True)
    epoch_start_time = time.time()
    batch_start_time = time.time()
    mae = 0
    cos_list = []

    acc = Accuracy(task="multiclass", num_classes=5).to(device)

    for batch_idx, (x, [y_age, bc, label], y_aux, y_sex) in enumerate(train_loader):
        batch_cos_list = []
        batch_size = x.shape[0]
        x = x.to(device, non_blocking=True)
        y_age = y_age.to(device, non_blocking=True)
        y_sex = y_sex.to(device, non_blocking=True)
        y_aux = y_aux.to(device, non_blocking=True)
        bc = bc.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        output_age, output_aux = model(x)
        output_age = output_age.reshape([batch_size, -1])
        output_aux = output_aux.reshape([batch_size, -1])
        loss_age = kldivloss(output_age, y_age)
        print(y_aux.size(), y_aux)
        if mode == 1:
            loss_aux = F.nll_loss(output_aux, y_aux)
        else:
            loss_aux = F.nll_loss(output_aux, y_sex)

        if mode == 0:
            loss = loss_age
        elif mode > 0:
            # if epoch < 100:
            # else:
            #    loss = loss_age
            loss = loss_age + loss_aux

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prob = output_age.detach().exp()
        pred = torch.einsum("bj,bj->b", prob, bc)
        mae += (pred - label).abs().sum().detach().cpu()
        print(output_aux.exp().argmax(dim=-1))
        print(acc(output_aux.exp(), y_aux))

        if batch_idx % 1000 == 0:
            print(
                "Training set [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    batch_idx * len(x), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()
                ),
                flush=True,
            )

            print("Time elasped per 1000 batches: " + str(time.time() - batch_start_time), flush=True)
            batch_start_time = time.time()

    avg_loss = train_loss / (batch_idx + 1)
    mae /= len(train_loader.dataset)

    print("Training set: Average loss: {:.6f}, MAE: {}".format(avg_loss, mae), flush=True)

    print("Time elasped per epoch: " + str(time.time() - epoch_start_time), flush=True)

    return avg_loss, cos_list


def test(model, device, data_loader):
    model.eval()
    test_loss = 0
    mae = 0
    acc = Accuracy(task="multiclass", num_classes=5).to(device)
    val_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, [y_age, bc, label], y_aux, y_sex) in enumerate(data_loader):
            batch_size = x.shape[0]
            x = x.to(device, non_blocking=True)
            y_age = y_age.to(device, non_blocking=True)
            y_sex = y_sex.to(device, non_blocking=True)
            y_aux = y_aux.to(device, non_blocking=True)
            bc = bc.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.no_grad():
                output_age, output_aux = model(x)
                output_age = output_age.reshape([batch_size, -1])
                output_aux = output_aux.reshape([batch_size, -1])
                prob = output_age.exp()
                pred = torch.einsum("bj,bj->b", prob, bc)
            loss_age = kldivloss(output_age, y_age)
            if mode == 1:
                loss_aux = F.nll_loss(output_aux, y_aux)
            else:
                loss_aux = 2 * F.nll_loss(output_aux, y_sex)

            loss = loss_age + loss_aux

            test_loss += float(loss.detach() * x.size(0))
            mae += (pred - label).abs().sum().detach()
            print(output_aux.exp().argmax(dim=-1))
            print(acc(output_aux.exp(), y_aux))

    avg_loss = test_loss / len(data_loader.dataset)
    mae = mae / len(data_loader.dataset)

    print("Validation set: Average loss: {:.6f}, MAE: {}".format(avg_loss, mae), flush=True)

    print("Time elasped for Val: " + str(time.time() - val_start_time), flush=True)

    return avg_loss, mae


args = parse_args()
save_dir = Path(args.output_dir)
setup_seeds(args.seed)
start_time = time.time()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Training on", device, flush=True)

print(args)
aux_size = 2 if args.mode == 2 else 5

model = SFCNAUX(aux_size).to(device)

lr_val = 0.01

UKBiobank_train = MriAux("train", args.seed, args.cnt, args.fold_ix)
train_loader = DataLoader(
    UKBiobank_train,
    batch_size=32,
    shuffle=True,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
UKBiobank_valid = MriAux("valid", args.seed, args.cnt, args.fold_ix)
valid_loader = DataLoader(
    UKBiobank_valid,
    batch_size=32,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

training_loss = []
validation_loss = []
current_epoch_time = time.time()
current_batch_time = time.time()

print("Setup Time: " + str(time.time() - start_time), flush=True)

epochs = args.epochs
mode = args.mode
save_dir.mkdir(exist_ok=True)
model_name = f"Mode{mode}_seed{args.seed}_cnt{args.cnt}_fold{args.fold_ix}"

save_dir = save_dir / model_name
save_dir.mkdir(exist_ok=True)

best_mae = 10000
best_mae_epoch = -1
cos_list_comb = []
for epoch in range(1, epochs + 1):
    if epoch != 0 and epoch % 30 == 0:
        lr_val *= 0.3
    optimizer = optim.SGD(model.parameters(), lr=lr_val, weight_decay=0.001)

    train_loss, cos_list = train(model, device, train_loader, optimizer, epoch, mode)
    cos_list_comb.append(cos_list)
    valid_loss, mae = test(model, device, valid_loader)
    if mae < best_mae:
        best_mae = mae
        best_mae_epoch = epoch
        model_file = save_dir / "best.pt"
        torch.save(model.state_dict(), model_file)
    training_loss.append(train_loss)
    validation_loss.append(valid_loss)
    model_file = save_dir / "last.pt"
    torch.save(model.state_dict(), model_file)

UKBiobank_test = MriAUX("test", args.seed, args.cnt, args.fold_ix)
test_loader = DataLoader(
    UKBiobank_test,
    batch_size=4,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
model.load_state_dict(torch.load(save_dir / "best.pt", weights_only=True))
print(model)
test_loss, test_mae = test(model, device, test_loader)
print("Final mae: ", test_mae.detach(), " | epoch: ", best_mae_epoch)
np.save(save_dir / "training.npy", np.array(training_loss))
np.save(save_dir / "validation.npy", np.array(validation_loss))
print("Total Time elasped: " + str(time.time() - start_time), flush=True)
