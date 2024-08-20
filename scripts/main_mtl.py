import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from multitaskNeuro.data import MriMTL
from multitaskNeuro.sfcn_processed import SFCNMTL
from multitaskNeuro.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="None", help="Output file.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cnt", type=int, default=-1)

    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    print("Epoch:", epoch, flush=True)
    epoch_start_time = time.time()
    batch_start_time = time.time()
    mae = 0
    acc = 0
    for batch_idx, (x, [y_age, bc, label], [y_sex]) in enumerate(train_loader):
        batch_size = x.shape[0]
        x = x.to(device, non_blocking=True)
        y_age = y_age.to(device, non_blocking=True)
        y_sex = y_sex.to(device, non_blocking=True)

        output_age = model(x, "age")
        output_age = output_age.reshape([batch_size, -1])
        loss_age = kldivloss(output_age, y_age)

        output_sex = model(x, "sex")

        output_sex = output_sex.reshape([batch_size, -1])
        loss_sex = F.nll_loss(output_sex, y_sex)

        loss = loss_age + loss_sex

        train_loss += float(loss.detach())

        output_age = output_age.detach().cpu().numpy()
        bc = bc.numpy()
        label = label.numpy()

        output_sex = output_sex.detach().cpu().numpy()
        y_sex = y_sex.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(batch_size):
            prob = np.exp(output_age[i])
            pred = prob @ bc[i]
            mae += abs(pred - label[i])

        for i in range(batch_size):
            prob = np.exp(output_sex[i])
            if np.argmax(prob) == int(y_sex[i]):
                acc += 1

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
    acc /= len(train_loader.dataset)

    print("Training set: Average loss: {:.6f}, MAE: {}, ACC: {}".format(avg_loss, mae, acc), flush=True)

    print("Time elasped per epoch: " + str(time.time() - epoch_start_time), flush=True)

    return avg_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    mae = 0
    acc = 0
    val_start_time = time.time()
    with torch.no_grad():
        for batch_idx, (x, [y_age, bc, label], [y_sex]) in enumerate(test_loader):
            batch_size = x.shape[0]
            x = x.to(device, non_blocking=True)
            y_age = y_age.to(device, non_blocking=True)
            y_sex = y_sex.to(device, non_blocking=True)

            output_age = model(x, "age")
            output_age = output_age.reshape([batch_size, -1])
            loss_age = kldivloss(output_age, y_age)

            output_sex = model(x, "sex")

            output_sex = output_sex.reshape([batch_size, -1])
            loss_sex = F.nll_loss(output_sex, y_sex)

            loss = loss_age + loss_sex

            test_loss += float(loss.detach())

            output_age = output_age.detach().cpu().numpy()
            bc = bc.numpy()
            label = label.numpy()

            output_sex = output_sex.detach().cpu().numpy()
            y_sex = y_sex.detach().cpu().numpy()

            for i in range(batch_size):
                prob = np.exp(output_age[i])
                pred = prob @ bc[i]
                mae += abs(pred - label[i])

            for i in range(batch_size):
                prob = np.exp(output_sex[i])
                if np.argmax(prob) == int(y_sex[i]):
                    acc += 1

    avg_loss = test_loss / (batch_idx + 1)
    mae /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)

    print("Validation set: Average loss: {:.6f}, MAE: {}, ACC: {}".format(avg_loss, mae, acc), flush=True)

    print("Time elasped for Val: " + str(time.time() - val_start_time), flush=True)

    return avg_loss, mae


args = parse_args()
setup_seeds(args.seed)
start_time = time.time()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Training on", device, flush=True)

print(args)

model = SFCN_MTL().to(device)

lr_val = 0.01


UKBiobank_train = MriMTL("train", args.seed, args.cnt)
train_loader = DataLoader(
    UKBiobank_train,
    batch_size=4,
    shuffle=True,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)
UKBiobank_test = MriMTL("test", args.seed, args.cnt)
test_loader = DataLoader(
    UKBiobank_test,
    batch_size=4,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
)

# train_loader = DataLoader(UKBiobank_train, batch_size=4)
# valid_loader = DataLoader(UKBiobank_valid, batch_size=4)

training_loss = []
validation_loss = []
current_epoch_time = time.time()
current_batch_time = time.time()

print("Setup Time: " + str(time.time() - start_time), flush=True)

epochs = args.epochs
os.mkdir(args.output_dir)

best_mae = 10000
best_mae_epoch = -1

for epoch in range(1, epochs + 1):
    if epoch != 0 and epoch % 30 == 0:
        lr_val *= 0.3
    optimizer = optim.SGD(model.parameters(), lr=lr_val, weight_decay=0.001)

    train_loss = train(model, device, train_loader, optimizer, epoch)
    test_loss, mae = test(model, device, test_loader)

    if mae < best_mae:
        best_mae = mae
        best_mae_epoch = epoch

    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    model_file = args.output_dir + "/saved_" + str(epoch) + ".pt"
    torch.save(model.state_dict(), model_file)

print("Best mae: ", best_mae, " | epoch: ", best_mae_epoch)
np.save(args.output_dir + "/training", np.array(training_loss))
np.save(args.output_dir + "/validation", np.array(validation_loss))

print("Total Time elasped: " + str(time.time() - start_time), flush=True)
