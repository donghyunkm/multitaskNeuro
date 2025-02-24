import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC
from torchvision.transforms import v2

from mtn.utils import kldivloss


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.hidden_sizes = [256, 64]
        self.input_size = 1431
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1], num_classes)

        self.dropout = nn.Dropout(0.5)
        # self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x


class MLPkl(nn.Module):
    def __init__(self, num_classes):
        super(MLPkl, self).__init__()
        self.hidden_sizes = [256, 64]
        self.input_size = 1431
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1], num_classes)

        self.dropout = nn.Dropout(0.5)
        # self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class LitMLP(L.LightningModule):
    def __init__(self, config):
        super(LitMLP, self).__init__()

        self.save_hyperparameters(config)

        # model
        cfg = self.hparams.model
        self.model = MLP(num_classes=cfg.num_classes)

        # related to training step
        self.transform = cfg.transform
        if self.transform:
            self.transform = v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.1)], p=0.2)
        else:
            self.transform = lambda x: x

        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        self.num_classes = cfg.num_classes
        # metrics
        options = {"num_classes": cfg.num_classes, "top_k": 1, "multidim_average": "global"}
        self.train_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (n_subjects, fnc)
        """
        age = self.model(x)
        return age

    def training_step(self, batch, batch_idx):
        fnc, y = batch
        fnc = self.transform(fnc)
        y = y.type(torch.LongTensor).to(device=fnc.device)
        age_pred = self.forward(x=fnc)

        # calculate loss
        train_loss_ce = self.loss_ce(age_pred, y)

        # calculate metrics
        self.train_auroc.update(age_pred, y)

        self.log("train_loss_ce", train_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss_ce

    def on_train_epoch_end(self):
        train_auroc = self.train_auroc.compute()
        self.log("train_auroc", train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_auroc.reset()
        return

    def validation_step(self, batch, batch_idx):
        fnc, y = batch
        # fnc = self.transform(fnc)
        y = y.type(torch.LongTensor).to(device=fnc.device)
        age_pred = self.forward(x=fnc)

        # calculate loss
        val_loss_ce = self.loss_ce(age_pred, y)

        # calculate metrics
        self.val_auroc.update(age_pred, y)

        self.log("val_loss_ce", val_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        self.log("val_auroc", val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_auroc.reset()
        return

    def configure_optimizers(self):
        if self.hparams.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.trainer.lr, weight_decay=0.001)
        elif self.hparams.optimizer.name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)
        elif self.hparams.optimizer.name == "rms":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.trainer.lr)
        return optimizer


class LitMLPkl(L.LightningModule):
    def __init__(self, config):
        super(LitMLPkl, self).__init__()

        self.save_hyperparameters(config)

        # model
        cfg = self.hparams.model
        self.model = MLPkl(num_classes=cfg.num_classes - 1)

        # related to training step
        self.transform = cfg.transform
        if self.transform:
            self.transform = v2.RandomApply([v2.GaussianNoise(mean=0.0, sigma=0.1)], p=0.2)
        else:
            self.transform = lambda x: x

        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        self.num_classes = cfg.num_classes - 1
        # metrics
        options = {"num_classes": cfg.num_classes, "top_k": 1, "multidim_average": "global"}
        # self.train_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        # self.val_auroc = AUROC(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (n_subjects, fnc)
        """
        age = self.model(x)
        return age

    def training_step(self, batch, batch_idx):
        fnc, y, bc, label, y_aux = batch
        fnc = self.transform(fnc)
        # y = y.type(torch.LongTensor).to(device=fnc.device)
        age_pred = self.forward(x=fnc)
        # calculate loss
        train_loss_kl = kldivloss(age_pred, y)

        # calculate metrics
        prob = age_pred.detach().exp()
        prob = prob.double()
        pred = torch.einsum("bj,bj->b", prob, bc)
        mae = (pred - label).abs().sum().detach().cpu()
        # self.train_auroc.update(age_pred, y)

        self.log("train_loss_kl", train_loss_kl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss_kl

    def on_train_epoch_end(self):
        # train_auroc = self.train_auroc.compute()
        # self.log("train_auroc", train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.train_auroc.reset()
        return

    def validation_step(self, batch, batch_idx):
        fnc, y, bc, label, y_aux = batch
        # fnc = self.transform(fnc)
        # y = y.type(torch.LongTensor).to(device=fnc.device)
        age_pred = self.forward(x=fnc)

        # calculate loss
        val_loss_kl = kldivloss(age_pred, y)

        # calculate metrics
        prob = age_pred.detach().exp()
        prob = prob.double()
        pred = torch.einsum("bj,bj->b", prob, bc)
        mae = (pred - label).abs().sum().detach().cpu()

        # self.val_auroc.update(age_pred, y)

        self.log("val_loss_kl", val_loss_kl, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mae", mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        # val_auroc = self.val_auroc.compute()
        # self.log("val_auroc", val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.val_auroc.reset()
        return

    def configure_optimizers(self):
        if self.hparams.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.trainer.lr, weight_decay=0.001)
        elif self.hparams.optimizer.name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.lr)
        elif self.hparams.optimizer.name == "rms":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.trainer.lr)
        return optimizer
