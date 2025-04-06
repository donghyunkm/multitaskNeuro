import csv
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC


class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.hidden_sizes = [256, 32]
        self.input_size = 1431
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc_label = nn.Linear(self.hidden_sizes[1], num_classes)
        self.fc_aux = nn.Linear(self.hidden_sizes[1], 5)
        self.dropout = nn.Dropout(0.5)
        # self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        label = self.fc_label(x)
        aux_label = self.fc_aux(x)

        return label, aux_label

    def embed(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        embedding = self.relu(x)

        label = self.fc_label(embedding)
        label = torch.argmax(label, axis=1)
        return label, embedding


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
            self.transform = lambda x: x
        else:
            self.transform = lambda x: x

        self.mode = cfg.mode

        print(f"Using the following mode: {self.mode}")

        # losses
        self.loss_ce = nn.CrossEntropyLoss()

        self.num_classes = cfg.num_classes
        # metrics
        options = {"num_classes": cfg.num_classes, "top_k": 1, "multidim_average": "global"}
        self.train_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=self.num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (n_subjects, fnc)
        """
        age, aux = self.model(x)
        return age, aux

    def embed(self, x):
        label, embedding = self.model.embed(x)
        return label, embedding

    def training_step(self, batch, batch_idx):
        fnc, y, y_aux = batch
        fnc = self.transform(fnc)
        y = y.type(torch.LongTensor).to(device=fnc.device)
        y_aux = y_aux.type(torch.LongTensor).to(device=fnc.device)
        age_pred, aux_pred = self.forward(x=fnc)

        # calculate loss
        train_loss_ce = self.loss_ce(age_pred, y)
        train_loss_aux_ce = self.loss_ce(aux_pred, y_aux)
        loss = train_loss_ce + train_loss_aux_ce
        # calculate metrics
        self.train_auroc.update(age_pred, y)

        if self.mode == 0:
            self.log("train_loss_ce", train_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return train_loss_ce
        else:
            self.log("train_loss_ce", train_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_aux_loss_ce", train_loss_aux_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_combined_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

    def on_train_epoch_end(self):
        train_auroc = self.train_auroc.compute()
        self.log("train_auroc", train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.train_auroc.reset()
        return

    def validation_step(self, batch, batch_idx):
        fnc, y, y_aux = batch
        fnc = self.transform(fnc)
        y = y.type(torch.LongTensor).to(device=fnc.device)
        y_aux = y_aux.type(torch.LongTensor).to(device=fnc.device)
        age_pred, aux_pred = self.forward(x=fnc)

        # calculate loss
        val_loss_ce = self.loss_ce(age_pred, y)
        val_loss_aux_ce = self.loss_ce(aux_pred, y_aux)
        loss = val_loss_ce + val_loss_aux_ce
        # calculate metrics
        self.val_auroc.update(age_pred, y)

        if self.mode == 0:
            self.log("val_loss_ce", val_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log("val_loss_ce", val_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_aux_loss_ce", val_loss_aux_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_combined_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        val_auroc = self.val_auroc.compute()
        self.log("val_auroc", val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_auroc.reset()
        return

    def test_step(self, batch, batch_idx):
        fnc, y, y_aux = batch
        fnc = self.transform(fnc)
        y = y.type(torch.LongTensor).to(device=fnc.device)
        y_aux = y_aux.type(torch.LongTensor).to(device=fnc.device)
        age_pred, aux_pred = self.forward(x=fnc)

        # calculate loss
        test_loss_ce = self.loss_ce(age_pred, y)
        test_loss_aux_ce = self.loss_ce(aux_pred, y_aux)
        loss = test_loss_ce + test_loss_aux_ce
        # calculate metrics
        self.test_auroc.update(age_pred, y)

        if self.mode == 0:
            self.log("test_loss_ce", test_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log("test_loss_ce", test_loss_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("test_aux_loss_ce", test_loss_aux_ce, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("test_combined_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        test_auroc = self.test_auroc.compute()
        self.log("test_auroc", test_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_auroc.reset()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.trainer.lr, weight_decay=0.001)
        return optimizer
