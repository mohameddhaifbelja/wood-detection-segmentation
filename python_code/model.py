
from pytorch_lightning import LightningModule
from torch import nn

import torchmetrics
import torch


class Classifier(LightningModule):

    def __init__(self, numChannels=1, classes=1):
        super().__init__()
        # Model Architecture
        self.model = nn.Sequential(
            nn.Linear(in_features=2680, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
            nn.Sigmoid()
        )


        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')

    def configure_optimizers(self, learning_rate=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def forward(self, x):

        x = torch.flatten(x, start_dim=2)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.to(torch.float)
        prediction = torch.flatten(self.forward(X)).to(torch.float)
        loss = nn.BCELoss(reduction='none')(prediction, y)
        self.log('train_loss', loss.mean())

        self.train_acc(prediction, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss.mean()

    def test_step(self, batch, batch_idx):
        X, y = batch
        y = y.to(torch.float)
        prediction = torch.flatten(self.forward(X)).to(torch.float)
        loss = nn.BCELoss(reduction='none')(prediction, y)
        self.log('test_loss', loss.mean())

        self.test_acc(prediction, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.to(torch.float)
        prediction = torch.flatten(self.forward(X)).to(torch.float)
        loss = nn.BCELoss(reduction='none')(prediction, y)
        self.log('val_loss', loss.mean())

        self.val_acc(prediction, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)

        return loss.mean()
