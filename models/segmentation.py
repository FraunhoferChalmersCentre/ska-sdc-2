from typing import Any, List, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.data.ska_dataset import AbstractSKADataset


class Segmenter(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fct: Any, training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset, batch_size=32, x_key='image', y_key='segmentmap'):
        super().__init__()
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.training_set = training_set
        self.y_key = y_key
        self.x_key = x_key
        self.loss_fct = loss_fct
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)

    def forward(self, x):
        print('forward!')
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
