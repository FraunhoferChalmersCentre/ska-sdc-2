from typing import Any
from itertools import starmap

import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from pytorch_toolbelt import losses

from utils.data.ska_dataset import AbstractSKADataset

def get_vis_id(dataset, shape, min_allocated):
    vis_id = None
    
    pos = dataset.get_attribute('position')
    alloc = dataset.get_attribute('allocated_voxels')
    while vis_id is None:
        random_id = np.random.randint(0, dataset.get_attribute('index'))
        candidate = True
        for r in pos[random_id].squeeze():
            
            for c, s in zip(r, shape):
                if torch.eq(c, 0) or torch.eq(c, s):
                    candidate = False
        
        if candidate and alloc[random_id].shape[0] > min_allocated:
            vis_id = random_id
    
    return vis_id

class Segmenter(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fct: Any, training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset, logger: pl.loggers.TensorBoardLogger, batch_size=128, x_key='image',
                 y_key='segmentmap', vis_max_angle=180, vis_rotations=4, vis_id=None, threshold = None):
        super().__init__()
        self.logger = logger
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.training_set = training_set
        self.y_key = y_key
        self.x_key = x_key
        self.loss_fct = loss_fct
        self.model = model
        self.vis_rotations = vis_rotations
        self.vis_max_angle = vis_max_angle

        self.vis_id = vis_id
        self.threshold = threshold
        self.log_image()
        
        self.metrics = {
            'Soft dice': losses.DiceLoss(mode='binary'),
            'Soft Jaccard': losses.JaccardLoss(mode='binary'),
            'Lovasz hinge': losses.BinaryLovaszLoss(),
            'Jaccard': self.jaccard,
            'Dice': self.dice,
            'Cross Entropy': self.bce_loss
        }
        
    def jaccard(self, y_hat, y):
        y_hat = torch.where(nn.Sigmoid()(y_hat).flatten() > self.threshold, 1, 0)
        y = torch.round(y.flatten())
        intersection = torch.sum(y_hat * y)

        union = y.sum() + y_hat.sum() - intersection
        print(intersection, y_hat.sum(), y.sum())
        if union < 1.:
            return torch.tensor(1.)
        return intersection / union


    def dice(self, y_hat, y):
        y_hat = torch.where(nn.Sigmoid()(y_hat).flatten() > self.threshold, 1, 0)
        y = torch.round(y.flatten())
        intersection = torch.sum(y_hat * y)
        denom = y.sum() + y_hat.sum()
        if denom < 1.:
            return torch.tensor(1.)
        return 2 * intersection / denom

    def bce_loss(self, y_hat, y):
        yf = y.flatten()
        y_hatf = y_hat.flatten()
        return nn.BCEWithLogitsLoss()(y_hatf, yf)
        
    def log_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d / 2),int(s / 2 - d / 2) + d), zip(image.shape, self.validation_set.get_attribute('dim'))))
        self._log_cross_sections(image[slices], self.validation_set[self.vis_id]['pa'], self.x_key)
        segmap = self.validation_set.get_attribute(self.y_key)[self.vis_id].squeeze()
        if segmap.sum() == 0:
            raise ValueError('Logged segmentmap contains no source voxels. Reshuffle!')
        self._log_cross_sections(segmap[slices], self.validation_set[self.vis_id]['pa'], self.y_key)
    
    def log_prediction_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d / 2),int(s / 2 - d / 2) + d), zip(image.shape, self.validation_set.get_attribute('dim'))))
        input_image = image[slices].unsqueeze(0).unsqueeze(0).to(self.device)
        prediction = nn.Sigmoid()(self.model(input_image)).squeeze()
        self._log_cross_sections(prediction, self.validation_set[self.vis_id]['pa'], 'Prediction')

    def _log_cross_sections(self, cube: torch.Tensor, pa: float, tag: str):
        for i in range(self.vis_rotations):
            rotated = TF.rotate(cube.squeeze().T, float(i * self.vis_max_angle / self.vis_rotations - pa))

            cropped_side = int(rotated.shape[1] / np.sqrt(2))

            cropped = TF.center_crop(rotated, [cropped_side] * 2)

            center = int(cropped_side / 2)
            log_tag = tag + '/{:.1f}'.format(i * self.vis_max_angle / self.vis_rotations)
            self.logger.experiment.add_image(log_tag, cropped[:, :, center].unsqueeze(0), self.global_step)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        #self.log('validation_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        for metric, f in self.metrics.items():
            self.log(metric, f(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)
        self.log_prediction_image()

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
