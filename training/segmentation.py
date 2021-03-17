from typing import Any
from itertools import starmap

import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms.functional as TF
import numpy as np
from pytorch_toolbelt import losses
import matplotlib.pyplot as plt

from utils.data.ska_dataset import AbstractSKADataset

class SortedSampler(Sampler):
    def __init__(self, dataset: AbstractSKADataset):
        allocations = list(starmap(lambda i, t: (i, len(t)), enumerate(dataset.get_attribute('allocated_voxels')[:-1])))
        self.sorted_indices = list(range(dataset.get_attribute('index'), len(dataset))) +  list(map(lambda i: i[0], sorted(allocations, key=lambda a: a[1])))
        
    
    def __len__(self):
        return len(self.sorted_indices)
    
    def __iter__(self):
        return iter(self.sorted_indices)


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

def batch_by_allocation(dataset: AbstractSKADataset, batches: int):
    allocations = list(map(len, dataset.get_attribute('allocated_pixels')))
    

class Segmenter(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fct: Any, training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset, batch_size=128, x_key='image',
                 y_key='segmentmap', vis_max_angle=180, vis_rotations=4, vis_id=None, threshold = None, lr=1e-1, momentum=.9):
        super().__init__()
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.training_set = training_set
        self.y_key = y_key
        self.x_key = x_key
        self.loss_fct = loss_fct
        self.model = model
        self.vis_rotations = vis_rotations
        self.vis_max_angle = vis_max_angle
        self.lr = lr
        self.momentum = momentum

        self.vis_id = vis_id
        self.threshold = threshold
        
        
        
        self.metrics = {
            'Soft dice': losses.DiceLoss(mode='binary'),
            'Soft Jaccard': losses.JaccardLoss(mode='binary'),
            'Lovasz hinge': losses.BinaryLovaszLoss(),
            'Cross Entropy': losses.SoftBCEWithLogitsLoss(),
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn,
            'TN': self.tn
        }
        
        self.derivatives = {
            'Jaccard': self.jaccard,
            'Dice': self.dice,
            'Specificity': self.specificity,
            'Precision': self.precision_metric,
            'Sensitivity': self.sensitivity
        }
    
    def on_fit_start(self):
        print('fit start')
        self.log_image()
        
        self.reset_metrics()
        
        
    
    def rounded(self, prediction, truth):
        prediction = torch.where(nn.Sigmoid()(prediction).flatten() > self.threshold, 1, 0)
        truth = truth.flatten().round()
        return prediction, truth
    
    def reset_metrics(self):
        self.tp_sum = torch.tensor(0.).to(self.device)
        self.fp_sum = torch.tensor(0.).to(self.device)
        self.fn_sum = torch.tensor(0.).to(self.device)
        self.tn_sum = torch.tensor(0.).to(self.device)
    
    def port_metrics(self):
        self.tp_sum = self.tp_sum.to(self.device)
        self.fp_sum = self.fp_sum.to(self.device)
        self.fn_sum = self.fn_sum.to(self.device)
        self.tn_sum = self.tn_sum.to(self.device)
    
    def tp(self, prediction, truth):
        prediction, truth = self.rounded(prediction, truth)
        current_tp = (prediction * truth).sum()
        self.tp_sum += current_tp
        return current_tp

    def fp(self, prediction, truth):
        prediction, truth = self.rounded(prediction, truth)
        current_fp = torch.sum(torch.logical_xor(prediction, truth) * prediction)
        self.fp_sum += current_fp
        return current_fp

    def fn(self, prediction, truth):
        prediction, truth = self.rounded(prediction, truth)
        current_fn = torch.sum(torch.logical_xor(prediction, truth) * torch.logical_not(prediction))
        self.fn_sum += current_fn
        return current_fn

    def tn(self, prediction, truth):
        prediction, truth = self.rounded(prediction, truth)
        current_tn = torch.sum(torch.logical_not(prediction) * torch.logical_not(truth))
        self.tn_sum += current_tn
        return current_tn
        
    def jaccard(self):
        denom = self.tp_sum + self.fp_sum + self.fn_sum
        if torch.eq(denom, 0):
            return torch.tensor(0)
        return self.tp_sum / denom

    def dice(self):
        denom = 2 * self.tp_sum + self.fp_sum + self.fn_sum
        if torch.eq(denom, 0):
            return torch.tensor(0)
        return 2 * self.tp_sum / denom
    
    def sensitivity(self):
        denom = self.tp_sum + self.fn_sum
        if torch.eq(denom, 0):
            return torch.tensor(0)
        return self.tp_sum / denom 
    
    def specificity(self):
        denom = self.tn_sum + self.fp_sum
        if torch.eq(denom, 0):
            return torch.tensor(0)
        return self.tn_sum / denom 
    
    def precision_metric(self):
        denom = self.tp_sum + self.fp_sum
        if torch.eq(denom, 0):
            return torch.tensor(0)
        return self.tp_sum / denom 
        
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
            rotated = TF.rotate(cube.squeeze().T, float(i * self.vis_max_angle / self.vis_rotations - pa + 90))

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
        #self.port_metrics()
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        for metric, f in self.metrics.items():
            self.log(metric, f(y_hat, y), on_step=True, on_epoch=True, sync_dist=True)
        return self.loss_fct(y_hat, y)
        
         
    def validation_epoch_end(self, outputs):
        for metric, f in self.derivatives.items():
            self.log(metric, f(), on_step=False, on_epoch=True, sync_dist=True)
        self.log_prediction_image()
        self.reset_metrics()
        

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, sampler=SortedSampler(self.validation_set))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer
