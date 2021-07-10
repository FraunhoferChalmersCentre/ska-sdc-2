from datetime import datetime
from typing import Any, List
from itertools import starmap

import pandas as pd
import pytorch_lightning as pl
import torch
from sofia import readoptions
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms.functional as TF
import numpy as np
from pytorch_toolbelt import losses
from astropy.io import fits
import matplotlib.pyplot as plt

from definitions import config, ROOT_DIR
from pipeline.segmentation.base import BaseSegmenter

from pipeline.data.ska_dataset import AbstractSKADataset
from pipeline.segmentation.scoring import score_df


class SortedSampler(Sampler):
    def __init__(self, dataset: AbstractSKADataset):
        allocations = list(starmap(lambda i, t: (i, len(t)), enumerate(dataset.get_attribute('allocated_voxels')[:-1])))
        self.sorted_indices = list(map(lambda i: i[0], sorted(allocations, key=lambda a: a[1], reverse=True))) + list(
            range(dataset.get_attribute('index'), len(dataset)))

    def __len__(self):
        return len(self.sorted_indices)

    def __iter__(self):
        return iter(self.sorted_indices)


class EquiBatchBootstrapSampler(Sampler):
    def __init__(self, index, n_total, source_batch_size, noise_batch_size, source_bs_start=None, intensities=None,
                 n_samples=None, random_seed=None, anneal_interval=2, batch_size_noise=.15):

        self.epochs = 0
        self.anneal_interval = anneal_interval
        self.random_seed = random_seed
        self.intensities = intensities
        self.source_bs_end = source_batch_size
        self.noise_bs_end = noise_batch_size
        self.batch_size = source_batch_size + noise_batch_size
        self.max_batch_size_noise = np.round(batch_size_noise * self.batch_size)

        if source_bs_start:
            self.current_source_bs = source_bs_start
            self.current_noise_bs = (self.source_bs_end + self.noise_bs_end) - self.current_source_bs
        else:
            self.current_source_bs = self.source_bs_end
            self.current_noise_bs = self.noise_bs_end

        self.sources = np.arange(index)
        self.empty = np.arange(index, n_total)
        self.n_total = n_total
        self.n_samples = n_samples

    def __len__(self):
        return self.n_total if not self.n_samples else self.n_samples

    def update_batch_size(self):
        if self.current_source_bs != self.source_bs_end and self.epochs % self.anneal_interval == 0:
            self.current_source_bs -= 1
            self.current_noise_bs += 1

    def get_batch_sizes(self, random_generator):
        if self.max_batch_size_noise > 0:
            min_noise = int(min(np.round(self.max_batch_size_noise / 2), self.current_source_bs,
                                self.batch_size - self.current_noise_bs))
            max_noise = int(min(np.round(self.max_batch_size_noise / 2), self.batch_size - self.current_source_bs,
                                self.current_noise_bs))
            noise = random_generator.randint(-min_noise, max_noise + 1)
            print(self.current_source_bs, self.current_noise_bs, min_noise, max_noise, noise)
            return self.current_source_bs + noise, self.current_noise_bs - noise
        else:
            return self.current_source_bs, self.current_noise_bs

    def __iter__(self):
        random_generator = np.random.RandomState(self.random_seed)

        source_bs, noise_bs = self.get_batch_sizes(random_generator)

        source_intensities = self.intensities[:len(self.sources)]
        source_intensities = source_intensities / source_intensities.sum()
        n_source_samples = len(self.sources) if not self.n_samples else int(
            self.n_samples * (source_bs / self.batch_size))
        source_samples = random_generator.choice(self.sources, n_source_samples, replace=True, p=source_intensities)

        empty_intensities = self.intensities[len(self.sources):]
        empty_intensities = empty_intensities / empty_intensities.sum()
        n_empty_samples = len(self.empty) if not self.n_samples else int(
            self.n_samples * (noise_bs / self.batch_size))
        empty_samples = random_generator.choice(self.empty, n_empty_samples, replace=True, p=empty_intensities)

        source_samples = random_generator.permutation(source_samples)
        empty_samples = random_generator.permutation(empty_samples)

        n_batches = int(np.ceil(len(source_samples) + len(empty_samples) / (source_bs + noise_bs)))
        batched_indices = []

        for i in range(n_batches):
            batch = []
            batch.extend(source_samples[i * source_bs:(i + 1) * source_bs])
            batch.extend(empty_samples[i * noise_bs:(i + 1) * noise_bs])
            batched_indices.extend(random_generator.permutation(batch))

        self.epochs += 1
        self.update_batch_size()

        return iter(batched_indices)


class TrainSegmenter(BaseSegmenter):
    def __init__(self, base: BaseSegmenter,
                 loss_fct: Any,
                 training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset,
                 header: fits.Header,
                 optimizer: torch.optim.Optimizer,
                 batch_size=128,
                 robust_validation=False,
                 vis_max_angle=180,
                 vis_rotations=4,
                 vis_id=None,
                 threshold=None,
                 sofia_parameters=None,
                 dataset_surrogates=True,
                 train_sampler=None,
                 val_sampler=None,
                 train_padding=0,
                 random_rotation=True,
                 random_mirror=True,
                 name=None,
                 check_val_every_n_epoch=1):
        super().__init__(base.model, base.scale, base.mean, base.std)

        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.robust_validation = robust_validation
        self.name = name
        self.tr_pad = train_padding
        self.header = header
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.training_set = training_set
        self.loss_fct = loss_fct
        self.vis_rotations = vis_rotations
        self.vis_max_angle = vis_max_angle
        self.optimizer = optimizer

        self.vis_id = vis_id
        self.threshold = threshold

        self.pixel_precision = pl.metrics.Precision(num_classes=1, multiclass=False)
        self.pixel_recall = pl.metrics.Recall(num_classes=1, multiclass=False)
        self.pixel_dice = pl.metrics.F1(num_classes=1, multiclass=False)

        self.sofia_precision = pl.metrics.Precision(num_classes=1, multiclass=False)
        self.sofia_recall = pl.metrics.Recall(num_classes=1, multiclass=False)
        self.sofia_dice = pl.metrics.F1(num_classes=1, multiclass=False)

        self.pixel_metrics = {
            'precision': self.pixel_precision,
            'recall': self.pixel_recall,
            'dice': self.pixel_dice
        }

        self.sofia_metrics = {
            'precision': self.sofia_precision,
            'recall': self.sofia_recall,
            'dice': self.sofia_dice
        }

        self.dice = losses.DiceLoss(mode='binary', from_logits=True)
        self.lovasz = losses.BinaryLovaszLoss()
        self.cross_entropy = losses.SoftBCEWithLogitsLoss()

        self.surrogates = {
            'soft_dice': self.dice,
            'lovasz_hinge': self.lovasz,
            'cross_entropy': self.cross_entropy,
            'val_loss': self.loss_fct
        }

        if sofia_parameters is None:
            self.sofia_parameters = readoptions.readPipelineOptions(
                ROOT_DIR + config['downstream']['sofia']['param_file'])
        else:
            self.sofia_parameters = sofia_parameters

        self.dataset_surrogates = dataset_surrogates
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.random_rotation = random_rotation
        self.random_flip = random_mirror

    def on_fit_start(self):
        self.log_image()
        self.log_prediction_image()

    def log_image(self):
        image = self.training_set.get_attribute('image')[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.training_set.get_attribute('dim'))))
        normed_img = (image[slices] - image[slices].min()) / (image[slices].max() - image[slices].min())
        self._log_cross_sections(normed_img, self.training_set[self.vis_id]['pa'], 'image')
        segmap = self.training_set.get_attribute('segmentmap')[self.vis_id].squeeze()
        if segmap.sum() == 0:
            raise ValueError('Logged segmentmap contains no source voxels. Reshuffle!')

        self._log_cross_sections(segmap[slices], self.training_set[self.vis_id]['pa'], 'segmentmap')

    def log_prediction_image(self):
        image = self.training_set.get_attribute('image')[self.vis_id].squeeze()
        position = self.training_set.get_attribute('position')[self.vis_id]
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.training_set.get_attribute('dim'))))
        input_image = image[slices].to(self.device).view(1, 1, *image[slices].shape)
        f_channels = torch.tensor([[position[0, -1] + slices[-1].start, position[0, -1] + slices[-1].stop]])
        prediction = nn.Sigmoid()(self(input_image, f_channels)).squeeze()
        self._log_cross_sections(prediction, self.training_set[self.vis_id]['pa'], 'Prediction')

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
        x, y = batch['image'], batch['segmentmap']

        self.log('galaxy_fraction', torch.sum(y) / torch.prod(torch.tensor(y.shape)), on_epoch=True, on_step=False)

        if self.random_rotation:
            for i in range(len(x)):
                k = np.random.randint(0, 4)
                x[i] = torch.rot90(x[i], k, [1, 2])
                y[i] = torch.rot90(y[i], k, [1, 2])

        if self.random_flip:
            for i in range(len(x)):
                if np.random.randint(2):
                    x[i, 0] = torch.fliplr(x[i, 0])
                    y[i, 0] = torch.fliplr(y[i, 0])

        f_channels = torch.empty((x.shape[0], 2), device=self.device)
        for i in range(x.shape[0]):
            f_channels[i, 0] = batch['position'][i, 0, -1] + batch['slices'][i][0][-1]
            f_channels[i, 1] = batch['position'][i, 0, -1] + batch['slices'][i][1][-1]

        y_hat = self(x, f_channels)
        del f_channels

        if self.tr_pad > 0:
            effective_y_hat = y_hat[:, :, self.tr_pad:-self.tr_pad, self.tr_pad:-self.tr_pad,
                              self.tr_pad:-self.tr_pad].clone()
            del y_hat
            effective_y = y[:, :, self.tr_pad:-self.tr_pad, self.tr_pad:-self.tr_pad, self.tr_pad:-self.tr_pad].clone()
            del y
        else:
            effective_y_hat, effective_y = y_hat, y

        loss = self.loss_fct(effective_y_hat, effective_y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.robust_validation:
            return self.do_robust_validation(batch, batch_idx)
        else:
            return self.validation_step_simple(batch, batch_idx)

    def validation_step_simple(self, batch, batch_idx):
        x, y = batch['image'], batch['segmentmap']
        f_channels = torch.empty((x.shape[0], 2), device=self.device)
        for i in range(x.shape[0]):
            f_channels[i, 0] = batch['position'][i, 0, -1] + batch['slices'][i][0][-1]
            f_channels[i, 1] = batch['position'][i, 0, -1] + batch['slices'][i][1][-1]

        y_hat = self(x, f_channels)

        return y_hat.cpu(), y.cpu()

    def do_robust_validation(self):
        evaluator = self.validation_set['evaluator']
        evaluator.model = self
        df_predicted = evaluator.traverse()

        df_predicted[['x_geo', 'y_geo', 'z_geo']] = self.validation_set['wcs'].all_world2pix(
            df_predicted[['ra', 'dec', 'central_freq']], 0)
        metrics = score_df(df_predicted, self.validation_set['df_true'], self.validation_set['segmentmap'].todense())
        for k, v in metrics.items():
            self.log(k, v)

    def validation_epoch_start(self):
        if self.robust_validation:
            self.do_robust_validation()

    def validation_epoch_end(self, validation_step_outputs):

        if validation_step_outputs is None or len(validation_step_outputs) == 0:
            return

        if self.dataset_surrogates:
            model_outs = torch.cat(tuple([p[0].reshape(-1) for p in validation_step_outputs])).reshape(1, 1, -1)
            segmaps = torch.cat(tuple([p[1].reshape(-1) for p in validation_step_outputs])).reshape(1, 1, -1)
            for surrogate, f in self.surrogates.items():
                self.log(surrogate, f(model_outs, segmaps), on_epoch=True)

            return

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log_prediction_image()

        if self.robust_validation and (self.current_epoch + 1) % self.check_val_every_n_epoch == 0:
            self.do_robust_validation()

    def train_dataloader(self):
        if self.train_sampler is not None:
            return DataLoader(self.training_set, batch_size=self.batch_size, sampler=self.train_sampler, shuffle=False)
        else:
            return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.robust_validation:
            return
        else:
            if self.val_sampler is not None:
                return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                                  sampler=self.val_sampler)
            else:
                return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)

    def configure_optimizers(self):
        return self.optimizer
