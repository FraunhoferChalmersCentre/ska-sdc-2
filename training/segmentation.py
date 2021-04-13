from typing import Any, List
from itertools import starmap

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms.functional as TF
import numpy as np
from pytorch_toolbelt import losses
from astropy.io import fits
from astropy.wcs import WCS

from training.downstream import extract_sources
from utils.data.ska_dataset import AbstractSKADataset

SOFIA_ESTIMATED_ATTRS = {'ra', 'dec', 'line_flux_integral', 'w20', 'HI_size', 'i', 'pa'}

SPEED_OF_LIGHT = 3e5


def final_shapes(dim, upper_left, final_shape):
    final_start = (dim * upper_left).astype(np.int32)
    final_end = (dim * upper_left + dim).astype(np.int32)

    ext_padding = np.zeros(3, dtype=np.int32)

    for i, (e, s) in enumerate(zip(final_end, final_shape)):
        ext_padding[i] = e - min(e, s)
        final_end[i] = min(e, s)

    return final_start, final_end, ext_padding


class SortedSampler(Sampler):
    def __init__(self, dataset: AbstractSKADataset):
        allocations = list(starmap(lambda i, t: (i, len(t)), enumerate(dataset.get_attribute('allocated_voxels')[:-1])))
        self.sorted_indices = list(map(lambda i: i[0], sorted(allocations, key=lambda a: a[1], reverse=True))) + list(
            range(dataset.get_attribute('index'), len(dataset)))

    def __len__(self):
        return len(self.sorted_indices)

    def __iter__(self):
        return iter(self.sorted_indices)


def get_vis_id(dataset, shape, min_allocated, random_state=np.random):
    vis_id = None

    pos = dataset.get_attribute('position')
    alloc = dataset.get_attribute('allocated_voxels')
    while vis_id is None:
        random_id = random_state.randint(0, dataset.get_attribute('index'))
        candidate = True
        for r in pos[random_id].squeeze():

            for c, s in zip(r, shape):
                if torch.eq(c, 0) or torch.eq(c, s):
                    candidate = False

        if candidate and alloc[random_id].shape[0] > min_allocated:
            vis_id = random_id

    return vis_id


def get_metrics():
    return


class Segmenter(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fct: Any, training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset, header: fits.Header, batch_size=128, x_key='image',
                 y_key='segmentmap', vis_max_angle=180, vis_rotations=4, vis_id=None, threshold=None, lr=1e-1,
                 momentum=.9):
        super().__init__()
        self.header = header
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

        self.pixel_precision = pl.metrics.Precision(num_classes=1, is_multiclass=False)
        self.pixel_recall = pl.metrics.Recall(num_classes=1, is_multiclass=False)
        self.pixel_dice = pl.metrics.F1(num_classes=1)

        self.sofia_precision = pl.metrics.Precision(num_classes=1, is_multiclass=False)
        self.sofia_recall = pl.metrics.Recall(num_classes=1, is_multiclass=False)
        self.sofia_dice = pl.metrics.F1(num_classes=1)

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

        self.dice = losses.DiceLoss(mode='binary')
        self.lovasz = losses.BinaryLovaszLoss()
        self.cross_entropy = losses.SoftBCEWithLogitsLoss()

        self.surrogates = {
            'Soft dice': self.dice,
            'Lovasz hinge': self.lovasz,
            'Cross Entropy': self.cross_entropy
        }

    def on_fit_start(self):
        self.log_image()

    def log_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.validation_set.get_attribute('dim'))))
        self._log_cross_sections(image[slices], self.validation_set[self.vis_id]['pa'], self.x_key)
        segmap = self.validation_set.get_attribute(self.y_key)[self.vis_id].squeeze()
        if segmap.sum() == 0:
            raise ValueError('Logged segmentmap contains no source voxels. Reshuffle!')
        self._log_cross_sections(segmap[slices], self.validation_set[self.vis_id]['pa'], self.y_key)

    def log_prediction_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.validation_set.get_attribute('dim'))))
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

    def validation_output(self, validation_cube, dim, padding):
        # Prepare output cubes (without padding in input)
        final_shape = tuple(starmap(lambda s, p: s - 2 * p, zip(validation_cube.shape[2:], padding)))
        final_cube = torch.empty(final_shape, device=self.device)

        # Get number of strides needed in each dimension
        patches_each_dim = tuple(
            starmap(lambda i, f: np.ceil(f / i), zip(dim, final_shape)))
        meshes = np.meshgrid(*map(np.arange, patches_each_dim))
        upper_lefts = tuple(map(np.ravel, meshes))

        model_input = torch.empty(len(upper_lefts[0]), 1, *[d + 2 * p for d, p in zip(dim, padding)],
                                  device=self.device)

        for i, upper_left in enumerate(zip(*upper_lefts)):
            final_start, final_end, ext_padding = final_shapes(dim, upper_left, final_shape)

            slices = tuple(
                starmap(lambda s, e, d, p: slice(s - p, e + d), zip(final_start, final_end, dim, ext_padding)))
            slices = (slice(None), slice(None), *slices)
            model_input[i] = validation_cube[slices]

        model_out = self.model(model_input)

        for i, upper_left in enumerate(zip(*upper_lefts)):
            final_start, final_end, ext_padding = final_shapes(dim, upper_left, final_shape)

            padding_slices = tuple(starmap(lambda p, e: slice(p + e, -p), zip(padding, ext_padding)))
            padding_slices = (i, slice(None), *padding_slices)

            final_slices = tuple(starmap(lambda s, e: slice(s, e), zip(final_start, final_end)))

            final_cube[final_slices] = model_out[padding_slices]

        return final_cube.view(1, 1, *final_cube.shape)

    def parametrise_sources(self, input_cube, mask, position, padding):
        if mask.sum() == 0.:
            return pd.DataFrame()

        input_cube, mask, position = tuple(map(lambda t: t.detach().cpu().numpy(), (input_cube, mask, position)))

        df = extract_sources(input_cube, mask, self.header['bunit'])

        if len(df) > 0:
            # cube_spans = tuple(
            #    [slice(int(pos[0] + pad), int(pos[1] - pad)) for pos, pad in zip(position.T, padding)])
            wcs = WCS(self.header)
            df[['ra', 'dec', 'central_freq']] = wcs.all_pix2world(
                np.array(df[['z_geo', 'y_geo', 'x_geo']] + position[0, 0] + padding, dtype=np.float32), 0)
            df['w20'] = df['w20'] * SPEED_OF_LIGHT * self.header['CDELT3'] / self.header['RESTFREQ']
            # df['f_int'] = df['f_int'] * self.header['CDELT3'] / (np.pi * (7 / 2.8) ** 2 / (4 * np.log(2)))

        return df

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch[self.x_key], batch[self.y_key]
        y_hat = self.model(x)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True)

        return loss

    def parametrisation_validation(self, parametrized_df, batch, has_source):
        if has_source:
            # There is a source in this
            for i, row in parametrized_df.iterrows():
                predicted_pos = np.array([row.ra, row.dec])
                true_pos = np.array([batch['ra'].cpu().numpy(), batch['dec'].cpu().numpy()]).flatten()

                line_width_freq = self.header['RESTFREQ'] * batch['w20'].cpu().numpy() / SPEED_OF_LIGHT
                if np.linalg.norm(predicted_pos - true_pos) * 3600 < batch['hi_size'].cpu().numpy() / 2 and np.abs(
                        batch['central_freq'].cpu().numpy() - row['central_freq']) < line_width_freq / 2:
                    return True
            return False
        else:
            # No sources in this batch
            return len(parametrized_df) > 0

    def validation_step(self, batch, batch_idx):
        # IMPORTANT: Single batch assumed when validating

        # Compute padding
        dim = np.array(self.validation_set.get_attribute('dim'))
        padding = (dim / 2).astype(np.int32)

        model_out = self.validation_output(batch['image'], dim, padding)
        mask = torch.round(nn.Sigmoid()(model_out))

        clipped_segmap = torch.empty(model_out.shape, device=self.device)
        clipped_segmap[0, 0] = batch['segmentmap'][0, 0][[slice(p, - p) for p in padding]]

        for surrogate, f in self.surrogates.items():
            self.log(surrogate, f(mask, clipped_segmap.float()), on_step=True, on_epoch=True)

        for metric, f in self.pixel_metrics.items():
            f(mask.int().view(-1), clipped_segmap.int().view(-1))
            self.log('pixel_{}'.format(metric), f, on_epoch=True)

        has_source = batch_idx < self.validation_set.get_attribute('index')
        clipped_input = torch.empty(model_out.shape, device=self.device).squeeze()
        clipped_input[:, :, :] = batch['image'][0, 0][[slice(p, - p) for p in padding]]

        parametrized_df = self.parametrise_sources(clipped_input, mask.squeeze(), batch['position'], padding)
        sofia_out = self.parametrisation_validation(parametrized_df, batch, has_source)

        has_source, sofia_out = tuple(
            map(lambda t: torch.tensor(t, device=self.device).view(-1), (has_source, sofia_out)))

        for metric, f in self.sofia_metrics.items():
            f(sofia_out, has_source)
            self.log('sofia_{}'.format(metric), f, on_epoch=True)

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log_prediction_image()

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=1, sampler=SortedSampler(self.validation_set))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
