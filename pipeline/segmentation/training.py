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
from pipeline.segmentation.clip import partition_overlap, cube_evaluation, connect_outputs

from pipeline.downstream import parametrise_sources
from pipeline.data.ska_dataset import AbstractSKADataset
from pipeline.segmentation.scoring import score_source, ANGLE_SCATTER_ATTRS, LINEAR_SCATTER_ATTRS


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
    def __init__(self, index, n_total, batch_size, bootstrap=False, intensities=None, n_samples=None, random_seed=None):
        # half batch size
        self.random_seed = random_seed
        self.bootstrap = bootstrap
        self.intensities = intensities
        self.hbs = int(batch_size / 2)
        self.sources = np.arange(index)
        self.empty = np.arange(index, n_total)
        self.n_total = n_total
        self.n_samples = n_samples

    def __len__(self):
        return self.n_total if not self.n_samples else self.n_samples

    def __iter__(self):
        random_generator = np.random.RandomState(self.random_seed)
        if self.bootstrap:
            source_intensities = self.intensities[:len(self.sources)]
            source_intensities = source_intensities / source_intensities.sum()
            n_source_samples = len(self.sources) if not self.n_samples else int(
                self.n_samples * len(self.sources) / self.n_total)
            source_samples = random_generator.choice(self.sources, n_source_samples, replace=True, p=source_intensities)

            empty_intensities = self.intensities[len(self.sources):]
            empty_intensities = empty_intensities / empty_intensities.sum()
            n_empty_samples = len(self.empty) if not self.n_samples else int(
                self.n_samples * len(self.empty) / self.n_total)
            empty_samples = random_generator.choice(self.empty, n_empty_samples, replace=True, p=empty_intensities)
        else:
            source_samples = self.sources
            empty_samples = self.empty

        source_samples = random_generator.permutation(source_samples)
        empty_samples = random_generator.permutation(empty_samples)

        n_batches = int(np.ceil(len(source_samples) / self.hbs))
        batched_indices = []

        for i in range(n_batches):
            batch = []
            batch.extend(source_samples[i * self.hbs:(i + 1) * self.hbs])
            batch.extend(empty_samples[i * self.hbs:(i + 1) * self.hbs])
            batched_indices.extend(random_generator.permutation(batch))

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
                 name=None):
        super().__init__(base.model, base.scale, base.mean, base.std)

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
        image = self.validation_set.get_attribute('image')[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.validation_set.get_attribute('dim'))))
        normed_img = (image[slices] - image[slices].min()) / (image[slices].max() - image[slices].min())
        self._log_cross_sections(normed_img, self.validation_set[self.vis_id]['pa'], 'image')
        segmap = self.validation_set.get_attribute('segmentmap')[self.vis_id].squeeze()
        if segmap.sum() == 0:
            raise ValueError('Logged segmentmap contains no source voxels. Reshuffle!')

        self._log_cross_sections(segmap[slices], self.validation_set[self.vis_id]['pa'], 'segmentmap')

    def log_prediction_image(self):
        image = self.validation_set.get_attribute('image')[self.vis_id].squeeze()
        position = self.validation_set.get_attribute('position')[self.vis_id]
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.validation_set.get_attribute('dim'))))
        input_image = image[slices].to(self.device).view(1, 1, *image[slices].shape)
        f_channels = torch.tensor([[position[0, -1] + slices[-1].start, position[0, -1] + slices[-1].stop]])
        prediction = nn.Sigmoid()(self(input_image, f_channels)).squeeze()
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
        x, y = batch['image'], batch['segmentmap']

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
            return self.validation_step_robust(batch, batch_idx)
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

    def validation_step_robust(self, batch, batch_idx):
        # IMPORTANT: Single batch assumed when validating

        # Compute padding
        dim = (np.array(self.validation_set.get_attribute('dim')) * 2).astype(np.int32)
        padding = (dim / 4).astype(np.int32)

        overlap_slices_partition, overlaps_partition = partition_overlap(torch.squeeze(batch['image']).shape, dim,
                                                                         padding)

        outputs, efficient_slices = cube_evaluation(torch.squeeze(batch['image']), dim, padding,
                                                    torch.squeeze(batch['position']), overlap_slices_partition[0],
                                                    overlaps_partition[0], self)

        model_out = connect_outputs(torch.squeeze(batch['image']), outputs, efficient_slices, padding)
        model_out = model_out.view(1, 1, *model_out.shape).to(self.device)
        mask = torch.round(nn.Sigmoid()(model_out) + .5 - self.threshold)

        clipped_segmap = torch.empty(model_out.shape, device=self.device)
        clipped_segmap[0, 0] = batch['segmentmap'][0, 0][[slice(p, - p) for p in padding]]

        for metric, f in self.pixel_metrics.items():
            f(mask.int().view(-1), clipped_segmap.int().view(-1))
            self.log('pixel_{}'.format(metric), f, on_epoch=True)

        has_source = batch_idx < self.validation_set.get_attribute('index')
        clipped_input = torch.empty(model_out.shape, device=self.device)
        clipped_input[0, 0] = batch['image'][0, 0][[slice(p, - p) for p in padding]]

        parametrized_df = parametrise_sources(self.header, clipped_input.T, mask.T, batch['position'],
                                              self.sofia_parameters, padding)
        any_sources_found = len(parametrized_df) > 0

        has_source, any_sources_found = tuple(
            map(lambda t: torch.tensor(t, device=self.device).view(-1), (has_source, any_sources_found)))

        points = 0
        predictions = None
        source_found = torch.tensor(False, device=self.device).view(-1)

        if has_source and any_sources_found:
            n_matched, scores, predictions = score_source(self.header, batch, parametrized_df)

            self.log('score_n_matches', n_matched, on_step=True, on_epoch=True)

            if n_matched > 0:
                source_found = torch.tensor(True, device=self.device).view(-1)
                self.log('n_found', torch.ones(1), on_step=False, on_epoch=True, reduce_fx=torch.sum,
                         tbptt_reduce_fx=torch.sum)

                for k, v in scores.items():
                    self.log('score_' + k, v, on_step=True, on_epoch=True)

                points = np.mean([scores[k] for k in scores.keys()])
                self.log('score_total', points, on_step=True, on_epoch=True)
            else:
                points = 0

            points -= (len(parametrized_df) - n_matched)

        if not has_source and any_sources_found:
            points = -len(parametrized_df)
            source_found = torch.tensor(True, device=self.device).view(-1)

        self.log('point', torch.tensor(points), on_step=True, on_epoch=True, reduce_fx=torch.sum,
                 tbptt_reduce_fx=torch.sum)

        for metric, f in self.sofia_metrics.items():
            f(source_found, has_source)
            self.log('sofia_{}'.format(metric), f, on_epoch=True)

        if self.dataset_surrogates:
            return model_out, clipped_segmap

        return predictions

    def validation_epoch_start(self):
        pass

    def validation_epoch_end(self, validation_step_outputs):

        if len(validation_step_outputs) == 0:
            return

        if self.dataset_surrogates:
            model_outs = torch.cat(tuple([p[0].view(-1) for p in validation_step_outputs])).view(1, 1, -1)
            segmaps = torch.cat(tuple([p[1].view(-1) for p in validation_step_outputs])).view(1, 1, -1)
            for surrogate, f in self.surrogates.items():
                self.log(surrogate, f(model_outs, segmaps), on_epoch=True)

            return

        if self.robust_validation and config['segmentation']['save_plots']:
            predictions = {k + '_pred': [v[k][0] for v in validation_step_outputs] for k in
                           validation_step_outputs[0].keys()}
            true_values = {k + '_true': [v[k][1] for v in validation_step_outputs] for k in
                           validation_step_outputs[0].keys()}
            df = pd.DataFrame({**predictions, **true_values})
            csv_name = 'result_{}.csv'.format(self.name) if self.name else datetime.now().strftime("%m%d%Y_%H%M%S")
            df.to_csv(csv_name)

            for k in validation_step_outputs[0].keys():
                pred_arr = df[[k + '_pred', k + '_true']]
                fig = plt.figure()

                if k in ANGLE_SCATTER_ATTRS:
                    pred_arr = np.deg2rad(pred_arr)
                    plt.scatter(np.cos(pred_arr[k + '_pred'] - pred_arr[k + '_true']),
                                np.sin(pred_arr[k + '_pred'] - pred_arr[k + '_true']), c='r', alpha=.05)

                    v = np.linspace(0, 2 * np.pi)
                    plt.plot(np.cos(v), np.sin(v), alpha=.01)
                elif k in LINEAR_SCATTER_ATTRS:
                    plt.scatter(df[k + '_pred'], df[k + '_true'], alpha=.05)
                    plt.xlabel('Prediction')
                    plt.ylabel('True value')
                    l = [df[k + '_pred'].min(), df[k + '_pred'].max()]
                    plt.plot(l, l, 'r')

                plt.gca().set_aspect('equal', adjustable='box')
                self.logger.experiment.add_figure('Scatter/' + k, fig, self.global_step)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log_prediction_image()

    def train_dataloader(self):
        if self.train_sampler is not None:
            return DataLoader(self.training_set, batch_size=self.batch_size, sampler=self.train_sampler, shuffle=False)
        else:
            return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.robust_validation:
            return DataLoader(self.validation_set, batch_size=1, shuffle=False,
                              sampler=SortedSampler(self.validation_set))
        else:
            if self.val_sampler is not None:
                return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                                  sampler=self.val_sampler)
            else:
                return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False)

    def configure_optimizers(self):
        return self.optimizer
