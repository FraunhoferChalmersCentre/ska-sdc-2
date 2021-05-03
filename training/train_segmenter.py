from typing import Any, List
from itertools import starmap

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchvision.transforms.functional as TF
import numpy as np
from pytorch_toolbelt import losses
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from pipeline.segmenter import BaseSegmenter
from utils.clip import partition_overlap, cube_evaluation, connect_outputs

from pipeline.downstream import extract_sources, parametrise_sources
from utils.data.ska_dataset import AbstractSKADataset
from utils.scoring import score_source, parametrisation_validation, ANGLE_SCATTER_ATTRS, LINEAR_SCATTER_ATTRS


class SortedSampler(Sampler):
    def __init__(self, dataset: AbstractSKADataset):
        allocations = list(starmap(lambda i, t: (i, len(t)), enumerate(dataset.get_attribute('allocated_voxels')[:-1])))
        self.sorted_indices = list(map(lambda i: i[0], sorted(allocations, key=lambda a: a[1], reverse=True))) + list(
            range(dataset.get_attribute('index'), len(dataset)))

    def __len__(self):
        return len(self.sorted_indices)

    def __iter__(self):
        return iter(self.sorted_indices)


def get_random_vis_id(dataset, shape, min_allocated, random_state=np.random.RandomState()):
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


class TrainSegmenter(BaseSegmenter):
    def __init__(self, base: BaseSegmenter, loss_fct: Any, training_set: AbstractSKADataset,
                 validation_set: AbstractSKADataset, header: fits.Header, batch_size=128, x_key='image',
                 y_key='segmentmap', vis_max_angle=180, vis_rotations=4, vis_id=None, threshold=None, lr=1e-2,
                 momentum=.9):
        super().__init__(base.model, base.scale, base.mean, base.std)
        self.header = header
        self.batch_size = batch_size
        self.validation_set = validation_set
        self.training_set = training_set
        self.y_key = y_key
        self.x_key = x_key
        self.loss_fct = loss_fct
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
        self.log_prediction_image()

    def log_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
        slices = tuple(starmap(lambda s, d: slice(int(s / 2 - d), int(s / 2 - d) + 2 * d),
                               zip(image.shape, self.validation_set.get_attribute('dim'))))
        normed_img = (image[slices] - image[slices].min()) / (image[slices].max() - image[slices].min())
        self._log_cross_sections(normed_img, self.validation_set[self.vis_id]['pa'], self.x_key)
        segmap = self.validation_set.get_attribute(self.y_key)[self.vis_id].squeeze()
        if segmap.sum() == 0:
            raise ValueError('Logged segmentmap contains no source voxels. Reshuffle!')

        self._log_cross_sections(segmap[slices], self.validation_set[self.vis_id]['pa'], self.y_key)

    def log_prediction_image(self):
        image = self.validation_set.get_attribute(self.x_key)[self.vis_id].squeeze()
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
        x, y = batch[self.x_key], batch[self.y_key]

        f_channels = torch.empty((x.shape[0], 2), device=self.device)
        for i in range(x.shape[0]):
            f_channels[i, 0] = batch['position'][i, 0, -1] + batch['slices'][i][0][-1]
            f_channels[i, 1] = batch['position'][i, 0, -1] + batch['slices'][i][1][-1]

        y_hat = self(x, f_channels)
        loss = self.loss_fct(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # IMPORTANT: Single batch assumed when validating

        # Compute padding
        dim = (np.array(self.validation_set.get_attribute('dim'))*2).astype(np.int32)
        padding = (dim / 4).astype(np.int32)

        overlap_slices_partition, overlaps_partition = partition_overlap(torch.squeeze(batch['image']).shape, dim,
                                                                         padding)

        outputs, efficient_slices = cube_evaluation(torch.squeeze(batch['image']), dim, padding,
                                                    torch.squeeze(batch['position']), overlap_slices_partition[0],
                                                    overlaps_partition[0], self)

        model_out = connect_outputs(torch.squeeze(batch['image']), outputs, efficient_slices, padding)
        model_out = model_out.view(1, 1, *model_out.shape).to(self.device)
        mask = torch.round(nn.Sigmoid()(model_out))

        clipped_segmap = torch.empty(model_out.shape, device=self.device)
        clipped_segmap[0, 0] = batch['segmentmap'][0, 0][[slice(p, - p) for p in padding]]

        for surrogate, f in self.surrogates.items():
            self.log(surrogate, f(model_out, clipped_segmap.float()), on_step=True, on_epoch=True)

        for metric, f in self.pixel_metrics.items():
            f(mask.int().view(-1), clipped_segmap.int().view(-1))
            self.log('pixel_{}'.format(metric), f, on_epoch=True)

        has_source = batch_idx < self.validation_set.get_attribute('index')
        clipped_input = torch.empty(model_out.shape, device=self.device)
        clipped_input[0, 0] = batch['image'][0, 0][[slice(p, - p) for p in padding]]

        parametrized_df = parametrise_sources(self.header, clipped_input, mask, batch['position'], padding)
        sofia_out = parametrisation_validation(self.header, parametrized_df, batch, has_source)

        has_source, sofia_out = tuple(
            map(lambda t: torch.tensor(t, device=self.device).view(-1), (has_source, sofia_out)))

        for metric, f in self.sofia_metrics.items():
            f(sofia_out, has_source)
            self.log('sofia_{}'.format(metric), f, on_epoch=True)

        if has_source and sofia_out:
            n_matched, scores, predictions = score_source(self.header, batch, parametrized_df)

            self.log('score_n_matches', n_matched, on_step=True, on_epoch=True)

            for k, v in scores.items():
                self.log('score_' + k, v, on_step=True, on_epoch=True)

            self.log('score_total', np.mean([scores[k] for k in scores.keys()]), on_step=True, on_epoch=True)

            return predictions

        return None

    def validation_epoch_start(self):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        if len(validation_step_outputs) == 0:
            return
        matched_outputs = {k: [v[k] for v in validation_step_outputs] for k in validation_step_outputs[0].keys()}

        for k, v in matched_outputs.items():
            if len(v) > 0:
                pred_arr = np.array(v)
                fig = plt.figure()

                if k in ANGLE_SCATTER_ATTRS:
                    pred_arr = np.deg2rad(pred_arr)
                    plt.scatter(np.cos(pred_arr[:, 0] - pred_arr[:, 1]), np.sin(pred_arr[:, 0] - pred_arr[:, 1]), c='r',
                                alpha=.1)

                    v = np.linspace(0, 2 * np.pi)
                    plt.plot(np.cos(v), np.sin(v), alpha=.1)
                elif k in LINEAR_SCATTER_ATTRS:
                    plt.scatter(pred_arr[:, 0], pred_arr[:, 1])
                    plt.xlabel('Prediction')
                    plt.ylabel('True value')
                    l = [pred_arr[:, 0].min(), pred_arr[:, 0].max()]
                    plt.plot(l, l, 'r')

                plt.gca().set_aspect('equal', adjustable='box')
                self.logger.experiment.add_figure('Scatter/' + k, fig, self.global_step)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log_prediction_image()

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=1, sampler=SortedSampler(self.validation_set))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
