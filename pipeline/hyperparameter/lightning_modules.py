import pytorch_lightning as pl
import torch
from astropy.io.fits import Header
from sofia import readoptions
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from definitions import config, ROOT_DIR
from pipeline.segmentation.base import BaseSegmenter
from pipeline.segmentation.training import SortedSampler
from pipeline.segmentation.clip import partition_overlap, cube_evaluation, connect_outputs

from pipeline.downstream import parametrise_sources
from pipeline.data.ska_dataset import AbstractSKADataset
from pipeline.segmentation.scoring import score_source


class ValidationOutputSaveSegmenter(BaseSegmenter):
    def __init__(self, base: BaseSegmenter, validation_set: AbstractSKADataset, header: Header):
        super().__init__(base.model, base.scale, base.mean, base.std)
        self.header = header
        self.validation_set = validation_set
        self.sofia_parameters = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])

        self.sofia_parameters['merge']['radiusX'] = 1
        self.sofia_parameters['merge']['radiusY'] = 1
        self.sofia_parameters['merge']['radiusZ'] = 1
        self.sofia_parameters['merge']['minSizeX'] = 1
        self.sofia_parameters['merge']['minSizeY'] = 1
        self.sofia_parameters['merge']['minSizeZ'] = 1
        self.sofia_parameters['merge']['minVoxels'] = 1
        self.sofia_parameters['parameters']['dilatePixMax'] = 0
        self.sofia_parameters['parameters']['dilateChanMax'] = 0

    def validation_step(self, batch, batch_idx):
        # Compute padding
        dim = (np.array(self.validation_set.get_attribute('dim')) * 2).astype(np.int32)
        padding = (dim / 4).astype(np.int32)

        overlap_slices_partition, overlaps_partition = partition_overlap(torch.squeeze(batch['image']).shape, dim,
                                                                         padding)

        outputs, efficient_slices = cube_evaluation(torch.squeeze(batch['image']), dim, padding,
                                                    torch.squeeze(batch['position']), overlap_slices_partition[0],
                                                    overlaps_partition[0], self)

        model_out = connect_outputs(torch.squeeze(batch['image']), outputs, efficient_slices, padding)
        model_out = model_out

        if batch_idx < self.validation_set.get_attribute('index'):
            clipped_input = batch['image'][0, 0][[slice(p, - p) for p in padding]]
            clipped_segmap = batch['segmentmap'][0, 0][[slice(p, - p) for p in padding]]
            segmap_sources = parametrise_sources(self.header, clipped_input.T, clipped_segmap.T, batch['position'],
                                                 self.sofia_parameters, padding)
            n_segmap_sources = torch.tensor(max(len(segmap_sources), 1)).view(-1)
        else:
            n_segmap_sources = torch.tensor(0).view(-1)
        return model_out.view(1, *model_out.shape).cpu(), n_segmap_sources

    def validation_epoch_end(self, outputs) -> None:
        model_out = [p[0] for p in outputs]
        n_sources = [p[1] for p in outputs]
        self.validation_set.add_attribute({'model_out': model_out}, ['model_out'], ['model_out'])
        self.validation_set.add_attribute({'n_sources': n_sources}, ['n_sources'], ['n_sources'])

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=1, shuffle=False)


class HyperoptSegmenter(pl.LightningModule):
    def __init__(self, validation_set: AbstractSKADataset, header: Header, sofia_parameters=None):

        super().__init__()

        self.header = header
        self.validation_set = validation_set
        self.sofia_precision = pl.metrics.Precision(num_classes=1, multiclass=False)
        self.sofia_recall = pl.metrics.Recall(num_classes=1, multiclass=False)
        self.sofia_dice = pl.metrics.F1(num_classes=1, multiclass=False)

        self.sofia_metrics = {
            'precision': self.sofia_precision,
            'recall': self.sofia_recall,
            'dice': self.sofia_dice
        }

        if sofia_parameters is None:
            self.sofia_parameters = readoptions.readPipelineOptions(
                ROOT_DIR + config['downstream']['sofia']['param_file'])
        else:
            self.sofia_parameters = sofia_parameters

    def validation_step(self, batch, batch_idx):
        dim = (np.array(self.validation_set.get_attribute('dim')) * 2).astype(np.int32)
        padding = (dim / 4).astype(np.int32)

        mask = torch.round(nn.Sigmoid()(batch['model_out']) + .5 - self.threshold)

        has_source = batch_idx < self.validation_set.get_attribute('index')
        clipped_input = torch.empty(batch['model_out'].shape, device=self.device)
        clipped_input[0, 0] = batch['image'][0, 0][[slice(p, - p) for p in padding]]

        parametrized_df = parametrise_sources(self.header, clipped_input.T, mask.T, batch['position'],
                                              self.sofia_parameters, padding)
        sources_found = len(parametrized_df) > 0

        has_source, sources_found = tuple(
            map(lambda t: torch.tensor(t, device=self.device).view(-1), (has_source, sources_found)))

        points = 0

        if has_source and sources_found:
            n_matched, scores, predictions = score_source(self.header, batch, parametrized_df)

            self.log('score_n_matches', n_matched, on_step=True, on_epoch=True)

            if n_matched > 0:
                self.log('n_found', torch.ones(1), on_step=False, on_epoch=True, reduce_fx=torch.sum,
                         tbptt_reduce_fx=torch.sum)

                for k, v in scores.items():
                    self.log('score_' + k, v, on_step=True, on_epoch=True)

                points = np.mean([scores[k] for k in scores.keys()])
                self.log('score_total', points, on_step=True, on_epoch=True)

                for metric, f in self.sofia_metrics.items():
                    f(torch.tensor(True).view(-1), torch.tensor(True).view(-1))
                    self.log('sofia_{}'.format(metric), f, on_epoch=True)

            else:
                for metric, f in self.sofia_metrics.items():
                    f(torch.tensor(False).view(-1), torch.tensor(True).view(-1))
                    self.log('sofia_{}'.format(metric), f, on_epoch=True)

        clipped_segmap = batch['segmentmap'][0, 0][[slice(p, - p) for p in padding]]
        penalty = 0
        for i, row in parametrized_df.iterrows():
            if clipped_segmap[int(row.x), int(row.y), int(row.z)] == 0:
                penalty -= config['hyperparameters']['fp_penalty']

        if has_source and not sources_found:
            for metric, f in self.sofia_metrics.items():
                f(torch.tensor(False).view(-1), torch.tensor(True).view(-1))
                self.log('sofia_{}'.format(metric), f, on_epoch=True)

        if not has_source and not sources_found:
            for metric, f in self.sofia_metrics.items():
                f(torch.tensor(False).view(-1), torch.tensor(False).view(-1))
                self.log('sofia_{}'.format(metric), f, on_epoch=True)

        self.log('point', torch.tensor(points), on_step=True, on_epoch=True, reduce_fx=torch.sum,
                 tbptt_reduce_fx=torch.sum)
        self.log('penalty', torch.tensor(penalty), on_step=True, on_epoch=True, reduce_fx=torch.sum,
                 tbptt_reduce_fx=torch.sum)
        self.log('adjusted_point', torch.tensor(points + penalty), on_step=True, on_epoch=True, reduce_fx=torch.sum,
                 tbptt_reduce_fx=torch.sum)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=1, sampler=SortedSampler(self.validation_set))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
