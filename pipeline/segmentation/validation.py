from typing import Dict
import glob

import pandas as pd
import sparse
import torch
from astropy.io.fits import getdata, Header
from astropy.wcs import WCS
from torchmetrics import Metric
import numpy as np

from pipeline.segmentation.base import AbstractValidator
from pipeline.segmentation.scoring import score_df
from pipeline.segmentation.training import BaseSegmenter
from pipeline.traversing.traverser import EvaluationTraverser


class SimpleValidator(AbstractValidator):
    def __init__(self, segmenter: BaseSegmenter, surrogates: Dict[str, Metric]):
        super().__init__(segmenter)
        self.surrogates = surrogates

    def on_validation_start(self):
        for name, surrogate in self.surrogates.items():
            surrogate.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['segmentmap']
        f_channels = torch.empty((x.shape[0], 2), device=x.device)
        for i in range(x.shape[0]):
            f_channels[i, 0] = batch['position'][i, 0, -1] + batch['slices'][i][0][-1]
            f_channels[i, 1] = batch['position'][i, 0, -1] + batch['slices'][i][1][-1]
        self.segmenter.to(x.device)
        y_hat = self.segmenter(x, f_channels)
        y = y.to(y_hat.device)

        for name, surrogate in self.surrogates.items():
            surrogate.update(y_hat, y)
        return

    def validation_epoch_end(self, validation_step_outputs):
        return {k: v.compute() for k, v in self.surrogates.items()}


class SKAScoreValidator(AbstractValidator):
    def __init__(self, segmenter: BaseSegmenter, evaluator: EvaluationTraverser, df: pd.DataFrame,
                 segmentmap: sparse.COO, header: Header):
        super().__init__(segmenter)
        self.header = header
        self.segmentmap = segmentmap
        self.df = df
        self.evaluator = evaluator
        self.metrics = {}

    def validation_step(self, batch, batch_idx):
        self.evaluator.model = self.segmenter
        df_predicted = self.evaluator.traverse()

        wcs = WCS(self.header)

        if len(df_predicted) > 0:
            df_predicted[['x_geo', 'y_geo', 'z_geo']] = wcs.all_world2pix(df_predicted[['ra', 'dec', 'central_freq']],
                                                                          0)
        self.metrics, _ = score_df(df_predicted, self.df, self.segmentmap)

    def validation_epoch_end(self, validation_step_outputs):
        return self.metrics

    def on_validation_start(self):
        return


class FullValidationSetValidator(AbstractValidator):
    def __init__(self, segmenter: BaseSegmenter, val_dataset_path: str, evaluator: EvaluationTraverser,
                 surrogates: Dict[str, Metric]):
        super().__init__(segmenter)
        self.surrogates = surrogates
        self.evaluator = evaluator
        self.val_dataset_path = val_dataset_path
        self.segmentmap = sparse.load_npz(f'{val_dataset_path}/segmentmap.npz')

    def on_validation_start(self):
        for name, surrogate in self.surrogates.items():
            surrogate.reset()

    def validation_step(self, batch, batch_idx):
        _ = self.evaluator.traverse(save_output=True, output_path=self.val_dataset_path)
        n_outputs = len(glob.glob(f'{self.val_dataset_path}/model_out/*.fits'))
        for i in range(n_outputs):
            model_out = torch.tensor(getdata(f'{self.val_dataset_path}/model_out/{i}.fits').astype(np.float32),
                                     dtype=torch.float32).T.reshape(-1)
            p = torch.load(f'{self.val_dataset_path}/partition_position/{i}.pb')
            partition_segmentmap = torch.tensor(
                self.segmentmap[p[0, 0]:p[1, 0], p[0, 1]:p[1, 1], p[0, 2]:p[1, 2]].todense().astype(
                    np.float32)).reshape(-1)
            partition_segmentmap = torch.where(partition_segmentmap >= 1, 1., 0.)
            for name, surrogate in self.surrogates.items():
                surrogate.update(model_out, partition_segmentmap)

        return

    def validation_epoch_end(self, validation_step_outputs):
        return {k: v.compute() for k, v in self.surrogates.items()}
