from abc import ABC, abstractmethod
import os

from astropy.io.fits import getheader

from pipeline.common import filename
from pipeline.common.filename import prepare_dir
from pipeline.data.segmentmap import create_from_df, prepare_df
from pipeline.segmentation.base import BaseSegmenter
from torch import nn
from pipeline.downstream import parametrise_sources
from definitions import config
from astropy.io import fits
import pandas as pd
from pipeline.segmentation.clip import partition_overlap, partition_expanding, cube_evaluation, connect_outputs
import numpy as np
import pickle
import torch
from tqdm import tqdm
from spectral_cube import SpectralCube
from astropy.wcs import WCS


class ModelTraverser(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def traverse(self, *args, **kwargs):
        pass


class SimpleModelTraverser(ModelTraverser):

    def __init__(self, model: BaseSegmenter, cube: torch.tensor, padding, input_dim):
        super().__init__(model)
        self.cube = cube
        self.padding = padding
        self.input_dim = input_dim

    def traverse(self, index) -> dict:
        pass


class EvaluationTraverser(ModelTraverser):

    def __init__(self, model: BaseSegmenter, fits_file: str, model_input_dim, desired_dim, cnn_padding,
                 sofia_padding, max_batch_size: int, n_parallel: int = 1, i_job: int = 0, j_loop: int = 0,
                 df_name: str = None):
        super().__init__(model)
        self.model_input_dim = model_input_dim
        self.desired_dim = desired_dim
        self.cnn_padding = cnn_padding
        self.sofia_padding = sofia_padding
        self.n_parallel = n_parallel
        self.i_job = i_job
        self.j_loop = j_loop
        self.max_batch_size = max_batch_size
        header = fits.getheader(fits_file, ignore_blank=True)
        self.cube_shape = np.array(list(map(lambda x: header[x], ['NAXIS1', 'NAXIS2', 'NAXIS3'])))
        self.header = header
        self.data_cache = CubeCache(fits_file)
        slices_partition = partition_expanding(self.cube_shape, desired_dim + 2 * cnn_padding,
                                               cnn_padding + sofia_padding)
        self.slices_partition = np.array_split(slices_partition[0], self.n_parallel)[self.i_job]
        if df_name is None:
            df_name = ''
        self.df_name = df_name + '_n_parallel' + str(n_parallel) + '_i_job' + str(i_job) + '.txt'

    def __len__(self):
        return len(self.slices_partition)

    def traverse(self, save_output=False, save_input=False, remove_cols=True, output_path=None) -> pd.DataFrame:
        if self.j_loop > 0:
            df = pd.read_csv(self.df_name)
        else:
            df = pd.DataFrame()

        for j, slices in enumerate(self.slices_partition):
            print('Loop {} of {}'.format(str(j), str(len(self.slices_partition))))
            if j >= self.j_loop:
                self.data_cache.cache_data(slices)
                hi_cube_tensor = self.data_cache.get_hi_data()
                position = np.array([[s.start, s.stop] for s in slices]).T
                overlap_slices_partition, overlaps_partition = partition_overlap(position[1] - position[0],
                                                                                 self.model_input_dim,
                                                                                 self.cnn_padding, self.max_batch_size)
                outputs = list()
                efficient_slices = list()
                for overlap_slices, overlaps in tqdm(zip(overlap_slices_partition, overlaps_partition),
                                                     total=len(overlap_slices_partition), desc='Propagating model'):
                    try:
                        o, e = cube_evaluation(hi_cube_tensor, self.model_input_dim, self.cnn_padding, position,
                                               overlap_slices, overlaps, self.model)
                    except:
                        pickle.dump({'j_loop': j, 'n_parallel': self.n_parallel, 'i_job': self.i_job},
                                    open("j_loop.p", "wb"))
                        raise ValueError('Memory Issue')
                    outputs += o
                    efficient_slices += e
                mask = connect_outputs(hi_cube_tensor, outputs, efficient_slices, self.cnn_padding)
                del outputs

                inner_slices = [slice(p, -p) for p in self.cnn_padding]
                hi_cube_tensor = hi_cube_tensor[inner_slices]

                partition_position = torch.tensor([[s.start + p for s, p in zip(slices, self.cnn_padding)],
                                                   [s.stop - p for s, p in zip(slices, self.cnn_padding)]])

                if save_output:
                    inner_slices.reverse()
                    wcs = WCS(self.header)[inner_slices]

                    model_out_fits = SpectralCube(mask.T.cpu().numpy(), wcs, header=self.header)
                    prepare_dir(f'{output_path}/model_out')
                    if os.path.isfile(f'{output_path}/model_out/{j}.fits'):
                        os.remove(f'{output_path}/model_out/{j}.fits')
                    model_out_fits.write(f'{output_path}/model_out/{j}.fits', format='fits')

                    del model_out_fits

                if save_input:
                    partial_input_fits = SpectralCube(hi_cube_tensor.T.cpu().numpy(), wcs, header=self.header)
                    prepare_dir(f'{output_path}/clipped_input')
                    if os.path.isfile(f'{output_path}/clipped_input/{j}.fits'):
                        os.remove(f'{output_path}/clipped_input/{j}.fits')
                    partial_input_fits.write(f'{output_path}/clipped_input/{j}.fits', format='fits')

                    del partial_input_fits

                if save_output or save_input:

                    prepare_dir(f'{output_path}/partition_position')
                    if os.path.isfile(f'{output_path}/partition_position/{j}.pb'):
                        os.remove(f'{output_path}/partition_position/{j}.pb')
                    torch.save(partition_position, f'{output_path}/partition_position/{j}.pb')

                    continue

                mask = torch.round(nn.Sigmoid()(mask) + 0.5 - config['hyperparameters']['threshold'])
                mask[mask > 1] = 1

                prediction = parametrise_sources(self.header, hi_cube_tensor.T, mask.T, partition_position,
                                                 min_intensity=config['hyperparameters']['min_intensity'],
                                                 max_intensity=config['hyperparameters']['max_intensity'])

                del hi_cube_tensor, mask

                # Filter Desired Characteristics and Non-edge-padding
                if len(prediction) > 0:
                    prediction = remove_non_edge_padding(slices, self.cube_shape, self.cnn_padding, self.sofia_padding,
                                                         prediction)
                    if remove_cols:
                        prediction = prediction[config['characteristic_parameters']]

                    # Concatenate DataFrames
                    df = df.append(prediction)

                    # Save DataFrame
                    df.index = np.arange(df.shape[0])
                    df.to_csv(self.df_name, sep=' ', index_label='id')

        # Fix Index
        df.index = np.arange(df.shape[0])
        df.to_csv(self.df_name, sep=' ', index_label='id')
        return df


class CubeCache:

    def __init__(self, fits_file: str, gradual_loading: bool = True):
        self.fits_file = fits_file
        self.hi_cube_tensor = None
        self.gradual_loading = gradual_loading

    def set_gradual_loading(self, value: bool):
        self.gradual_loading = value

    def comp_statistics(self, channels, percentiles=None):
        if percentiles is None:
            percentiles = [.1, 99.9]

        scale = list()
        mean = list()
        std = list()

        for channel in channels:
            hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
            comp_percentiles = np.percentile(hi_data_fits[channel], percentiles)
            clipped = np.clip(hi_data_fits[channel], *comp_percentiles)
            clipped = (clipped - comp_percentiles[0]) / (comp_percentiles[1] - comp_percentiles[0])

            scale.append(torch.tensor(comp_percentiles, dtype=torch.float32))
            mean.append(torch.tensor(clipped.mean(), dtype=torch.float32))
            std.append(torch.tensor(clipped.std(), dtype=torch.float32))

        return scale, mean, std

    def get_hi_data(self):
        return self.hi_cube_tensor

    def cache_data(self, slices):
        if self.gradual_loading:
            f0, f1 = slices[-1].start, slices[-1].stop
            self.hi_cube_tensor = torch.empty(tuple(map(lambda x: x.stop - x.start, slices)))
            for i, f in enumerate(range(f0, f1)):
                hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
                self.hi_cube_tensor[:, :, i] = torch.tensor(hi_data_fits[f].astype(np.float32),
                                                            dtype=torch.float32).T[slices[:2]]
        else:
            hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
            f0, f1 = slices[-1].start, slices[-1].stop
            self.hi_cube_tensor = torch.tensor(hi_data_fits[f0:f1].astype(np.float32), dtype=torch.float32).T[
                slices[:2]]


def remove_non_edge_padding(slices, cube_shape, cnn_padding, sofia_padding, df):
    rpositions = np.round(df[['x_geo', 'y_geo', 'z_geo']].values)
    l_padd = np.array([0 if s.start == 0 else sp for s, c, sp in zip(slices, cube_shape, sofia_padding)])
    u_padd = np.array([c if s.stop == c else s.stop - s.start - sp - 2 * cp for s, sp, cp, c in
                       zip(slices, sofia_padding, cnn_padding, cube_shape)])
    df_filter = np.all(np.logical_and(rpositions >= l_padd, rpositions <= u_padd), axis=1)
    return df[df_filter]
