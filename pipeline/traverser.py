from abc import ABC, abstractmethod
from pipeline.segmenter import BaseSegmenter
from pipeline.downstream import extract_sources
from astropy.io import fits
import pandas as pd
from utils.clip import partition_overlap, partition_expanding, cube_evaluation, connect_outputs
import numpy as np
import torch


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
                 sofia_padding, gpu_memory: float, n_parallel: int = 1, i: int = 0):
        super().__init__(model)
        self.model_input_dim = model_input_dim
        self.desired_dim = desired_dim
        self.cnn_padding = cnn_padding
        self.sofia_padding = sofia_padding
        self.n_parallel = n_parallel
        self.i = i
        self.gpu_memory = gpu_memory
        header = fits.getheader(fits_file, ignore_blank=True)
        self.cube_shape = np.array(list(map(lambda x: header[x], ['NAXIS1', 'NAXIS2', 'NAXIS3'])))
        self.bunit = header['bunit']
        self.data_cache = CubeCache(fits_file)
        slices_partition = partition_expanding(self.cube_shape, desired_dim, cnn_padding+sofia_padding)
        self.slices_partition = np.array_split(slices_partition[0], self.n_parallel)[self.i]

    def __len__(self):
        return len(self.slices_partition)

    def traverse(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for slices in self.slices_partition:
            self.data_cache.cache_data(slices)
            hi_cube_tensor = self.data_cache.get_hi_data()
            overlap_slices_partition, overlaps_partition = partition_overlap(self.desired_dim, self.model_input_dim,
                                                                             self.cnn_padding, self.gpu_memory)
            outputs = list()
            efficient_slices = list()
            for i, overlap_slices in enumerate(overlap_slices_partition):
                position = np.array([[sos.start, sos.stop]for sos in overlap_slices]).T
                o, e = cube_evaluation(hi_cube_tensor, self.model_input_dim, self.cnn_padding, position, overlap_slices,
                                       overlaps_partition[i], self.model)
                outputs += o
                efficient_slices += o
            mask = connect_outputs(hi_cube_tensor, outputs, efficient_slices, self.cnn_padding)

            # Convert to numpy for Sofia
            mask = mask.numpy().T
            hi_cube = hi_cube_tensor.numpy().T
            prediction = extract_sources(mask, hi_cube, self.bunit)

            # Filter Desired Characteristics and Non-edge-padding
            # TODO
            df = remove_non_edge_padding(slices, self.cube_shape, self.cnn_padding, self.sofia_padding, df)

            # Concatenate DataFrames
            df = df.append(prediction)

        # Fix Index
        df.index = np.arange(df.shape[0])
        return df


class CubeCache:

    def __init__(self, fits_file):
        self.fits_file = fits_file
        self.hi_cube_tensor = None

    def get_hi_data(self):
        return self.hi_cube_tensor

    def cache_data(self, slices):
        hi_data_fits = fits.getdata(self.fits_file, ignore_blank=True)
        f0, f1 = slices[-1].start, slices[-1].stop
        self.hi_cube_tensor = torch.tensor(hi_data_fits[f0:f1].astype(np.float32), dtype=torch.float32).T[slices[:2]]


def remove_non_edge_padding(slices, cube_shape, cnn_padding, sofia_padding, df):
    rpositions = np.round(df[['x_geo', 'y_geo', 'z_geo']].values)
    l_padd = np.array([0 if s.start == 0 else sp for s, c, sp in zip(slices, cube_shape, sofia_padding)])
    print(l_padd)
    u_padd = np.array([c if s.stop == c else s.stop - s.start - sp - 2*cp for s, sp, cp, c in
                       zip(slices, sofia_padding, cnn_padding, cube_shape)])
    print(u_padd)
    df_filter = np.all(np.logical_and(rpositions >= l_padd, rpositions <= u_padd), axis=1)
    return df[df_filter]

