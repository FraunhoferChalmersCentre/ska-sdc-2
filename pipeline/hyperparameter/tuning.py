import pickle
from datetime import datetime

import glob

import multiprocessing

import numpy as np
import pandas as pd
import sparse
import torch
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS
from hyperopt import STATUS_OK, STATUS_FAIL
from sparse import COO
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from definitions import config
from pipeline.downstream import parametrise_sources
from pipeline.hyperparameter.timeout import timeout
from pipeline.segmentation.scoring import score_df
from pipeline.traversing.traverser import remove_non_edge_padding


def scale_value(value, interval=None):
    if interval:
        return (value - interval[0]) / (interval[1] - interval[0])
    else:
        return value


class AbstractTuner:
    def __init__(self, threshold: float, min_intensity: float, max_intensity: float, sofia_parameters: dict,
                 segmentmap: COO, df_true: pd.DataFrame, name: str = None):

        self.df_true = df_true
        self.segmentmap = segmentmap
        self.max_intensity = max_intensity
        self.min_intensity = min_intensity
        self.name = name
        self.sofia_parameters = sofia_parameters
        self.threshold = threshold
        self.iteration = 0

    def score_strategy(self, score_metrics) -> float:
        raise NotImplementedError

    def create_catalogue(self) -> pd.DataFrame:
        raise NotImplementedError

    def update_args(self, args):
        self.threshold = args['mask_threshold']
        self.min_intensity = args['min_intensity']
        self.max_intensity = args['max_intensity']
        self.sofia_parameters['merge']['radiusX'] = int(np.round(args['radius_spatial']))
        self.sofia_parameters['merge']['radiusY'] = int(np.round(args['radius_spatial']))
        self.sofia_parameters['merge']['radiusZ'] = int(np.round(args['radius_freq']))
        self.sofia_parameters['merge']['minSizeX'] = int(np.round(args['min_size_spatial']))
        self.sofia_parameters['merge']['minSizeY'] = int(np.round(args['min_size_spatial']))
        self.sofia_parameters['merge']['minSizeZ'] = int(np.round(args['min_size_freq']))
        self.sofia_parameters['merge']['maxSizeX'] = int(np.round(args['max_size_spatial']))
        self.sofia_parameters['merge']['maxSizeY'] = int(np.round(args['max_size_spatial']))
        self.sofia_parameters['merge']['maxSizeZ'] = int(np.round(args['max_size_freq']))
        self.sofia_parameters['merge']['minVoxels'] = int(np.round(args['min_voxels']))
        self.sofia_parameters['parameters']['dilatePixMax'] = int(np.round(args['dilation_max_spatial']))
        self.sofia_parameters['parameters']['dilateChanMax'] = int(np.round(args['dilation_max_freq']))

    @timeout(config['hyperparameters']['catalogue_generation_timelimit'])
    def generate_single_cube_catalogue(self, input_cube: torch.tensor, header: Header, model_out: torch.tensor,
                                       sofia_params: dict, mask_threshold: float, min_intensity: float,
                                       max_intensity: float, position=None):

        mask = torch.round(nn.Sigmoid()(model_out.to(torch.float32)) + 0.5 - mask_threshold).to(torch.float32)
        mask[mask > 1] = 1

        if position is None:
            position = torch.zeros(2, 3)
            position[1] = torch.tensor(input_cube.shape)
        df_predicted = parametrise_sources(header, input_cube, mask, position, sofia_params,
                                           min_intensity=min_intensity,
                                           max_intensity=max_intensity)

        return df_predicted

    def produce_score(self, args):
        self.iteration += 1
        try:
            version = str(self.iteration) + datetime.now().strftime("_%Y%m%d_%H%M%S")

            writer = SummaryWriter(log_dir='hparam_logs/' + self.name + '/' + version)

            for k, v in args.items():
                writer.add_scalar('hparams/' + k, v)

            self.update_args(args)

            df_predicted = self.create_catalogue()

            metrics, df_predicted = score_df(df_predicted, self.df_true, self.segmentmap)

            for k, v in metrics.items():
                writer.add_scalar(k, v)

            writer.flush()

            return {'loss': - self.score_strategy(metrics), 'status': STATUS_OK,
                    'sofia_params': self.sofia_parameters,
                    'df': df_predicted.to_dict(), **metrics}
        except Exception as err:
            print('ERROR', err)
            return {'status': STATUS_FAIL, 'sofia_params': self.sofia_parameters}


class SingleInputTuner(AbstractTuner):
    def __init__(self, threshold: float, min_intensity: float, max_intensity: float, sofia_parameters: dict,
                 segmentmap: COO, df_true: pd.DataFrame, input_cube: np.ndarray, header: Header,
                 model_out: np.ndarray,
                 name=None):
        super().__init__(threshold, min_intensity, max_intensity, sofia_parameters, segmentmap, df_true, name)
        self.header = header
        self.df_true = df_true
        self.segmentmap = segmentmap
        self.input_cube = torch.tensor(input_cube.astype(np.float32), dtype=torch.float32)
        self.model_out = torch.tensor(model_out.astype(np.float32), dtype=torch.float32)

    def create_catalogue(self) -> pd.DataFrame:
        return self.generate_single_cube_catalogue(self.input_cube, self.header, self.model_out,
                                                   self.sofia_parameters, self.threshold,
                                                   self.min_intensity, self.max_intensity)


class MultiInputTuner(AbstractTuner):
    def __init__(self, threshold: float, min_intensity: float, max_intensity: float, sofia_parameters: dict,
                 test_set_path: str, header: Header, cnn_padding: np.ndarray, sofia_padding: np.ndarray, name=None):
        segmentmap = sparse.load_npz(f'{test_set_path}/segmentmap.npz')
        df_true = pd.read_csv(f'{test_set_path}/df.txt', sep=' ', index_col='id')
        super().__init__(threshold, min_intensity, max_intensity, sofia_parameters, segmentmap, df_true, name)
        self.test_set_path = test_set_path
        self.header = header
        self.cube_shape = np.array(list(map(lambda x: header[x], ['NAXIS1', 'NAXIS2', 'NAXIS3'])))
        self.cnn_padding = cnn_padding
        self.sofia_padding = sofia_padding

    def create_catalogue(self) -> pd.DataFrame:
        n_model_outs = len(glob.glob(f'{self.test_set_path}/model_out/*.fits'))
        catalogues = []
        for i in tqdm(range(n_model_outs)):
            input_cube = torch.tensor(
                fits.getdata(f'{self.test_set_path}/clipped_input/{i}.fits').astype(np.float32),
                dtype=torch.float32)
            model_out = torch.tensor(fits.getdata(f'{self.test_set_path}/model_out/{i}.fits').astype(np.float32),
                                     dtype=torch.float32)
            position = torch.load(f'{self.test_set_path}/partition_position/{i}.pb')
            slices = pickle.load(open(f'{self.test_set_path}/slices/{i}.pb', 'rb'))

            df = self.generate_single_cube_catalogue(input_cube, self.header, model_out, self.sofia_parameters,
                                                     self.threshold, self.min_intensity, self.max_intensity,
                                                     position)

            df = remove_non_edge_padding(slices, self.cube_shape, self.cnn_padding, self.sofia_padding, df)
            catalogues.append(df)
            del input_cube, model_out

        catalogues = [c for c in catalogues if len(c) > 0]
        if len(catalogues) > 0:
            merged_catalogue = pd.concat(catalogues)

            wcs = WCS(self.header)
            merged_catalogue[['x_geo', 'y_geo', 'z_geo']] = wcs.all_world2pix(
                merged_catalogue[['ra', 'dec', 'central_freq']], 0)

            return merged_catalogue
        return pd.DataFrame()


class SKAScoreTuner(MultiInputTuner):
    def __init__(self, threshold: float, min_intensity: float, max_intensity: float, sofia_parameters: dict,
                 test_set_path: str, header: Header, cnn_padding: np.ndarray, sofia_padding: np.ndarray, name=None):
        super().__init__(threshold, min_intensity, max_intensity, sofia_parameters, test_set_path, header,
                         cnn_padding,
                         sofia_padding, name)

    def score_strategy(self, score_metrics):
        return score_metrics['sdc2_score']


class PrecisionRecallTradeoffTuner(MultiInputTuner):
    def __init__(self, alpha, threshold: float, min_intensity: float, max_intensity: float, sofia_parameters: dict,
                 test_set_path: str, header: Header, cnn_padding: np.ndarray, sofia_padding: np.ndarray, name=None):
        super().__init__(threshold, min_intensity, max_intensity, sofia_parameters, test_set_path, header,
                         cnn_padding,
                         sofia_padding, name)
        self.alpha = alpha

    def score_strategy(self, score_metrics):
        p = scale_value(score_metrics['precision'])
        r = scale_value(score_metrics['recall'])
        return self.alpha * p + (1 - self.alpha) * r
