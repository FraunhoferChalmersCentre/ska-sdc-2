from datetime import datetime

import numpy as np
import pandas as pd
import torch
from astropy.io.fits import Header
from sparse import COO
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from pipeline.downstream import parametrise_sources
from pipeline.segmentation.scoring import score_df

iteration = 0


def create_predicted_catalogue(input_cube: torch.tensor, header: Header, model_out: torch.tensor, sofia_params: dict,
                               mask_threshold: float, min_intensity: float, max_intensity: float):
    mask = torch.round(nn.Sigmoid()(model_out.to(torch.float32)) + 0.5 - mask_threshold).to(torch.float32)
    mask[mask > 1] = 1

    position = torch.zeros(2, 3)
    position[1] = torch.tensor(input_cube.shape)
    df_predicted = parametrise_sources(header, input_cube, mask, position, sofia_params, min_intensity=min_intensity,
                                       max_intensity=max_intensity)

    return df_predicted


class Tuner:
    def __init__(self, threshold: float, sofia_parameters: dict, input_cube: np.ndarray, header: Header,
                 model_out: np.ndarray, segmap: COO, df: pd.DataFrame, name=None):

        self.name = name
        self.header = header
        self.df_true = df
        self.segmentmap = segmap
        self.input_cube = torch.tensor(input_cube.astype(np.float32), dtype=torch.float32)
        self.model_out = torch.tensor(model_out.astype(np.float32), dtype=torch.float32)
        self.sofia_parameters = sofia_parameters
        self.threshold = threshold
        self.iteration = 0

    def tuning_objective(self, args):
        self.iteration += 1
        try:
            with open("notebooks/hyperparams.txt", "a+") as file_object:
                file_object.write(str(self.iteration) + '\t' + str(args) + '\n')

            version = str(self.iteration) + datetime.now().strftime("_%Y%m%d_%H%M%S")

            writer = SummaryWriter(log_dir='hparam_logs/' + self.name + '/' + version)

            self.threshold = args['mask_threshold']
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

            for k, v in args.items():
                writer.add_scalar('hparams/' + k, v)

            df_predicted = create_predicted_catalogue(self.input_cube, self.header, self.model_out,
                                                      self.sofia_parameters, self.threshold,
                                                      args['min_intensity'], args['max_intensity'])

            metrics = score_df(df_predicted, self.df_true, self.segmentmap.todense(), sub_directory=self.name)

            for k, v in metrics.items():
                writer.add_scalar(k, v)

            writer.flush()

            return - metrics['score']
        except Exception as err:
            print(err)
            return float('inf')
