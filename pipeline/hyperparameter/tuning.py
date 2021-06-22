from datetime import datetime

import numpy
import numpy as np
import torch
from astropy.io.fits import Header
from sparse import COO
from torch.utils.tensorboard import SummaryWriter

from pipeline.hyperparameter.scoring import score_df

iteration = 0


class Tuner:
    def __init__(self, threshold: float, sofia_parameters: dict, input_cube: np.ndarray, header: Header,
                 model_out: np.ndarray, segmap: np.ndarray, df):

        self.header = header
        self.df = df
        self.segmap = segmap
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

            writer = SummaryWriter(log_dir='hparam_logs/' + version)

            self.threshold = args['mask_threshold']
            self.sofia_parameters['merge']['radiusX'] = int(np.round(args['radius_spatial']))
            self.sofia_parameters['merge']['radiusY'] = int(np.round(args['radius_spatial']))
            self.sofia_parameters['merge']['radiusZ'] = int(np.round(args['radius_freq']))
            self.sofia_parameters['merge']['minSizeX'] = int(np.round(args['min_size_spatial']))
            self.sofia_parameters['merge']['minSizeY'] = int(np.round(args['min_size_spatial']))
            self.sofia_parameters['merge']['minSizeZ'] = int(np.round(args['min_size_freq']))
            self.sofia_parameters['merge']['minVoxels'] = int(np.round(args['min_voxels']))
            self.sofia_parameters['parameters']['dilatePixMax'] = int(np.round(args['dilation_max_spatial']))
            self.sofia_parameters['parameters']['dilateChanMax'] = int(np.round(args['dilation_max_freq']))

            score = score_df(self.input_cube, self.header, self.model_out, self.df, self.segmap, self.sofia_parameters,
                             self.threshold, writer)


            for k, v in args.items():
                writer.add_scalar('hparams/' + k, v)

            writer.flush()

            return - score
        except Exception as err:
            print(err)
            return float('inf')
