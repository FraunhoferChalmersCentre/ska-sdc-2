import numpy as np
from astropy.io.fits import getheader, getdata
from astropy.wcs import WCS
from hyperopt import hp, fmin, tpe
import pandas as pd
from sofia import readoptions
from sparse import COO

from definitions import config, ROOT_DIR
from pipeline.common import filename
from pipeline.hyperparameter.tuning import Tuner

size = config['segmentation']['size']
reduction = config['segmentation']['validation']['reduction']
checkpoint = config['traversing']['checkpoint']

validation_set = filename.processed.validation_dataset(size, 100 * reduction)
hyperparam_set = filename.processed.hyperopt_dataset(size, 100 * reduction, checkpoint)
name = f'{checkpoint}/{size}/{reduction}/'

header = getheader(hyperparam_set + '/clipped_input.fits', ignore_blank=True)
input_cube = getdata(hyperparam_set + '/clipped_input.fits', ignore_blank=True).astype(np.float32)
model_out = getdata(hyperparam_set + '/output.fits', ignore_blank=True).astype(np.float32)
segmap = COO.from_numpy(np.load(validation_set + '/segmentmap.npz')['arr_0'].astype(np.float32))
df = pd.read_csv(validation_set + '/df.txt', sep=' ', index_col='id')

sofia_params = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])

tuner = Tuner(config['hyperparameters']['threshold'], sofia_params, input_cube, header, model_out, segmap, df,
              name=name)

space = {'radius_spatial': hp.uniform('radius_spatial', .5, 5),
         'radius_freq': hp.uniform('radius_freq', .5, 100),
         'min_size_spatial': hp.uniform('min_size_spatial', .5, 5),
         'min_size_freq': hp.uniform('min_size_freq', 10, 50),
         'max_size_spatial': hp.uniform('max_size_spatial', 5, 30),
         'max_size_freq': hp.uniform('max_size_freq', 50, 300),
         'min_voxels': hp.uniform('min_voxels', 1, 300),
         'dilation_max_spatial': hp.uniform('dilation_max_spatial', .5, 5),
         'dilation_max_freq': hp.uniform('dilation_max_freq', .5, 20),
         'mask_threshold': hp.uniform('mask_threshold', 1e-2, 1),
         'min_intensity': hp.uniform('min_intensity', 0, 30),
         'max_intensity': hp.uniform('max_intensity', 200, 1000)
         }

init_values = [{'radius_spatial': sofia_params['merge']['radiusX'],
                'radius_freq': sofia_params['merge']['radiusZ'],
                'min_size_spatial': sofia_params['merge']['minSizeX'],
                'min_size_freq': sofia_params['merge']['minSizeZ'],
                'max_size_spatial': sofia_params['merge']['maxSizeX'],
                'max_size_freq': sofia_params['merge']['maxSizeZ'],
                'min_voxels': sofia_params['merge']['minVoxels'],
                'dilation_max_spatial': sofia_params['parameters']['dilatePixMax'],
                'dilation_max_freq': sofia_params['parameters']['dilateChanMax'],
                'mask_threshold': config['hyperparameters']['threshold'],
                'min_intensity': config['hyperparameters']['min_intensity'],
                'max_intensity': config['hyperparameters']['max_intensity']
                }]

best = fmin(tuner.tuning_objective, space, algo=tpe.suggest, max_evals=10000, points_to_evaluate=init_values)

print(best)
