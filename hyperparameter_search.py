import glob

import numpy as np
import torch
from astropy.io.fits import getheader, getdata
from astropy.wcs import WCS
from hyperopt import hp, fmin, tpe
import pandas as pd
from sofia import readoptions
from sparse import load_npz, COO

from definitions import config, ROOT_DIR
from pipeline.data.ska_dataset import SKADataSet, ValidationItemGetter
from pipeline.common import filename
from pipeline.hyperparameter.lightning_modules import HyperoptSegmenter
from pipeline.hyperparameter.tuning import Tuner

size = config['segmentation']['size']
modelname = config['segmentation']['model_name']

directory = filename.processed.hyperopt_dataset(size, modelname)

df = pd.read_csv(filename.data.true(size), sep=' ')
header = getheader(directory + '/cube.fits')
cube_in = getdata(directory + '/cube.fits').astype(np.float32)
model_out = getdata(directory + '/modelout.fits').astype(np.float32)
segmap = np.load(directory + '/segmap.npz')['arr_0'].astype(np.float32)

wcs = WCS(header)
df[['x', 'y', 'z']] = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)
for i, p in enumerate(['z', 'y', 'x']):
    df = df[(df[p] > 0) & (df[p] < cube_in.shape[i])]

sofia_params = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])

tuner = Tuner(.5, sofia_params, cube_in, header, model_out, segmap, df)

del cube_in, model_out

space = {'radius_spatial': hp.uniform('radius_spatial', .5, 5),
         'radius_freq': hp.uniform('radius_freq', .5, 100),
         'min_size_spatial': hp.uniform('min_size_spatial', .5, 5),
         'min_size_freq': hp.uniform('min_size_freq', 10, 50),
         'min_voxels': hp.uniform('min_voxels', 1, 300),
         'dilation_max_spatial': hp.uniform('dilation_max_spatial', .5, 5),
         'dilation_max_freq': hp.uniform('dilation_max_freq', .5, 20),
         'mask_threshold': hp.uniform('mask_threshold', 1e-2, 1),
         'min_intensity': hp.uniform('min_intensity', 0, 120)
         }

init_values = [{'radius_spatial': 2,
                'radius_freq': 59,
                'min_size_spatial': 5,
                'min_size_freq': 16,
                'min_voxels': 231,
                'dilation_max_spatial': 5,
                'dilation_max_freq': 5,
                'mask_threshold': .87,
                'min_intensity': 24.04
                }]

best = fmin(tuner.tuning_objective, space, algo=tpe.suggest, max_evals=10000, points_to_evaluate=init_values)

print(best)
