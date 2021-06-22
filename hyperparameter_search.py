import glob

import numpy as np
import torch
from astropy.io.fits import getheader, getdata
from hyperopt import hp, fmin, tpe
import pandas as pd
from sofia import readoptions
from sparse import load_npz

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
cube_in = getdata(directory + '/cube.fits')
model_out = getdata(directory + '/modelout.fits')
segmap = np.load(directory + '/segmap.npz')['arr_0']

sofia_params = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])

tuner = Tuner(.5, sofia_params, cube_in, header, model_out, segmap, df)

del cube_in, model_out

space = {'radius_spatial': hp.uniform('radius_spatial', .5, 10),
         'radius_freq': hp.uniform('radius_freq', .5, 50),
         'min_size_spatial': hp.uniform('min_size_spatial', .5, 10),
         'min_size_freq': hp.uniform('min_size_freq', 10, 100),
         'min_voxels': hp.uniform('min_voxels', 1, 150),
         'dilation_max_spatial': hp.uniform('dilation_max_spatial', .5, 20),
         'dilation_max_freq': hp.uniform('dilation_max_freq', .5, 20),
         'mask_threshold': hp.uniform('mask_threshold', .1, 1),
         }

best = fmin(tuner.tuning_objective, space, algo=tpe.suggest, max_evals=1000)

print(best)
