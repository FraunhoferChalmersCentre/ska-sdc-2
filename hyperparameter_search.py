import glob

import torch
from astropy.io.fits import getheader
from hyperopt import hp, fmin, tpe

from definitions import config
from pipeline.data.ska_dataset import SKADataSet, ValidationItemGetter
from pipeline.common import filename
from pipeline.hyperparameter.validation import HyperoptSegmenter
from pipeline.hyperparameter.tuning import Tuner

size = config['segmentation']['size']
modelname = config['segmentation']['model_name']

directory = filename.processed.hyperopt_dataset(size, modelname)
file = glob.glob(directory + '/*.pt')[0]
dataset = torch.load(file)
validation_set = SKADataSet(dataset, ValidationItemGetter(), empty_keys=['position', 'model_out'])

segmenter = HyperoptSegmenter(validation_set, getheader(filename.data.sky(config['segmentation']['size'])))
segmenter.eval()

tuner = Tuner(segmenter)

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
