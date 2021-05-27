# !/usr/bin/env python
# coding: utf-8

# Add project root to path

# In[ ]:


import os
import sys

import pandas as pd

from definitions import ROOT_DIR

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Load dataset

# In[ ]:


import torch
from utils import filename
from utils import filehandling
import glob

size = 'dev_s'
prob = 50
modelname = 'resnet18'

directory = filename.processed.hyperopt_dataset(size, prob, modelname)
file = glob.glob(directory + '/*.pt')[0]
dataset = torch.load(file)

# Split to train & test

# In[ ]:
from utils.data.ska_dataset import SKADataSet, ValidationItemGetter

test = SKADataSet(dataset, ValidationItemGetter(), empty_keys=['position', 'model_out'])

# Load pretrained 2D model

# In[ ]:

from pipeline.convert2Dto3D import Conv3dConverter
import segmentation_models_pytorch as smp

modelname = 'resnet18'
model = smp.Unet(encoder_name=modelname, encoder_weights='ssl', in_channels=1, classes=1,
                 decoder_channels=[256, 128, 64, 32], encoder_depth=4, decoder_use_batchnorm=True)
# Convert pretrained 2D model to 3D
Conv3dConverter(model, -1)

# In[ ]:
from astropy.io import fits

from utils.data.generating import get_hi_shape
from utils import filename

hi_shape = get_hi_shape(filename.data.sky(size))
header = fits.getheader(filename.data.sky(size))

# In[ ]:


import pytorch_lightning as pl
from training.hyperopt_segmenter import HyperoptSegmenter

segmenter = HyperoptSegmenter(test, header)
segmenter.eval()
# %%
from hyperopt import hp, fmin, tpe
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.linear_model import LinearRegression

iteration = 0


def objective(args):
    try:
        global iteration, segmenter

        with open("hyperparams.txt", "a+") as file_object:
            file_object.write(str(iteration) + '\t' + str(args) + '\n')

        version = str(iteration) + datetime.now().strftime("_%Y%m%d_%H%M%S")
        name = 'hyperparam' + version
        logger = TensorBoardLogger("tb_logs", name="hyperparam", version=version)
        logger.log_metrics({'hparams/' + k: v for k, v in args.items()})

        segmenter.name = name

        segmenter.threshold = args['mask_threshold']
        # /saved_models/10-epoch=59-sofia_dice=0.27.ckpt
        segmenter.sofia_parameters['merge']['radiusX'] = int(np.round(args['radius_spatial']))
        segmenter.sofia_parameters['merge']['radiusY'] = int(np.round(args['radius_spatial']))
        segmenter.sofia_parameters['merge']['radiusZ'] = int(np.round(args['radius_freq']))
        segmenter.sofia_parameters['merge']['minSizeX'] = int(np.round(args['min_size_spatial']))
        segmenter.sofia_parameters['merge']['minSizeY'] = int(np.round(args['min_size_spatial']))
        segmenter.sofia_parameters['merge']['minSizeZ'] = int(np.round(args['min_size_freq']))
        segmenter.sofia_parameters['merge']['minVoxels'] = int(np.round(args['min_voxels']))
        segmenter.sofia_parameters['parameters']['dilatePixMax'] = int(np.round(args['dilation_max_spatial']))
        segmenter.sofia_parameters['parameters']['dilateChanMax'] = int(np.round(args['dilation_max_freq']))

        trainer = pl.Trainer(gpus=0, logger=logger)

        results = trainer.validate(segmenter)[0]

        points = results['point_epoch']

        """
        results_df = pd.read_csv('result_{}.csv'.format(name), index_col=0)
        os.remove('result_{}.csv'.format(name))
        points -= results['score_hi_size_epoch'] * results['n_found'] / 7
        hi_lr = LinearRegression()
        hi_lr.fit(results_df[['hi_size_pred']], results_df['hi_size_true'])
        predicted = hi_lr.predict(results_df[['hi_size_pred']])
        errors = np.abs(predicted - results_df['hi_size_true']) / results_df['hi_size_true']
        points += np.clip(errors / .3, a_min=0, a_max=1).sum() / 7
        """

        iteration += 1

        return - points
    except Exception as err:
        print(err)
        return float('inf')


space = {'radius_spatial': hp.uniform('radius_spatial', .5, 10),
         'radius_freq': hp.uniform('radius_freq', .5, 50),  # hp.uniform('radius_freq', .5, 100)
         'min_size_spatial': hp.uniform('min_size_spatial', .5, 10),  # hp.uniform('min_size_spatial', .5, 10)
         'min_size_freq': hp.uniform('min_size_freq', 10, 100),  # hp.uniform('min_size_freq', 5, 50),
         'min_voxels': hp.uniform('min_voxels', 1, 150),  # hp.uniform('min_voxels', 50, 150)
         'dilation_max_spatial': hp.uniform('dilation_max_spatial', .5, 20),
         'dilation_max_freq': hp.uniform('dilation_max_freq', .5, 20),
         'mask_threshold': hp.uniform('mask_threshold', .1, 1),
         }

start_values = [{'radius_spatial': 10,
                 'radius_freq': 10,
                 'min_size_spatial': 2,
                 'min_size_freq': 10,
                 'min_voxels': 50,
                 'dilation_max_spatial': 10,
                 'dilation_max_freq': 3,
                 'mask_threshold': .5,
                 }]
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)

print(best)
