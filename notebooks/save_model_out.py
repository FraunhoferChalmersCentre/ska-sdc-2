# !/usr/bin/env python
# coding: utf-8

# Add project root to path

# In[ ]:


import os
import sys

from definitions import ROOT_DIR

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Load dataset

# In[ ]:


import torch
from utils import filename
from utils import filehandling

size = 'dev_s'
prob = 50

directory = filename.processed.dataset(size, prob)
dataset = filehandling.read_splitted_dataset(directory)

# Split to train & test

# In[ ]:

import numpy as np
from utils.data import splitting

random_state = np.random.RandomState(5)
train, test = splitting.train_val_split(dataset, .8, random_state=random_state, train_filter=None)
print(len(train), len(test))

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

# Create Lightning objects
# %%
import numpy as np

training_set_ratio = 0.0016

n_mask_voxels = sum(map(lambda t: t.sum(), dataset['segmentmap'])).numpy()
real_ratio = n_mask_voxels / np.prod(hi_shape)

pos_weight = torch.tensor(real_ratio / training_set_ratio)
# %%
from operator import itemgetter
import pickle
from pipeline.traverser import CubeCache
import numpy as np

if os.path.isfile(ROOT_DIR + "/saved_models/statistic.p"):
    scale, mean, std = itemgetter('scale', 'mean', 'std')(
        pickle.load(open(ROOT_DIR + "/saved_models/statistic.p", "rb")))
else:
    header = fits.getheader(filename.data.sky(size), ignore_blank=True)
    cc = CubeCache(filename.data.sky(size))
    scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))
    pickle.dump({'scale': scale, 'mean': mean, 'std': std}, open(ROOT_DIR + "/saved_models/statistic.p", 'wb'))

# %%
state_dict = torch.load(ROOT_DIR + '/saved_models/resnet18-15-epoch=359-val_loss=1.11.ckpt')['state_dict']
state_dict = {k: v for k, v in state_dict.items() if k.startswith('model')}
# %%


import pytorch_lightning as pl
from pipeline.segmenter import BaseSegmenter
from training.hyperopt_segmenter import ValidationOutputSaveSegmenter

base_segmenter = BaseSegmenter(model, train.get_attribute('scale'), train.get_attribute('mean'),
                               train.get_attribute('std'))
segmenter = ValidationOutputSaveSegmenter(base_segmenter, test)
segmenter.load_state_dict(state_dict)
segmenter.eval()

trainer = pl.Trainer(gpus=1)

trainer.validate(segmenter)
test.delete_key('segmentmap')

# %%

datadict = {}
for k in test.get_keys():
    datadict[k] = test.get_attribute(k)

import torch
import string
import random
import glob

directory = filename.processed.hyperopt_dataset(size, prob, modelname)

random.seed(10)

for f in glob.glob('{}/*'.format(directory)):
    os.remove(f)

name = ''.join(random.choice(string.ascii_letters) for i in range(20))
torch.save(datadict, directory + '/{}.pt'.format(name))
