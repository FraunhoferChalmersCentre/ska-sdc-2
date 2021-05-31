import os

from astropy.io import fits
import torch
import string
import random
import glob
import pytorch_lightning as pl

from pipeline.common import filename
from pipeline.segmentation.utils import get_data, get_checkpoint_callback, get_state_dict, get_base_segmenter
from pipeline.hyperparameter.validation import ValidationOutputSaveSegmenter
from definitions import config

validation_set = get_data(only_validation=True)
checkpoint_callback = get_checkpoint_callback()
base_segmenter = get_base_segmenter()

segmenter = ValidationOutputSaveSegmenter(base_segmenter,
                                          validation_set=validation_set,
                                          header=fits.getheader(filename.data.sky(config['segmentation']['size'])))

state_dict = get_state_dict(config['traversing']['checkpoint'])
segmenter.load_state_dict(state_dict)
segmenter.eval()

trainer = pl.Trainer(gpus=1)

trainer.validate(segmenter)

datadict = {}
for k in validation_set.get_keys():
    datadict[k] = validation_set.get_attribute(k)

directory = filename.processed.hyperopt_dataset(config['segmentation']['size'], config['segmentation']['model_name'])

random.seed(10)

for f in glob.glob('{}/*'.format(directory)):
    os.remove(f)

name = ''.join(random.choice(string.ascii_letters) for i in range(20))
torch.save(datadict, directory + '/{}.pt'.format(name))
