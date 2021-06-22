import os

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from spectral_cube import SpectralCube

from pipeline.common import filename
from pipeline.traversing.traverser import EvaluationTraverser
from definitions import config
from pipeline.traversing.memory import max_batch_size

from pipeline.segmentation.utils import get_base_segmenter, get_state_dict

fits_file = filename.processed.hyperopt_dataset(config['segmentation']['size'],
                                                config['segmentation']['model_name']) + '/tmp.fits'


def save_tmp(split_point):
    if os.path.isfile(fits_file):
        os.remove(fits_file)
    full_fits_file = filename.data.sky(config['segmentation']['size'])
    data_cube = fits.getdata(full_fits_file)
    header = fits.getheader(full_fits_file)

    wcs = WCS(header)[:, :, split_point:]
    tmp = SpectralCube(data_cube[:, :, split_point:], wcs, header=header)

    tmp.write(fits_file)

    return tmp.shape


shape = save_tmp(486)

segmenter = get_base_segmenter()
device = torch.device('cuda')

state_dict = get_state_dict(config['traversing']['checkpoint'])

segmenter.load_state_dict(state_dict)
segmenter.to(device)
torch.cuda.empty_cache()

# EvaluationTraverser
model_input_dim = np.array([128, 128, 128])
cnn_padding = np.array([8, 8, 8])
desired_dim = np.flip(shape) - 2 * cnn_padding

sofia_padding = np.array([0, 0, 0])

mbatch = max_batch_size(segmenter.model, model_input_dim, config['traversing']['gpu_memory_max'])

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, mbatch)

_ = evaluator.traverse(for_validation=True)
