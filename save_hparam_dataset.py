import os

import numpy as np
import torch
from astropy.io.fits import getdata
from astropy.wcs import WCS
from spectral_cube import SpectralCube

from pipeline.common import filename
from pipeline.traversing.traverser import EvaluationTraverser
from definitions import config
from pipeline.traversing.memory import max_batch_size
from pipeline.segmentation.utils import get_base_segmenter, get_state_dict, generate_validation_input_cube, \
    generate_validation_segmentmap, generate_validation_catalogue

val_dataset_path = filename.processed.validation_dataset(config['segmentation']['size'],
                                                         100 * config['segmentation']['validation']['reduction'])

cube_shape, header = generate_validation_input_cube(val_dataset_path)

fits_file = val_dataset_path + '/input_cube.fits'

segmentmap = generate_validation_segmentmap(val_dataset_path, header)

df_true = generate_validation_catalogue(val_dataset_path, segmentmap)



# EvaluationTraverser
model_input_dim = np.array([64, 64, 64])
cnn_padding = np.array([8, 8, 8])
desired_dim = np.flip(cube_shape) - 2 * cnn_padding

hyperparam_dataset_path = filename.processed.hyperopt_dataset(config['segmentation']['size'],
                                                              100 * config['segmentation']['validation']['reduction'],
                                                              config['traversing']['checkpoint'])
model_out_path = hyperparam_dataset_path + '/output.fits'
if not os.path.exists(model_out_path):
    segmenter = get_base_segmenter()
    device = torch.device('cuda')

    state_dict = get_state_dict(config['traversing']['checkpoint'])

    segmenter.load_state_dict(state_dict)
    segmenter.to(device)
    torch.cuda.empty_cache()

    sofia_padding = np.array([0, 0, 0])

    mbatch = max_batch_size(segmenter.model, model_input_dim, config['traversing']['gpu_memory_max'])

    evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                    sofia_padding, mbatch)
    _ = evaluator.traverse(save_output=True, output_name=model_out_path)

clipped_input_path = hyperparam_dataset_path + '/clipped_input.fits'
if not os.path.exists(clipped_input_path):
    wcs = WCS(header)[[slice(c, -c) for c in cnn_padding]]
    clipped_data_cube = getdata(fits_file, ignore_blank=True)[[slice(c, -c) for c in cnn_padding]]
    clipped_input = SpectralCube(clipped_data_cube, wcs, header=header)
    clipped_input.write(clipped_input_path)
