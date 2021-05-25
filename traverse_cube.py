from utils import filename
from pipeline.traverser import EvaluationTraverser, CubeCache
from astropy.io import fits
import numpy as np
from pipeline.segmenter import BaseSegmenter
from pipeline.convert2Dto3D import Conv3dConverter
import torch
from definitions import ROOT_DIR
import segmentation_models_pytorch as smp
from operator import itemgetter
from utils.memory import max_batch_size
import pickle
import os

fits_file = filename.data.sky('eval')

# Check if saved
if os.path.isfile(ROOT_DIR + "/saved_models/statistic.p"):
    scale, mean, std = itemgetter('scale', 'mean', 'std')(
        pickle.load(open(ROOT_DIR + "/saved_models/statistic.p", "rb")))
else:
    header = fits.getheader(fits_file, ignore_blank=True)
    cc = CubeCache(fits_file)
    scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))
    pickle.dump({'scale': scale, 'mean': mean, 'std': std}, open(ROOT_DIR + "/saved_models/statistic.p", 'wb'))

# Model
model = smp.Unet(encoder_name='resnet18', encoder_weights='ssl', in_channels=1, classes=1,
                 decoder_channels=[256, 128, 64, 32], encoder_depth=4, decoder_use_batchnorm=True)
Conv3dConverter(model, -1)
device = torch.device('cuda')

# BaseSegmenter
segmenter = BaseSegmenter(model, scale, mean, std)
segmenter.load_state_dict(torch.load(ROOT_DIR + '/saved_models/12-epoch=519-val_loss=1.03.ckpt')['state_dict'])
segmenter.to(device)
torch.cuda.empty_cache()

# EvaluationTraverser
model_input_dim = np.array([128, 128, 128 + 64])
cnn_padding = np.array([8, 8, 8])
desired_dim = 4 * (model_input_dim - 2 * cnn_padding)

sofia_padding = np.array([8, 8, 50])
gpu_memory_mib = 7000

mbatch = max_batch_size(segmenter.model, model_input_dim, gpu_memory_mib)

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, mbatch)

df = evaluator.traverse()
