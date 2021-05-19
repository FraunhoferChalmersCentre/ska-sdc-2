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
    scale, mean, std = itemgetter('scale', 'mean', 'std')(pickle.load(open(ROOT_DIR + "/saved_models/statistic.p", "rb")))
else:
    header = fits.getheader(fits_file, ignore_blank=True)
    cc = CubeCache(fits_file)
    scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))

# Model
model = smp.Unet(encoder_name='resnet18', in_channels=1, classes=1, encoder_weights='imagenet')
Conv3dConverter(model, -1, torch.ones(1, 1, 32, 32, 32))
device = torch.device('cuda')

# BaseSegmenter
segmenter = BaseSegmenter(model, scale, mean, std)
segmenter.load_state_dict(torch.load(ROOT_DIR + '/saved_models/10-epoch=59-sofia_dice=0.27.ckpt')['state_dict'],
                          strict=False)
segmenter.to(device)
torch.cuda.empty_cache()

# EvaluationTraverser
model_input_dim = np.array([64, 64, 256])
desired_dim = np.array([256, 256, 4096])
#desired_dim = np.array([100, 100, 550])
cnn_padding = np.array([16, 16, 16])
sofia_padding = np.array([8, 8, 50])
gpu_memory_mib = 5000

mbatch = max_batch_size(segmenter.model, model_input_dim, gpu_memory_mib)

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, mbatch)

df = evaluator.traverse()
