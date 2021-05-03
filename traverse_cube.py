from utils import filename
from pipeline.traverser import EvaluationTraverser, CubeCache
from astropy.io import fits
import numpy as np
from pipeline.segmenter import BaseSegmenter
from pipeline.convert2Dto3D import Conv3dConverter
import torch
from definitions import ROOT_DIR
import segmentation_models_pytorch as smp

fits_file = filename.data.sky('eval')

# Should Save for Large Cube
header = fits.getheader(fits_file, ignore_blank=True)
cc = CubeCache(fits_file)
scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))

# Model
model = smp.Unet(encoder_name='resnet18', in_channels=1, classes=1, encoder_weights='imagenet')
Conv3dConverter(model, -1, torch.ones(1, 1, 32, 32, 32))
device = torch.device('cuda')
model.load_state_dict(torch.load(ROOT_DIR + '/saved_models/state.pt'), strict=False)
model.to(device)

# BaseSegmenter
segmenter = BaseSegmenter(model, scale, mean, std)
segmenter.to(device)

# EvaluationTraverser
model_input_dim = np.array([64, 64, 256])
desired_dim = np.array([100, 100, 550])
cnn_padding = np.array([16, 16, 32])
sofia_padding = np.array([16, 16, 32])
gpu_memory = 1.0

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, gpu_memory)

df = evaluator.traverse()
