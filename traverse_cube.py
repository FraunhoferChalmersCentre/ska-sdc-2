from utils import filename
from pipeline.traverse import EvaluationTraverser
import numpy as np
import torch
from definitions import config
from utils.memory import max_batch_size

from utils.training import get_base_segmenter, get_state_dict

fits_file = filename.data.sky('eval')

segmenter = get_base_segmenter()
device = torch.device('cuda')

state_dict = get_state_dict(config['traversing']['checkpoint'])
segmenter.load_state_dict(state_dict)
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
