from pipeline.common import filename
from pipeline.traversing.traverser import EvaluationTraverser
import numpy as np
import torch
from definitions import config
from pipeline.traversing.memory import max_batch_size

from pipeline.segmentation.utils import get_base_segmenter, get_state_dict

fits_file = config['traversing']['fits_file']

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

mbatch = max_batch_size(segmenter.model, model_input_dim, config['traversing']['gpu_memory_max'])

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, mbatch)

df = evaluator.traverse()
