import numpy as np
import pandas as pd
import torch
from astropy.io.fits import getheader

from pipeline.common import filename
from pipeline.data.segmentmap import prepare_df
from pipeline.traversing.traverser import EvaluationTraverser
from definitions import config
from pipeline.traversing.memory import max_batch_size
from pipeline.segmentation.utils import get_base_segmenter, get_state_dict, generate_validation_segmentmap
from pipeline.common import filename

test_dataset_path = filename.processed.test_dataset(config['traversing']['checkpoint'])

fits_file = filename.data.test_sky()

df = pd.read_csv(filename.data.test_true(), sep=' ', index_col='id')

header = getheader(fits_file)

segmentmap = generate_validation_segmentmap(test_dataset_path, header, df.copy())

df_true = prepare_df(df, header)
df_true.to_csv(f'{test_dataset_path}/df.txt', sep=' ', index_label='id')

# EvaluationTraverser
model_input_dim = np.array([128, 128, 128])
cnn_padding = np.array([8, 8, 8])
desired_dim = np.array([2, 2, 20]) * (model_input_dim - 2 * cnn_padding)

sofia_padding = np.array([12, 12, 100])

segmenter = get_base_segmenter()
device = torch.device('cuda')

state_dict = get_state_dict(config['traversing']['checkpoint'])

segmenter.load_state_dict(state_dict)
segmenter.to(device)
torch.cuda.empty_cache()

mbatch = max_batch_size(segmenter.model, model_input_dim, config['traversing']['gpu_memory_max'])

evaluator = EvaluationTraverser(segmenter, fits_file, model_input_dim, desired_dim, cnn_padding,
                                sofia_padding, mbatch)
_ = evaluator.traverse(save_output=True, save_input=True, output_path=test_dataset_path)
