from definitions import config

size = config['segmentation']['size']
prob = 50
splitsize = config['data']['splitsize']

# %%


import pandas as pd

from utils import filename
from utils.data.segmentmap import create_from_files

segmentmap, allocation_dict = create_from_files(size, regenerate=True)
df = pd.read_csv(filename.data.true(size), sep=' ')
fname = filename.data.sky(size)

# %%

from utils.data.generating import split_by_size

spatial = config['segmentation']['cube_size']['spatial']
freq = config['segmentation']['cube_size']['freq']
cube_dim = (spatial, spatial, freq)

galax_prob = prob / 100

splitted_datasets = split_by_size(df, fname, segmentmap, allocation_dict, galax_prob, cube_dim,
                                  n_memory_batches=config['data']['memory_batches'], splitsize=splitsize)

# %%

import os
import torch
from tqdm.auto import tqdm
import string
import random
import glob

directory = filename.processed.dataset(size, prob)

random.seed(10)

for f in glob.glob('{}/*'.format(directory)):
    os.remove(f)

for i, dataset_split in enumerate(tqdm(splitted_datasets)):
    name = ''.join(random.choice(string.ascii_letters) for j in range(20))
    torch.save(dataset_split, directory + '/{}.pt'.format(name))
