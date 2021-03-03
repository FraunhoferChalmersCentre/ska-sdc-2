import glob
import random

import torch
from tqdm.notebook import tqdm

from utils.data.splitting import merge

def read_splitted_dataset(directory: str):
    files = glob.glob(directory + '/split_*.pt')
    #random.shuffle(files)
    
    dataset_splits = list()
    for f in tqdm(files):
        dataset_splits.append(torch.load(f))
    merged_dataset = merge(*dataset_splits)
    
    return merged_dataset