import glob
import random
from collections import OrderedDict

import torch
from tqdm.auto import tqdm

from utils.data.splitting import merge


def read_splitted_dataset(directory: str):
    files = glob.glob(directory + '/*.pt')
    random.shuffle(files)

    dataset_splits_dict = dict()
    for f in tqdm(files):
        dataset_splits_dict[f] = torch.load(f)
    dataset_splits_dict = OrderedDict(sorted(dataset_splits_dict.items(), key=lambda x: x[0]))

    merged_dataset = merge(*dataset_splits_dict.values())

    return merged_dataset
