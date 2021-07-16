import glob
from collections import OrderedDict

import torch
from tqdm.auto import tqdm

from pipeline.data.splitting import merge


def read_splitted_dataset(directory: str, limit_files: int = None):
    files = glob.glob(directory + '/*.pt')

    dataset_splits_dict = dict()
    for i, f in enumerate(tqdm(files)):
        if limit_files and i >= limit_files:
            break
        dataset_splits_dict[f] = torch.load(f)
    dataset_splits_dict = OrderedDict(sorted(dataset_splits_dict.items(), key=lambda x: x[0]))

    merged_dataset = merge(*dataset_splits_dict.values())

    return merged_dataset
