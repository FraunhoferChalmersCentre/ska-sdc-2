import numpy as np
from typing import Dict, Tuple, List
from utils.data.ska_dataset import SKADataSet, StaticSKATransformationDecorator
import torch


def to_float(tensors: List[torch.Tensor]): return list(map(lambda t: t.float(), tensors))

def unsqueeze(tensors: List[torch.Tensor]): return list(map(lambda t: t.unsqueeze(0), tensors))

def fill_dict(units: np.ndarray, dataset: Dict[str, np.ndarray], required_attrs: List[str]):
    split_dict = dict()
    for k, v in dataset.items():
        if k == 'index':
            split_dict[k] = len(units[units < v])
            continue
        if k == 'dim':
            split_dict[k] = v
            continue
        
        split_dict[k] = list()
        for i in units:
            if k not in required_attrs and i > dataset['index']:
                continue
            split_dict[k].append(v[i])
            
        if k not in required_attrs:
            split_dict[k].append(v[-1])
            
    return split_dict


def split(dataset: Dict, left_fraction: float, required_attrs: List[str]):
    n_units = len(dataset[required_attrs[0]])
    n_left = int(n_units * left_fraction)

    left_units = np.random.choice(n_units, size=n_left, replace=False).astype(np.int32)
    right_units = np.setdiff1d(np.arange(n_units), left_units).astype(np.int32)
    
    splits = tuple(map(np.sort, [left_units, right_units]))

    return tuple(map(lambda s: fill_dict(s, dataset, required_attrs), splits))

def add_transforms(base_dataset):
    for attr in ['image', 'segmentmap']:
        base_dataset = StaticSKATransformationDecorator(attr, to_float, base_dataset)
        base_dataset = StaticSKATransformationDecorator(attr, unsqueeze, base_dataset)
    return base_dataset


def splitted_loaders(dataset: Dict, train_fraction: float, required_attrs: List[str] = ['image', 'position']):
    train, validation = split(dataset, train_fraction, required_attrs)
    datsets = (SKADataSet(train), SKADataSet(validation, random_type=1))

    return tuple(map(add_transforms, datsets))