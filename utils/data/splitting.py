import numpy as np
from typing import Dict, Tuple
from utils.data.ska_dataset import SKADataSet, StaticSKATransformationDecorator
import torch


def to_float(*tensors: Tuple[torch.Tensor]): return tuple(map(lambda t: t.float(), tensors))


def split(dataset: Dict[str, np.ndarray], left_fraction: float, unit_key: str):
    n_units = np.unique(dataset[unit_key])
    n_left = int(len(n_units) * left_fraction)

    left_units = np.random.choice(n_units, size=n_left, replace=False)
    is_left = np.array([u in left_units for u in dataset[unit_key]])

    return {k: v[is_left] for k, v in dataset.items()}, {k: v[~is_left] for k, v in dataset.items()}


def splitted_loaders(dataset: Dict[str, np.ndarray], train_fraction: float, unit_key: str = 'cluster'):
    train, validation = split(dataset, train_fraction, unit_key)
    training_set, validation_set = SKADataSet(train), SKADataSet(validation)

    training_set = StaticSKATransformationDecorator(['image', 'segmentmap'], to_float, training_set)
    validation_set = StaticSKATransformationDecorator(['image', 'segmentmap'], to_float, validation_set)
    return training_set, validation_set
