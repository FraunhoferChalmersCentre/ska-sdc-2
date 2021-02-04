import numpy as np
from typing import Dict
from utils.data.ska_dataset import SKADataSet
from utils.tensor import num_workers
from torch.utils.data import DataLoader


def split(dataset: Dict[str, np.ndarray], left_fraction: float, unit_key: str):
    n_units = np.unique(dataset[unit_key])
    n_left = int(len(n_units) * left_fraction)

    left_units = np.random.choice(n_units, size=n_left, replace=False)
    is_left = np.array([u in left_units for u in dataset[unit_key]])

    return {k: v[is_left] for k, v in dataset.items()}, {k: v[~is_left] for k, v in dataset.items()}


def splitted_loaders(dataset: Dict[str, np.ndarray], train_fraction: float, batch_size: int, unit_key: str = 'cluster'):
    train, validation = split(dataset, train_fraction, unit_key)
    trainset, validation_set = SKADataSet(train), SKADataSet(validation)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers())
    valloader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, num_workers=num_workers())
    return trainloader, valloader
