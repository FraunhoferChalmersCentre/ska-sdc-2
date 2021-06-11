from typing import Dict, List

import numpy as np
import torch

from pipeline.data.ska_dataset import SKADataSet, StaticSKATransformationDecorator, TrainingItemGetter, \
    ValidationItemGetter
from pipeline.data.generating import COMMON_ATTRIBUTES, SOURCE_ATTRIBUTES, GLOBAL_ATTRIBUTES


def to_float(tensors: List[torch.Tensor]): return list(map(lambda t: t.float(), tensors))


def unsqueeze(tensors: List[torch.Tensor]): return list(map(lambda t: t.unsqueeze(0), tensors))


def fill_dict(units: np.ndarray, dataset: Dict[str, np.ndarray], required_attrs: List[str]):
    split_dict = dict()
    for k, v in dataset.items():
        if k == 'index':
            split_dict[k] = len(units[units < v])
            continue
        elif k in GLOBAL_ATTRIBUTES:
            split_dict[k] = v
            continue

        split_dict[k] = list()
        for i in units:
            if k not in required_attrs and i >= dataset['index']:
                continue
            split_dict[k].append(v[i])

        if k not in required_attrs:
            split_dict[k].append(v[-1])

    return split_dict


def filter_units(dataset: Dict, units: np.ndarray, attribute: str, fraction: float):
    empty_units = units[units >= dataset['index']]
    empty_units = np.random.choice(empty_units, size=int(len(empty_units) * fraction), replace=False)

    source_units = units[units < dataset['index']]
    filter_values = np.array([dataset[attribute][u] for u in source_units])
    threshold = np.percentile(filter_values, fraction * 100)
    source_units = source_units[filter_values > threshold]

    return np.concatenate([source_units, empty_units])


def split(dataset: Dict, left_fraction: float, required_attrs: List[str], random_state: np.random.RandomState,
          left_filter: float = None, right_filter: float = None, filter_attr: str = 'line_flux_integral'):
    n_units = len(dataset[required_attrs[0]])

    all_units = np.arange(n_units)
    noise_units = all_units[all_units >= dataset['index']]
    source_units = all_units[all_units < dataset['index']]

    noise_positions = [p[0, 0] for p in dataset['position'][dataset['index']:]]
    noise_split = np.percentile(noise_positions, int(left_fraction * 100))
    source_positions = [p[0, 0] for p in dataset['position'][:dataset['index']]]
    source_split = np.percentile(source_positions, int(left_fraction * 100))

    left_units = np.concatenate(
        ([noise_units[i] for i in range(len(noise_units)) if noise_positions[i] < noise_split],
         [source_units[i] for i in range(len(source_units)) if source_positions[i] < source_split]))

    right_units = np.setdiff1d(all_units, left_units).astype(np.int32)

    if left_filter is not None:
        left_units = filter_units(dataset, left_units, filter_attr, left_filter)

    if right_filter is not None:
        right_units = filter_units(dataset, right_units, filter_attr, right_filter)

    splits = tuple(map(np.sort, [left_units, right_units]))

    return tuple(map(lambda s: fill_dict(s, dataset, required_attrs), splits))


def add_transforms(base_dataset):
    for attr in ['image', 'segmentmap']:
        base_dataset = StaticSKATransformationDecorator(attr, to_float, base_dataset)
        base_dataset = StaticSKATransformationDecorator(attr, unsqueeze, base_dataset)
    return base_dataset


def merge(*datasets: Dict):
    merged = dict()
    index = 0

    for d in datasets:
        index += d['index']

    merged['index'] = index

    # Add source boxes
    for d in datasets:
        for k, v in d.items():
            if k == 'index':
                continue
            elif k in GLOBAL_ATTRIBUTES:
                if k not in merged.keys():
                    merged[k] = v
            else:
                if k not in merged.keys():
                    merged[k] = list()

                merged[k].extend(v[:d['index']])

    # Add empty boxes common attributes
    for d in datasets:
        for k, v in d.items():
            if k in COMMON_ATTRIBUTES:
                merged[k].extend(v[d['index']:])

    # Add dummy values for empty boxes
    # Assumed that datasets[0] has no empty boxes
    for k, v in datasets[0].items():
        if k in SOURCE_ATTRIBUTES:
            merged[k].append(v[-1])

    return merged


def train_val_split(dataset: Dict, train_fraction: float, required_attrs: List[str] = ['image', 'position'],
                    random_state=np.random.RandomState(), train_filter=None,
                    validation_item_getter=ValidationItemGetter()):
    train, validation = split(dataset, train_fraction, required_attrs, random_state, left_filter=train_filter)
    datsets = (SKADataSet(train, TrainingItemGetter()), SKADataSet(validation, validation_item_getter, random_type=1))

    return tuple(map(add_transforms, datsets))
