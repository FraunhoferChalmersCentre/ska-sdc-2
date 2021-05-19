import copy
from abc import ABC
from typing import Callable, List, Union

import torch
import numpy as np
from torch.utils.data import Dataset


class ItemGettingStrategy(ABC):
    def get_item_strategy(self, dataset, item):
        raise NotImplementedError


class ValidationItemGetter(ItemGettingStrategy):
    def get_item_strategy(self, dataset, item):
        if isinstance(item, slice):
            raise NotImplementedError('Not implemented slice items')
        elif isinstance(item, tuple):
            raise NotImplementedError('Not implemented tuple items')
        else:
            item_dict = dict()
            if item < dataset.get_attribute('index'):
                item_dict['image'] = dataset.get_attribute('image')[item]
                item_dict['segmentmap'] = dataset.get_attribute('segmentmap')[item]
                item_dict.update({k: dataset.get_attribute(k)[item] for k in dataset.get_source_keys()})
            else:
                item_dict['image'] = dataset.get_attribute('image')[item]
                item_dict['segmentmap'] = dataset.get_attribute('segmentmap')[dataset.get_attribute('index')]
                item_dict.update({k: dataset.get_attribute(k)[item] for k in dataset.get_common_keys()})
                item_dict.update(
                    {k: dataset.get_attribute(k)[dataset.get_attribute('index')] for k in dataset.get_different_keys()})
            return item_dict


class TrainingItemGetter(ItemGettingStrategy):
    @staticmethod
    def _inside_cube(pos, random, dim_s, dim_l):
        if pos - dim_s < 0:
            start = int(pos - random * pos)
        elif pos + dim_s > dim_l:
            start = int(pos - dim_s + random * (dim_l - pos))
        else:
            start = int(pos - random * dim_s)
        end = int(start + dim_s)
        return start, end

    def get_item_strategy(self, dataset, item):
        if isinstance(item, slice):
            raise NotImplementedError('Not implemented slice items')
        elif isinstance(item, tuple):
            raise NotImplementedError('Not implemented tuple items')
        else:
            shape = dataset.get_attribute('image')[item].size()

            slices = [slice(None)] * len(shape)

            dim = dataset.get_attribute('dim')
            item = item % dataset.__len__()
            randoms = dataset.get_randomizer().random(len(dim), item)

            item_dict = dict()

            if item < dataset.get_attribute('index'):
                pos_index = int(
                    dataset.get_attribute('allocated_voxels')[item].shape[0] * dataset.get_randomizer().random(1, item))
                position = dataset.get_attribute('allocated_voxels')[item][pos_index]
                slices[-len(dim):] = [slice(*TrainingItemGetter._inside_cube(p, r, d, s)) for s, p, d, r in
                                      zip(shape[-len(dim):], position, dim, randoms)]

                item_dict['segmentmap'] = dataset.get_attribute('segmentmap')[item][slices]
                item_dict.update({k: dataset.get_attribute(k)[item] for k in dataset.get_source_keys()})
            else:
                slices[-len(dim):] = [slice(int(r * (s - d)), int(r * (s - d)) + d) for s, d, r in
                                      zip(shape[-len(dim):], dim, randoms)]

                item_dict['segmentmap'] = dataset.get_attribute('segmentmap')[dataset.get_attribute('index')][slices]
                item_dict.update({k: dataset.get_attribute(k)[item] for k in dataset.get_common_keys()})
                item_dict.update(
                    {k: dataset.get_attribute(k)[dataset.get_attribute('index')] for k in dataset.get_different_keys()})

            item_dict['image'] = dataset.get_attribute('image')[item][slices]
            item_dict['slices'] = torch.tensor([[s.start, s.stop] for s in slices[1:]]).T
        return item_dict


class AbstractSKADataset(Dataset):

    def add_attribute(self, additional_attributes: dict):
        raise NotImplementedError

    def get_keys(self):
        raise NotImplementedError

    def get_source_keys(self):
        raise NotImplementedError

    def get_empty_keys(self):
        raise NotImplementedError

    def get_common_keys(self):
        raise NotImplementedError

    def get_different_keys(self):
        raise NotImplementedError

    def delete_key(self, key):
        raise NotImplementedError

    def get_attribute(self, key):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item) -> dict:
        raise NotImplementedError

    def get_item_getter(self) -> ItemGettingStrategy:
        raise NotImplementedError


class SKADataSet(AbstractSKADataset):
    def __init__(self, attributes: dict, item_getter: ItemGettingStrategy, random_type: Union[int, None] = None,
                 source_keys: list = None,
                 empty_keys: list = None):

        self.item_getter = item_getter
        self.data = attributes
        for k, v in attributes.items():
            if isinstance(v, list) and not isinstance(v[0], torch.Tensor):
                if isinstance(v[0], np.ndarray):
                    self.data[k] = [torch.Tensor(value.astype(np.float32)) for value in v]
                else:
                    self.data[k] = torch.Tensor(v)

        # Randomizer
        self.randomizer = Randomizer(random_type)

        # Key Handling
        if source_keys is None:
            self.source_keys = [k for k in list(self.get_keys()) if k not in ['index', 'dim', 'image', 'segmentmap',
                                                                              'allocated_voxels']]
        if empty_keys is None:
            self.empty_keys = ['position']
        self.common_keys = list(set.intersection(set(self.source_keys), set(self.empty_keys)))
        self.different_keys = list(set(self.source_keys) - set(self.common_keys))

    def __getitem__(self, item):
        return self.get_item_getter().get_item_strategy(self, item)

    def add_attribute(self, additional_attributes: dict, source_keys: list = None, empty_keys: list = None):
        for k, v in additional_attributes.items():
            if isinstance(v, list) and not isinstance(v[0], torch.Tensor):
                if isinstance(v[0], np.ndarray):
                    self.data[k] = [torch.Tensor(value.astype(np.float32)) for value in v]
                else:
                    self.data[k] = torch.Tensor(v)

        if source_keys is not None:
            self.source_keys += source_keys
        if empty_keys is not None:
            self.empty_keys += empty_keys
        self.common_keys = list(set.intersection(set(self.source_keys), set(self.empty_keys)))
        self.different_keys = list(set(self.source_keys) - set(self.common_keys))


    def get_keys(self):
        return self.data.keys()

    def get_source_keys(self):
        return self.source_keys

    def get_empty_keys(self):
        return self.empty_keys

    def get_common_keys(self):
        return self.common_keys

    def get_different_keys(self):
        return self.different_keys

    def get_randomizer(self):
        return self.randomizer

    def delete_key(self, key):
        del self.data[key]

    def get_attribute(self, key):
        return self.data[key]

    def clone(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.get_attribute('image'))

    def get_item_getter(self) -> ItemGettingStrategy:
        return self.item_getter


class StaticSKATransformationDecorator(AbstractSKADataset):

    def __init__(self, transformed_keys: Union[List[str], str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated
        self.transformed_data = self._transform_attributes(transform, transformed_keys)

    def _transform_attributes(self, transform, transformed_keys):
        if isinstance(transformed_keys, str):
            transformed = transform(self.decorated.get_attribute(transformed_keys))
            self.decorated.delete_key(transformed_keys)
            return {transformed_keys: transformed}
        else:
            raise NotImplementedError

    def get_keys(self):
        return list(self.transformed_data.keys()) + list(self.decorated.get_keys())

    def get_source_keys(self):
        return self.decorated.get_source_keys()

    def get_empty_keys(self):
        return self.decorated.get_empty_keys()

    def get_common_keys(self):
        return self.decorated.get_common_keys()

    def get_different_keys(self):
        return self.decorated.get_different_keys()

    def add_attribute(self, additional_attributes: dict):
        self.decorated.add_attribute(additional_attributes)

    def get_randomizer(self):
        return self.decorated.get_randomizer()

    def delete_key(self, key):
        if key in self.transformed_data.keys():
            del self.transformed_data[key]
        else:
            self.decorated.delete_key(key)

    def get_attribute(self, key):
        if key in self.transformed_data.keys():
            return self.transformed_data[key]

        return self.decorated.get_attribute(key)

    def clone(self):
        return self

    def __len__(self):
        return len(self.get_attribute('image'))

    def get_item_getter(self):
        return self.decorated.get_item_getter()

    def __getitem__(self, item):
        return self.get_item_getter().get_item_strategy(self, item)


class DynamicSKATransformationDecorator(AbstractSKADataset, ABC):
    def __init__(self, transformed_keys: Union[List[str], str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated
        self.transformed_keys = transformed_keys
        if not isinstance(transformed_keys, str):
            raise NotImplementedError
        self.transform = transform

    def get_keys(self):
        return self.decorated.get_keys()

    def get_source_keys(self):
        return self.decorated.get_source_keys()

    def get_empty_keys(self):
        return self.decorated.get_empty_keys()

    def get_common_keys(self):
        return self.decorated.get_common_keys()

    def get_different_keys(self):
        return self.decorated.get_different_keys()

    def add_attribute(self, additional_attributes: dict):
        self.decorated.add_attribute(additional_attributes)

    def get_attribute(self, key):
        return self.decorated.get_attribute(key)

    def delete_key(self, key):
        self.decorated.delete_key(key)

    def __getitem__(self, item):
        items = self.decorated.__getitem__(item)
        return self.decorate(items)

    def __len__(self):
        return len(self.decorated)

    def decorate(self, items: dict):
        items[self.transformed_keys] = self.transform(items[self.transformed_keys])
        return items


class Randomizer:

    def __init__(self, rtype: Union[int, None] = None):
        self.rtype = rtype

    def random(self, size, seed=None):
        if isinstance(self.rtype, int):
            if seed is None:
                np.random.seed(self.rtype)
            else:
                np.random.seed(self.rtype + seed)
        rand = np.random.random(size)
        return rand
