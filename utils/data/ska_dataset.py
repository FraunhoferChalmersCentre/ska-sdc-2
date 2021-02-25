import copy
from abc import ABC
from typing import Callable, List, Union

import torch
import numpy as np
from torch.utils.data import Dataset


class AbstractSKADataset(Dataset):

    def add_attribute(self, additional_attributes: dict):
        raise NotImplementedError

    def get_keys(self):
        raise NotImplementedError

    def get_soruce_keys(self):
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

class BaseSKADataSet(AbstractSKADataset):
    def __getitem__(self, item):
        if isinstance(item, slice):
            raise NotImplementedError('Not implemented slice items')
        elif isinstance(item, tuple):
            raise NotImplementedError('Not implemented tuple items')
        else:
            shape = self.get_attribute('image')[item].size()
            
            slices = [slice(None)] * len(shape)
            
            dim = self.get_attribute('dim').astype(np.int32)
            randoms = self.get_randomizer().random(len(dim), item)
            slices[-len(dim):] = [slice(int(r * (s - d)), int(r * (s - d)) + d) for s, d, r in zip(shape[-len(dim):], dim, randoms)]

            get_item = dict()
            get_item['image'] = self.get_attribute('image')[item][slices]
            
            if item < self.get_attribute('index'):
                get_item['segmentmap'] = self.get_attribute('segmentmap')[item][slices]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_soruce_keys()})
            else:
                get_item['segmentmap'] = self.get_attribute('segmentmap')[self.get_attribute('index')]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_common_keys()})
                get_item.update(
                    {k: self.get_attribute(k)[self.get_attribute('index')] for k in self.get_different_keys()})
        return get_item


class SKADataSet(BaseSKADataSet):
    def __init__(self, attributes: dict, random_type: Union[int, bool] = True, source_keys: list = None,
                 empty_keys: list = None):

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
            self.source_keys = [k for k in list(self.get_keys()) if k not in ['index', 'dim', 'image', 'segmentmap']]
        if empty_keys is None:
            self.empty_keys = ['position']
        self.common_keys = list(set.intersection(set(self.source_keys), set(self.empty_keys)))
        self.different_keys = list(set(self.source_keys) - set(self.common_keys))

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

    def get_soruce_keys(self):
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


class StaticSKATransformationDecorator(BaseSKADataSet):

    def __init__(self, transformed_keys: Union[List[str], str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated.clone()
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

    def get_soruce_keys(self):
        return self.decorated.get_soruce_keys()

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


class DynamicSKATransformationDecorator(AbstractSKADataset, ABC):
    def __init__(self, transformed_keys: Union[List[str], str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated
        self.transformed_keys = transformed_keys
        if not isinstance(transformed_keys, str):
            raise NotImplementedError
        self.transform = transform

    def get_keys(self):
        return self.decorated.get_keys()

    def get_soruce_keys(self):
        return self.decorated.get_soruce_keys()

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

    def __init__(self, rtype: Union[int, bool] = True):
        self.rtype = rtype

    def random(self, size, seed=None):
        if not isinstance(self.rtype, bool):
            if seed is None:
                np.random.seed(self.rtype)
            else:
                np.random.seed(seed)
        if not isinstance(self.rtype, bool) or self.rtype:
            rand = np.random.random(size)
        else:
            rand = np.ones(size) * 0.5
        return rand
