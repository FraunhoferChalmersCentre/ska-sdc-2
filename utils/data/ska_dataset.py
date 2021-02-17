import copy
from abc import ABC
from typing import Callable, List

import torch
from torch.utils.data import Dataset


class AbstractSKADataset(Dataset):

    def add_attribute(self, additional_attributes: dict):
        raise NotImplementedError

    def get_keys(self):
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


class SKADataSet(AbstractSKADataset):
    def __init__(self, attributes: dict):

        self.data = attributes
        for k, v in attributes.items():
            self.data[k] = torch.Tensor(v)

    def add_attribute(self, additional_attributes: dict):
        for k, v in additional_attributes.items():
            self.data[k] = torch.Tensor(v)

    def get_keys(self):
        return self.data.keys()

    def delete_key(self, key):
        del self.data[key]

    def get_attribute(self, key):
        return self.data[key]

    def clone(self):
        return copy.deepcopy(self)

    def __len__(self):
        first_key = list(self.get_keys())[0]
        return len(self.get_attribute(first_key))

    def __getitem__(self, item):
        return {k: self.get_attribute(k)[item] for k in self.get_keys()}


class StaticSKATransformationDecorator(AbstractSKADataset, ABC):
    def __init__(self, transformed_keys: List[str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated.clone()
        self.transformed_data = self._transform_attributes(transform, transformed_keys)

    def _transform_attributes(self, transform, transformed_keys):
        items = tuple(map(lambda k: self.decorated.get_attribute(k), transformed_keys))

        for key in transformed_keys:
            self.decorated.delete_key(key)

        items = transform(*items)
        if type(items) is torch.Tensor:
            items = tuple(items.view(1, *items.shape))

        return {k: v for k, v in zip(transformed_keys, items)}

    def get_keys(self):
        return list(self.transformed_data.keys()) + list(self.decorated.get_keys())

    def add_attribute(self, additional_attributes: dict):
        self.decorated.add_attribute(additional_attributes)

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
        first_key = list(self.get_keys())[0]
        return len(self.get_attribute(first_key))

    def __getitem__(self, item):
        return {k: self.get_attribute(k)[item] for k in self.get_keys()}


class DynamicSKATransformationDecorator(AbstractSKADataset, ABC):
    def __init__(self, attributes: List[str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated
        self.attributes = attributes
        self.transform = transform

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
        base_items = tuple(map(lambda k: items[k], self.attributes))
        transformed_items = self.transform(*base_items)
        if type(transformed_items) is torch.Tensor:
            transformed_items = tuple(transformed_items.view(1, *transformed_items.shape))
        for attr, transformed in zip(self.attributes, transformed_items):
            items[attr] = transformed
        return items
