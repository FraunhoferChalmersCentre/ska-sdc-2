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


class SKADataSet(AbstractSKADataset):
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

    def __getitem__(self, item):
        if isinstance(item, slice):
            indeces = np.arange(self.__len__())[item]
            shapes = [v.size() for v in self.get_attribute('image')[item]]
            index_from = [([shape[0] - self.get_attribute('dim')[0], shape[1] - self.get_attribute('dim')[1],
                            shape[2] - self.get_attribute('dim')[2]] *
                           self.get_randomizer().random(3, idx)).astype(np.int) for shape, idx in zip(shapes, indeces)]
            index_to = [i_from + np.array(
                [self.get_attribute('dim')[0], self.get_attribute('dim')[1], self.get_attribute('dim')[2]]) for i_from
                        in index_from]
            get_item = dict()
            get_item['image'] = torch.stack([image[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] for image, i0, i1 in
                                             zip(self.get_attribute('image')[item], index_from, index_to)])
            indeces_less = indeces[indeces < self.get_attribute('index')]
            indeces_greater = indeces[indeces >= self.get_attribute('index')]
            segmentmap = list()
            segmentmap += [self.get_attribute('segmentmap')[indeces_less[i]]
                           [index_from[i][0]:index_to[i][0], index_from[i][1]:index_to[i][1],
                           index_from[i][2]:index_to[i][2]]
                           for i in range(len(indeces_less))]
            segmentmap += [self.get_attribute('segmentmap')[self.get_attribute('index')] for i in
                           range(len(indeces_greater))]
            get_item['segmentmap'] = torch.stack(segmentmap)
            for c_key in self.get_common_keys():
                get_item[c_key] = torch.stack([self.get_attribute(c_key)[i] for i in indeces])
            for d_key in self.get_different_keys():
                get_item[d_key] = torch.stack([self.get_attribute(d_key)[i] for i in indeces_less])
        else:
            shape = self.get_attribute('image')[item].size()
            index_from = ([shape[0] - self.get_attribute('dim')[0], shape[1] - self.get_attribute('dim')[1],
                           shape[2] - self.get_attribute('dim')[2]] *
                          self.get_randomizer().random(3, item)).astype(np.int)
            index_to = index_from + np.array(
                [self.get_attribute('dim')[0], self.get_attribute('dim')[1], self.get_attribute('dim')[2]])
            get_item = dict()
            get_item['image'] = self.get_attribute('image')[item][index_from[0]:index_to[0], index_from[1]:index_to[1],
                                index_from[2]:index_to[2]]
            if item < self.get_attribute('index'):
                get_item['segmentmap'] = self.get_attribute('segmentmap')[item][index_from[0]:index_to[0],
                                         index_from[1]:index_to[1], index_from[2]:index_to[2]]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_soruce_keys()})
            else:
                get_item['segmentmap'] = self.get_attribute('segmentmap')[self.get_attribute('index')]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_common_keys()})
                get_item.update(
                    {k: self.get_attribute(k)[self.get_attribute('index')] for k in self.get_different_keys()})
        return get_item


class StaticSKATransformationDecorator(AbstractSKADataset, ABC):

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

    """
    def __init__(self, transform_input: List[str], transform_output: List[str], transform: Callable, decorated: AbstractSKADataset):
        self.decorated = decorated.clone()
        self.transformed_data = self._transform_attributes(transform, transform_input, transform_output)

    def _transform_attributes(self, transform, transform_input, transform_output):
        items = tuple(map(lambda k: self.decorated.get_attribute(k), transform_input))
        items = transform(*items)
        for key in transform_output:
            self.decorated.delete_key(key)
        if isinstance(items, list):
            items = (items,)
        return {k: v for k, v in zip(transform_output, items)}
    """

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

    def __getitem__(self, item):
        if isinstance(item, slice):
            indeces = np.arange(self.__len__())[item]
            shapes = [v.size() for v in self.get_attribute('image')[item]]
            index_from = [([shape[0] - self.get_attribute('dim')[0], shape[1] - self.get_attribute('dim')[1],
                            shape[2] - self.get_attribute('dim')[2]] *
                           self.get_randomizer().random(3, idx)).astype(np.int) for shape, idx in zip(shapes, indeces)]
            index_to = [i_from + np.array(
                [self.get_attribute('dim')[0], self.get_attribute('dim')[1], self.get_attribute('dim')[2]]) for i_from
                        in index_from]
            get_item = dict()
            get_item['image'] = torch.stack([image[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] for image, i0, i1 in
                                             zip(self.get_attribute('image')[item], index_from, index_to)])
            indeces_less = indeces[indeces < self.get_attribute('index')]
            indeces_greater = indeces[indeces >= self.get_attribute('index')]
            segmentmap = list()
            segmentmap += [self.get_attribute('segmentmap')[indeces_less[i]]
                           [index_from[i][0]:index_to[i][0], index_from[i][1]:index_to[i][1],
                           index_from[i][2]:index_to[i][2]]
                           for i in range(len(indeces_less))]
            segmentmap += [self.get_attribute('segmentmap')[self.get_attribute('index')] for i in
                           range(len(indeces_greater))]
            get_item['segmentmap'] = torch.stack(segmentmap)
            for c_key in self.get_common_keys():
                get_item[c_key] = torch.stack([self.get_attribute(c_key)[i] for i in indeces])
            for d_key in self.get_different_keys():
                get_item[d_key] = torch.stack([self.get_attribute(d_key)[i] for i in indeces_less])
        else:
            shape = self.get_attribute('image')[item].size()
            index_from = ([shape[0] - self.get_attribute('dim')[0], shape[1] - self.get_attribute('dim')[1],
                           shape[2] - self.get_attribute('dim')[2]] *
                          self.get_randomizer().random(3, item)).astype(np.int)
            index_to = index_from + np.array(
                [self.get_attribute('dim')[0], self.get_attribute('dim')[1], self.get_attribute('dim')[2]])
            get_item = dict()
            get_item['image'] = self.get_attribute('image')[item][index_from[0]:index_to[0], index_from[1]:index_to[1],
                                index_from[2]:index_to[2]]
            if item < self.get_attribute('index'):
                get_item['segmentmap'] = self.get_attribute('segmentmap')[item][index_from[0]:index_to[0],
                                         index_from[1]:index_to[1], index_from[2]:index_to[2]]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_soruce_keys()})
            else:
                get_item['segmentmap'] = self.get_attribute('segmentmap')[self.get_attribute('index')]
                get_item.update({k: self.get_attribute(k)[item] for k in self.get_common_keys()})
                get_item.update(
                    {k: self.get_attribute(k)[self.get_attribute('index')] for k in self.get_different_keys()})
        return get_item


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
