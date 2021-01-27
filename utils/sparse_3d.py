from typing import Tuple

from scipy.sparse import dok_matrix
import numpy as np

from collections import Iterable


class Sparse3DArray:
    def __init__(self, value=None, shape: Tuple = None):
        if len(shape) != 3:
            raise Exception('Not supported yet')

        if value is not None:
            if not isinstance(value, Iterable):
                value = np.array([value])

            if len(value.shape) == 1:
                if shape is not None and len(value) == 1:
                    self.shape = shape
                    self._data = {i: dok_matrix(self.shape[1:]) for i in range(self.shape[0])}
                    for k, v in self._data.items():
                        v[:, :] = value
                elif shape is not None and len(value) != 1:
                    raise Exception('Not supported yet')
                else:
                    value = value.reshape(1, *value.shape, )
                    self.shape = (1, *value.shape,)
                    self._data = {0: dok_matrix(value)}

            elif len(value.shape) == 2:
                if shape is not None:
                    if shape[1:] != value.shape:
                        raise Exception('Not supported yet')
                    self.shape = shape
                    self._data = {i: dok_matrix(value) for i in range(shape[0])}
                else:
                    self.shape = (1, *value.shape,)
                    self._data = {0: dok_matrix(value)}

            elif len(value.shape) == 3:
                if shape is not None and shape != value.shape:
                    raise Exception('Not supported yet')
                self.shape = value.shape
                self._data = {i: dok_matrix(v) for i, v in enumerate(value)}

        elif shape is not None and len(shape) == 3:
            # Init an empty array
            self.shape = shape
            self._data = {}
        else:
            raise Exception('Specify either valid shape or value, or both.')

    def __len__(self):
        return self.shape[0]

    def _get_all_dim_keys(self, key):
        new_keys = []
        if not isinstance(key, Iterable):
            key = [key]
        for i, s in enumerate(self.shape):
            if len(key) < i - 1:
                new_keys.append(list(range(s)))
            else:
                if isinstance(key[i], int) or np.issubdtype(type(key[i]), np.integer) or isinstance(key[i], Iterable):
                    new_keys[i] = key[i]
                elif isinstance(key[i], slice):
                    new_keys[i] = list(range(*key[i].indices(self.shape[0])))
                else:
                    raise TypeError('Index must be either valid slice or integer.')

        return tuple(new_keys)

    def _get_key_shape(self, key):
        if not isinstance(key, Iterable):
            key = [key]

        out_shape = [s for s in self.shape]

        for i, k in enumerate(key):
            if isinstance(key[i], int) or np.issubdtype(type(key[i]), np.integer):
                out_shape[i] = 1
            elif isinstance(key[i], Iterable):
                out_shape[i] = len(key[i])
            elif isinstance(k, slice):
                stop = k.stop if k.stop is not None else self.shape[i]
                start = k.start if k.start is not None else 0
                step = k.step if k.step is not None else 1
                out_shape[i] = int(np.ceil((stop - start) / step))
            else:
                raise Exception('Index must be either valid slice or integer.')

        return tuple(out_shape)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]

        key = self._get_all_dim_keys(key)

        out_shape = self._get_key_shape(key)
        out_array = Sparse3DArray(shape=out_shape)

        for i, channel_index in enumerate(key[0]):
            if channel_index in self._data:
                out_array[i] = self._data[channel_index][key[1:]]
            else:
                out_array[i] = dok_matrix(out_shape[1:])

        return out_array

    def __iter__(self):
        return iter([self._data[k] for k in sorted(self._data.keys())])

    def __setitem__(self, key, value):
        while len(value.shape) != 3:
            insert_shape = self._get_key_shape(key)
            value = Sparse3DArray(value=value, shape=insert_shape)

        key = self._get_all_dim_keys(key)
        for insert, channel_index in zip(value, key[0]):
            if channel_index not in self._data:
                self._data[channel_index] = dok_matrix(self.shape[1:])
            self._data[channel_index][key[1:]] = insert

    def toarray(self):
        return np.array(
            [self._data[i].toarray() if i in self._data else np.zeros(self.shape[1:]) for i in range(self.shape[0])])
