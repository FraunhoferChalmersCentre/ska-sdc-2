from itertools import starmap
from typing import List
import copy

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
import sparse
import torch
from tqdm.auto import tqdm

# attributes for the dataset generally
from definitions import config

empty_cube_shape = (config['segmentation']['cube_size']['spatial'], config['segmentation']['cube_size']['spatial'],
                    config['segmentation']['cube_size']['freq'])

GLOBAL_ATTRIBUTES = {'dim', 'index', 'scale', 'mean', 'std'}

# attributes only defined in boxes with source
SOURCE_ATTRIBUTES = {'segmentmap', 'allocated_voxels', 'ra', 'dec', 'hi_size', 'line_flux_integral',
                     'central_freq', 'pa', 'i', 'w20'}

# attributes only defined in boxes without source
EMPTY_ATTRIBUTES = {}

# attributes defined for all boxes
COMMON_ATTRIBUTES = {'image', 'position'}

hi_cube_tensor = None
f0 = None
scale = []
mean = []
std = []


def cache_hi_cube(hi_cube_file, min_f0, max_f1):
    global scale, hi_cube_tensor, f0
    f0 = min_f0
    hi_data_fits = fits.getdata(hi_cube_file, ignore_blank=True)
    hi_cube_tensor = torch.tensor(hi_data_fits[min_f0:max_f1].astype(np.float32), dtype=torch.float32).T
    # (max_f1)
    for l in range(min_f0, max_f1):
        if len(scale) <= l:
            percentiles = np.percentile(hi_data_fits[l], [.1, 99.9])
            tmp = np.clip(hi_data_fits[l], *percentiles)
            tmp = (tmp - percentiles[0]) / (percentiles[1] - percentiles[0])

            scale.append(torch.tensor(percentiles, dtype=torch.float32))
            mean.append(torch.tensor(tmp.mean(), dtype=torch.float32))
            std.append(torch.tensor(tmp.std(), dtype=torch.float32))


def get_hi_cube_slice(slices: tuple):
    global hi_cube_tensor, f0
    if hi_cube_tensor is None:
        raise RuntimeError('H1 cube slice not loaded')
    f_slice = slice(slices[-1].start - f0, slices[-1].stop - f0)
    mod_slices = (slices[0], slices[1], f_slice)
    return hi_cube_tensor[mod_slices].clone()


def freq_boundary(central_freq, w20):
    bw = config['constants']['h1_rest_freq'] * w20 / config['constants']['speed_of_light']
    upper_freq = central_freq - bw / 2
    lower_freq = central_freq + bw / 2
    return lower_freq, upper_freq


def prepare_df(df: pd.DataFrame, hi_cube_file: str, coord_keys: List[str], cube_dim: tuple):
    header = fits.getheader(hi_cube_file, ignore_blank=True)
    wcs = WCS(header)

    if config['segmentation']['filtering']['fraction']:
        df = df.sort_values(by=config['segmentation']['filtering']['power_measure'], ignore_index=False, ascending=False)
        df = df.head(int(config['segmentation']['filtering']['fraction'] * len(df)))

    df[coord_keys] = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int32)
    lower_freq, upper_freq = freq_boundary(df['central_freq'].values, df['w20'].values)
    upper_band = wcs.all_world2pix(
        np.concatenate((df[['ra', 'dec']].values, lower_freq.reshape(lower_freq.shape[0], 1)), axis=1), 0).astype(
        np.int32)[:, 2]
    lower_band = wcs.all_world2pix(
        np.concatenate((df[['ra', 'dec']].values, upper_freq.reshape(upper_freq.shape[0], 1)), axis=1), 0).astype(
        np.int32)[:, 2]
    pixel_width_arcsec = abs(header['CDELT1']) * 3600
    major_radius_pixels = np.ceil(df['hi_size'] / (pixel_width_arcsec * 2)).astype(np.int32)

    # Size of each cube with a source
    df['total_x_size'] = cube_dim[0] + major_radius_pixels + 2
    df['total_y_size'] = cube_dim[1] + major_radius_pixels + 2
    df['total_f_size'] = ((upper_band - lower_band) / 2 + cube_dim[2]).astype(np.int32) + 1

    shape = tuple(map(lambda i: header['NAXIS{}'.format(i)], range(1, 4)))

    for i, (p, dim_length) in enumerate(zip(coord_keys, cube_dim)):
        lower = (df[p] - df['total_{}_size'.format(p)])
        upper = (df[p] + df['total_{}_size'.format(p)])

        upper[lower < 0] = np.where(upper[lower < 0] < 2 * dim_length, 2 * dim_length, upper[lower < 0])
        lower[lower < 0] = 0

        lower[upper > shape[i]] = np.where(lower[upper > shape[i]] > shape[i] - 2 * dim_length,
                                           shape[i] - 2 * dim_length, lower[upper > shape[i]])
        upper[upper > shape[i]] = shape[i]

        df['{}0'.format(p)] = lower
        df['{}1'.format(p)] = upper

    df = df.sort_values(by='f1', ignore_index=False)

    return df


def prepare_dicts(**kwargs):
    sources_dict = dict()

    empty_dict = dict()

    for d in (sources_dict, empty_dict):
        d['dim'] = kwargs['dim']
        for attr in COMMON_ATTRIBUTES:
            d[attr] = list()

    for attr in SOURCE_ATTRIBUTES:
        sources_dict[attr] = list()

    return sources_dict, empty_dict


def merge_dict(source_dict, empty_dict, **kwargs):
    merged = dict()

    for attr in GLOBAL_ATTRIBUTES:
        merged[attr] = source_dict[attr]

    for attr in COMMON_ATTRIBUTES:
        merged[attr] = source_dict[attr]
        merged[attr].extend(empty_dict[attr])

    for attr in SOURCE_ATTRIBUTES:
        merged[attr] = source_dict[attr]
        if attr == 'segmentmap':
            merged['segmentmap'].append(torch.zeros(kwargs['cube_dim']))
        else:
            merged[attr].append(torch.tensor(np.nan))

    return merged


def append_common_attributes(data: dict, **kwargs):
    for attr in COMMON_ATTRIBUTES:
        if attr == 'image':
            image = get_hi_cube_slice(kwargs['slices'])
            data[attr].append(image)
        elif attr == 'position':
            position = list(map(lambda s: [s.start, s.stop], kwargs['slices']))
            position_tensor = torch.tensor(position, dtype=torch.int32).T
            data[attr].append(position_tensor)
        else:
            raise NotImplementedError
    return data


def append_source_attributes(data: dict, row: pd.Series, **kwargs):
    for attr in SOURCE_ATTRIBUTES:
        if attr == 'segmentmap':
            data[attr].append(kwargs['segmentmap'])
        elif attr == 'allocated_voxels':
            voxel_indices = torch.tensor(kwargs['allocated_voxels'], dtype=torch.float32)
            data[attr].append(voxel_indices - data['position'][-1][0])
        else:
            data[attr].append(torch.tensor(row[attr]))
    return data


def get_hi_shape(hi_cube_file: str):
    header = fits.getheader(hi_cube_file, ignore_blank=True)
    return np.fromiter(map(lambda i: header['NAXIS{}'.format(i)], range(1, 4)), dtype=np.int32)


def add_boxes(sources_dict: dict, empty_dict: dict, df: pd.DataFrame, hi_cube_file: str, coord_keys: List[str],
              segmentmap: sparse.COO, allocation_dict: dict, n_memory_batches: int):
    global scale
    batch_fetches = int(len(df) / n_memory_batches)

    max_f1 = 0
    prev_max_f1 = 0
    batch_counter = 0

    n_noise_boxes = int(len(allocation_dict) * config['segmentation']['noise_per_source'])

    freq_size = get_hi_shape(hi_cube_file)[-1]
    df['new_loc'] = np.arange(len(df))

    with tqdm(total=len(df)) as pbar:
        pbar.set_description('Sampling dataset')
        for i, row in df.iterrows():

            pbar.update(1)

            if row.f1 > max_f1:
                # Add empty boxes from current cache
                n_empty_batch = int(n_noise_boxes * (max_f1 - prev_max_f1) / freq_size)
                empty_dict = add_noise_boxes(empty_dict, hi_cube_file, segmentmap, n_empty_batch, empty_cube_shape,
                                             prev_max_f1, max_f1)

                batch_counter = 0

                # Update channel spans
                prev_max_f1 = max_f1
                min_f0 = int(df['f0'].iloc[int(row.new_loc):int(row.new_loc) + batch_fetches].min())
                min_f0 = min(min_f0, prev_max_f1)

                max_f1 = int(df['f1'].iloc[int(row.new_loc):int(row.new_loc) + batch_fetches].max())

                if max_f1 - prev_max_f1 < empty_cube_shape[-1]:
                    max_f1 = prev_max_f1 + empty_cube_shape[-1]
                    max_f1 = min(max_f1, get_hi_shape(hi_cube_file)[-1])

                cache_hi_cube(hi_cube_file, min_f0, max_f1)

            if i in allocation_dict.keys():
                batch_counter += 1
                slices = tuple(
                    map(lambda p: slice(int(row['{}0'.format(p)]), int(row['{}1'.format(p)])), coord_keys))
                sources_dict = append_common_attributes(sources_dict, hi_cube_file=hi_cube_file, slices=slices)
                sources_dict = append_source_attributes(sources_dict, row, segmentmap=segmentmap[slices],
                                                        allocated_voxels=allocation_dict[i])

        # Ensure scale is computed for all channels

        n_empty_batch = int(n_noise_boxes * (freq_size - prev_max_f1) / freq_size)
        cache_hi_cube(hi_cube_file, prev_max_f1, freq_size)
        empty_dict = add_noise_boxes(empty_dict, hi_cube_file, segmentmap, n_empty_batch, empty_cube_shape,
                                     prev_max_f1, freq_size)

    sources_dict['index'] = len(sources_dict['image'])
    sources_dict['scale'] = scale
    sources_dict['mean'] = mean
    sources_dict['std'] = std

    return sources_dict, empty_dict


def add_noise_boxes(data: dict, hi_cube_file: str, segmentmap: sparse.COO, n_empty_cubes: int,
                    empty_cube_dim: tuple,
                    min_f: int,
                    max_f: int):
    hi_shape = get_hi_shape(hi_cube_file)
    hi_shape[-1] = max_f - min_f
    counter = 0
    xyf_max = hi_shape - empty_cube_dim
    while counter < n_empty_cubes:
        corner = (np.random.random(3) * xyf_max).astype(np.int32)
        corner[-1] += min_f
        slices = tuple(starmap(lambda c, d: slice(c, c + d), zip(corner, empty_cube_dim)))
        if segmentmap[slices].sum() == 0:
            data = append_common_attributes(data, slices=slices)
            counter += 1
    return data


def split_by_size(df: pd.DataFrame, hi_cube_file: str, segmentmap: sparse.COO, allocation_dict: dict, cube_dim: tuple,
                  n_memory_batches=20, splitsize=60):
    """
    :param df: truth catalogue values of the galaxies
    :param hi_cube_file: filename of H1 data cube
    :param segmentmap: sparse source map
    :param wcs: world coordinate system
    :param cube_dim: dimension of SMALL sampling cube (n*m*o)
    :return: Dictionary with attribute as key
    """

    source_dict, empty_dict = prepare_dicts(dim=cube_dim)
    coord_keys = ['x', 'y', 'f']

    df = prepare_df(df, hi_cube_file, coord_keys, cube_dim)

    source_dict, empty_dict = add_boxes(source_dict, empty_dict, df, hi_cube_file, coord_keys, segmentmap,
                                        allocation_dict, n_memory_batches)

    n_splits = int((len(source_dict['image']) + len(empty_dict['image'])) / splitsize)

    # Init splitted source dicts
    splitted_source_dicts = [dict() for s in range(n_splits)]
    for source_split in splitted_source_dicts:
        for k in SOURCE_ATTRIBUTES.union(COMMON_ATTRIBUTES):
            source_split[k] = list()
        for k in GLOBAL_ATTRIBUTES:
            source_split[k] = source_dict[k]

    # Move source boxes to splitted source dicts
    for k in source_dict.keys():
        if k in GLOBAL_ATTRIBUTES:
            continue

        for i in range(len(source_dict[k])):
            item = source_dict[k].pop(-1)
            splitted_source_dicts[i % len(splitted_source_dicts)][k].append(copy.deepcopy(item))

    for source_split in splitted_source_dicts:
        source_split['index'] = len(source_split['image'])

    # Init splitted empty dicts
    splitted_empty_dicts = [dict() for s in range(n_splits)]
    for empty_split in splitted_empty_dicts:
        for k in COMMON_ATTRIBUTES:
            empty_split[k] = list()

    # Move source boxes to splitted source dicts
    for k in empty_dict.keys():
        if k in GLOBAL_ATTRIBUTES:
            continue

        for i in range(len(empty_dict[k])):
            item = empty_dict[k].pop(-1)
            splitted_empty_dicts[i % len(splitted_empty_dicts)][k].append(item.clone())

    merged_dicts = []
    for source, empty in zip(splitted_source_dicts, splitted_empty_dicts):
        merged_dicts.append(merge_dict(source, empty, cube_dim=tuple(np.array(2) * cube_dim)))

    return merged_dicts
