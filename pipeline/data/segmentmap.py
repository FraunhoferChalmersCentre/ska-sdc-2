import os
from itertools import starmap
from typing import Tuple
import pickle

import pandas as pd
from sparse import COO, save_npz, load_npz
from astropy.io.fits.header import Header
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from skimage import draw, transform, filters
from tqdm import tqdm

from pipeline.common import filename
from definitions import config, logger

SPEED_OF_LIGHT = 3e5
ALPHA = .2

segmap_config = config['segmentation']['target']

PADDING = 5


def prepare_df(df: pd.DataFrame, header: Header, do_filter=True, extended_radius=0):
    df = df.copy()
    wcs = WCS(header)
    df[['x', 'y', 'z']] = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)

    df['n_channels'] = header['RESTFREQ'] * df['w20'] / (SPEED_OF_LIGHT * header['CDELT3'])

    pixel_width_arcsec = abs(header['CDELT1']) * 3600
    df['major_radius_pixels'] = df['hi_size'] / (pixel_width_arcsec * 2) + extended_radius
    ratio = np.sqrt((np.cos(np.deg2rad(df.i)) ** 2) * (1 - ALPHA ** 2) + ALPHA ** 2)
    df['minor_radius_pixels'] = ratio * df['major_radius_pixels']

    if do_filter and config['segmentation']['filtering']['fraction']:
        df = df.sort_values(by=config['segmentation']['filtering']['power_measure'], ignore_index=False, ascending=False)
        df = df.head(int(config['segmentation']['filtering']['fraction'] * len(df)))

    half_lengths = (df.major_radius_pixels, df.major_radius_pixels, df.n_channels / 2)

    full_cube_shape = (header['NAXIS1'], header['NAXIS2'], header['NAXIS3'])
    spans = get_spans(full_cube_shape, df[['x', 'y', 'z']], half_lengths)
    df = df.join(spans)

    for i, p in enumerate(['x', 'y', 'z']):
        df = df[(df[f'{p}_upper'] > 0) & (df[f'{p}_lower'] < full_cube_shape[i])]

    return df


def get_allocations(row: pd.Series, full_cube_shape: Tuple):
    cube_shape = tuple(map(lambda p: int(row[f'{p}_upper'] - row[f'{p}_lower']), ['x', 'y', 'z']))
    cross_section = np.zeros(cube_shape[:2], dtype=np.float32)

    # Compute center of galaxy in small cube, with Python pixel convention
    center = tuple(starmap(lambda p, s: p - s - .5, zip(row[['x', 'y']], row[['x_lower', 'y_lower']])))

    # axes = map(np.ceil, row[['major_radius_pixels', 'minor_radius_pixels']])
    axes = row[['major_radius_pixels', 'minor_radius_pixels']]
    # Draw cross-section with base ellipse
    rows, cols = draw.ellipse(*center, *axes, shape=cube_shape[:2])
    allocations = np.array([])
    if len(rows) > 0:
        cross_section[rows, cols] = 1

        # Get span of rows in cross-section containing the ellipse
        minimum_start, maximum_end = rows.min(), rows.max() + 1

        # Span of frequency bands of complete HI cube to be filled, relative to small cube
        channel_span = tuple(
            map(lambda s: np.round(row['z'] + s * row['n_channels'] / 2 - row['z_lower']).astype(np.int32), (-1, 1)))
        channel_span = (max(channel_span[0], 0), min(channel_span[1], cube_shape[2] - 1))

        # Middle row of cross-section
        middle_row = (maximum_end + minimum_start) / 2

        # Middle channel, relative to small cube
        middle_channel = (channel_span[1] + channel_span[0]) / 2

        cube = np.zeros(cube_shape, dtype=np.float32)

        half_width = (maximum_end - minimum_start) / 2

        # Increase of cross-sections rows per frequency channel
        n_rows_per_channel = (maximum_end - minimum_start) / (channel_span[1] - channel_span[0])

        for c in range(*channel_span):
            if maximum_end - minimum_start == 1:
                # Galaxy occupies only a single frequency band
                start = minimum_start
                end = maximum_end
            elif c < middle_channel:
                # For frequency bands of the approaching side of the galaxy
                i = c - channel_span[0]
                start = minimum_start + i * n_rows_per_channel - segmap_config['padding'] * half_width
                end = middle_row + segmap_config['padding'] * half_width
            else:
                # For frequency bands of the receding side of the galaxy
                i = c - middle_channel
                start = middle_row - segmap_config['padding'] * half_width
                end = middle_row + i * n_rows_per_channel + segmap_config['padding'] * half_width

            start = max(start, minimum_start)
            start = np.floor(start).astype(np.int32)

            end = max(end, start + 1)
            end = min(end, maximum_end)
            end = np.ceil(end).astype(np.int32)

            cube[start:end, :, c] = cross_section[start:end]
            cube[:, :, c] = transform.rotate(cube[:, :, c], angle=row.pa - 90, center=center, order=0)

        cube = np.round(cube)

        if cube.sum() > 0:
            min_pos = row[['x_lower', 'y_lower', 'z_lower']].values.astype(np.int32)
            allocations = np.argwhere(cube > 0).astype(np.int32)
            allocations = allocations + min_pos

            for i in range(allocations.shape[1]):
                allocations = allocations[
                    (allocations[:, i] > 0) & (allocations[:, i] < full_cube_shape[i])]
    return allocations


def get_spans(full_cube_shape: Tuple, image_coords: pd.DataFrame, half_length: Tuple):
    columns = [f'{p}_lower' for p in image_coords.columns] + [f'{p}_upper' for p in image_coords.columns]
    spans_df = pd.DataFrame(index=image_coords.index, columns=columns)

    for i in range(len(image_coords.columns)):
        p_name = image_coords.columns[i]
        coords = image_coords[p_name]

        spans_df[f'{p_name}_lower'][(coords - half_length[i]) < 5] = 0
        spans_df[f'{p_name}_lower'][(coords - half_length[i]) >= 5] = np.floor(coords - half_length[i]) - 5

        spans_df[f'{p_name}_upper'][coords + half_length[i] > full_cube_shape[i] - 5] = full_cube_shape[i] - 1
        spans_df[f'{p_name}_upper'][coords + half_length[i] <= full_cube_shape[i] - 5] = np.ceil(
            coords + half_length[i]) + 5

    return spans_df


def gaussian_convolution(small_dense_cube: np.ndarray, header: Header):
    fwhm_arcsec = segmap_config['smoothing_fwhm']
    if fwhm_arcsec == 0.:
        return small_dense_cube
    fwhm_pixel = fwhm_arcsec / (abs(header['CDELT1']) * 3600)
    sigma = fwhm_pixel / (2 * np.sqrt(2 * np.log(2)))
    small_dense_cube = filters.gaussian(small_dense_cube, sigma=sigma)
    small_dense_cube -= small_dense_cube.min()
    small_dense_cube /= small_dense_cube.max()
    small_dense_cube[small_dense_cube < segmap_config['min_value']] = 0.
    return small_dense_cube


def create_from_df(df: pd.DataFrame, header: Header, fill_value=1.):
    df['fill_value'] = fill_value if fill_value is not None else df.index + 1

    full_cube_shape = (header['NAXIS1'], header['NAXIS2'], header['NAXIS3'])

    allocation_dict = {}
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Compute allocations'):
        row_allocations = get_allocations(row, full_cube_shape)
        if len(row_allocations) > 0:
            allocation_dict[i] = row_allocations

    df['n_allocations'] = [len(allocation_dict[i]) if i in allocation_dict.keys() else 0 for i, row in df.iterrows()]
    df = df[df['n_allocations'] > 0]

    all_allocations = np.empty((df['n_allocations'].sum(), 5), dtype=np.int32)

    df = df.sort_values(by='n_allocations', ignore_index=False, ascending=False)

    c = 0
    allocations = dict()
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Creating segmentmap from catalogue'):
        row_allocations = allocation_dict[i]
        indices = np.ravel_multi_index(row_allocations.T, full_cube_shape)

        if fill_value is None:
            collision = np.array([i in allocations.keys() for i in indices])

            for j in indices[collision]:
                row_index = allocations[j]
                collided_index = all_allocations[row_index, 4]
                position = all_allocations[row_index, :3]
                if np.linalg.norm(row[['x', 'y', 'z']] - position) < np.linalg.norm(
                        df.loc[collided_index][['x', 'y', 'z']] - position):
                    all_allocations[row_index, 3] = row.fill_value
                    all_allocations[row_index, 4] = i

            row_allocations = row_allocations[~collision]
            for j, p in enumerate(indices[~collision]):
                allocations[p] = c + j

        all_allocations[c:c + len(row_allocations), :3] = row_allocations
        all_allocations[c:c + len(row_allocations), 3] = row.fill_value
        all_allocations[c:c + len(row_allocations), 4] = i
        c += len(row_allocations)

    all_allocations = all_allocations[:c]
    coo = COO(all_allocations[:, :3].T.astype(np.int32), all_allocations[:, 3].astype(np.int32), shape=full_cube_shape)
    return coo, allocation_dict


def from_processed(file_type: str):
    cube = None
    allocation_dict = None

    segmap_fname = filename.processed.segmentmap(file_type)
    if os.path.exists(segmap_fname):
        cube = load_npz(segmap_fname)
        logger.info('Loaded segmentmap from disk')

    allocation_dict_fname = filename.processed.allocation_dict(file_type)
    if os.path.exists(allocation_dict_fname):
        with open(allocation_dict_fname, 'rb') as f:
            allocation_dict = pickle.load(f)

    return cube, allocation_dict


def save_to_processed(file_type, cube, allocation_dict):
    segmap_fname = filename.processed.segmentmap(file_type)
    save_npz(segmap_fname, cube)

    allocation_dict_fname = filename.processed.allocation_dict(file_type)
    with open(allocation_dict_fname, 'wb') as f:
        pickle.dump(allocation_dict, f)


def create_from_files(file_type: str, regenerate=False, save_to_disk=True):
    cube, allocation_dict = None, None
    if not regenerate:
        cube, allocation_dict = from_processed(file_type)
    if cube is None or allocation_dict is None:
        logger.info('Computing segmentmap from truth catalogue...')
        df = pd.read_csv(filename.data.true(file_type), sep=' ', index_col='id')
        header = fits.getheader(filename.data.sky(file_type), ignore_blank=True)
        df = prepare_df(df, header)
        cube, allocation_dict = create_from_df(df, header)
        if save_to_disk:
            save_to_processed(file_type, cube, allocation_dict)

    return cube, allocation_dict
