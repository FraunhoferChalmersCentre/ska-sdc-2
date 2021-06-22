import os
from itertools import starmap
from typing import Tuple
import pickle

import pandas as pd
from sparse import DOK, save_npz, load_npz
from astropy.io.fits.header import Header
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from skimage import draw, transform, filters

from pipeline.common import filename
from definitions import config, logger

SPEED_OF_LIGHT = 3e5
ALPHA = .2

segmap_config = config['segmentation']['target']

PADDING = 5


def prepare_df(df: pd.DataFrame, header: Header):
    df = df.copy()
    wcs = WCS(header)
    df[['x', 'y', 'z']] = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)

    df['n_channels'] = header['RESTFREQ'] * df['w20'] / (SPEED_OF_LIGHT * header['CDELT3'])

    pixel_width_arcsec = abs(header['CDELT1']) * 3600
    df['major_radius_pixels'] = df['hi_size'] / (pixel_width_arcsec * 2)
    ratio = np.sqrt((np.cos(np.deg2rad(df.i)) ** 2) * (1 - ALPHA ** 2) + ALPHA ** 2)
    df['minor_radius_pixels'] = ratio * df['major_radius_pixels']

    return df


def dense_cube(row: pd.Series, spans: Tuple, fill_value=1.):
    cube_shape = tuple(map(lambda s: int(s[1] - s[0]), spans))
    cross_section = np.zeros(cube_shape[:2], dtype=np.float32)

    # Compute center of galaxy in small cube, with Python pixel convention
    center = tuple(starmap(lambda p, s: p - s[0] - .5, zip(row[['x', 'y']], spans[:2])))

    # axes = map(np.ceil, row[['major_radius_pixels', 'minor_radius_pixels']])
    axes = row[['major_radius_pixels', 'minor_radius_pixels']]
    # Draw cross-section with base ellipse
    rows, cols = draw.ellipse(*center, *axes, shape=cube_shape[:2])
    if len(rows) == 0:
        return None
    cross_section[rows, cols] = fill_value

    # Get span of rows in cross-section containing the ellipse
    minimum_start, maximum_end = rows.min(), rows.max() + 1

    # Span of frequency bands of complete HI cube to be filled, relative to small cube
    channel_span = tuple(
        map(lambda s: np.round(row['z'] + s * row['n_channels'] / 2 - spans[-1][0]).astype(np.int32), (-1, 1)))

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
    return cube


def get_spans(full_cube_shape: Tuple, image_coords: pd.Series, half_length: Tuple):
    spans = []

    for p, full_length, half_length in zip(image_coords, full_cube_shape, half_length):
        if p - half_length < - 5:
            dim_span_lower = 0
        else:
            dim_span_lower = np.floor(p - half_length) - 5

        if p + half_length > full_length + 5:
            dim_span_upper = full_length
        else:
            dim_span_upper = np.ceil(p + half_length) + 5

        spans.append((int(dim_span_lower), int(dim_span_upper)))
    return spans


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
    full_cube_shape = (header['NAXIS1'], header['NAXIS2'], header['NAXIS3'])
    cube = DOK(full_cube_shape, dtype=np.float32)

    allocation_dict = dict()

    for i, row in df.iterrows():
        half_lengths = (row.major_radius_pixels, row.major_radius_pixels, row.n_channels / 2)
        spans = get_spans(full_cube_shape, row[['x', 'y', 'z']], half_lengths)

        fill_value_this = fill_value if fill_value is not None else i
        small_dense_cube = dense_cube(row, spans, fill_value_this)

        if small_dense_cube is None or small_dense_cube.sum() == 0:
            continue

        cube_allocations = np.argwhere(small_dense_cube == fill_value_this).astype(np.int32)

        min_pos = np.fromiter(map(lambda s: s[0], spans), dtype=np.int32)

        allocations = []

        for c in cube_allocations:
            full_cube_pos = c + min_pos
            ignore = False
            for pos, shape in zip(full_cube_pos, full_cube_shape):
                if pos < 0 or pos > shape:
                    ignore = True

            if not ignore:
                cube[tuple(full_cube_pos)] = small_dense_cube[tuple(c)]
                allocations.append(full_cube_pos)
            else:
                print(i, full_cube_pos, full_cube_shape)

        allocation_dict[row.id] = np.array(allocations)

    return cube.to_coo(), allocation_dict


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
        df = pd.read_csv(filename.data.true(file_type), sep=' ')
        header = fits.getheader(filename.data.sky(file_type), ignore_blank=True)
        df = prepare_df(df, header)
        cube, allocation_dict = create_from_df(df, header)
        if save_to_disk:
            save_to_processed(file_type, cube, allocation_dict)

    return cube, allocation_dict
