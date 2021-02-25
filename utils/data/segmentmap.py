import os
from itertools import starmap
from typing import Tuple

import pandas as pd
from sparse import DOK, COO, save_npz, load_npz
from astropy.io.fits.header import Header
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from skimage import draw, transform

from utils import filename
from definitions import config, logger

SPEED_OF_LIGHT = 3e5
ALPHA = .2


def prepare_df(df: pd.DataFrame, header: Header):
    df = df.copy()
    wcs = WCS(header)
    positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)
    df['x'] = positions[:, 0].astype(np.int32)
    df['y'] = positions[:, 1].astype(np.int32)
    df['z'] = positions[:, 2].astype(np.int32)

    df['n_channels'] = header['RESTFREQ'] * df['w20'] / (SPEED_OF_LIGHT * header['CDELT3'])

    pixel_width_arcsec = abs(header['CDELT1']) * 3600
    df['major_radius_pixels'] = df['hi_size'] / (pixel_width_arcsec * 2)
    ratio = np.sqrt((np.cos(np.deg2rad(df.i)) ** 2) * (1 - ALPHA ** 2) + ALPHA ** 2)
    df['minor_radius_pixels'] = ratio * df['major_radius_pixels']

    return df


def dense_cube(row: pd.Series):
    square_side = np.round(2 * max(row.major_radius_pixels, row.minor_radius_pixels)).astype(np.int32)
    if not square_side % 2:
        square_side += 1

    square_side = max(3, square_side)
    cross_section = np.zeros((square_side, square_side))

    base_ellipse = draw.ellipse(int(square_side / 2), int(square_side / 2), row.major_radius_pixels,
                                row.minor_radius_pixels, rotation=np.deg2rad(row.pa))

    cross_section[base_ellipse[0], base_ellipse[1]] = 1.

    # Only use odd numbers of channels
    n_channels = np.round(row.n_channels).astype(np.int32)
    n_channels += 1 if not n_channels % 2 else 0

    return np.repeat(cross_section[np.newaxis, :, :], n_channels, axis=0).T


def get_spans(full_cube_shape: Tuple, small_cube_shape: Tuple, image_coords: pd.Series):
    spans = []

    for p, full_length, small_length in zip(image_coords, full_cube_shape, small_cube_shape):
        if p - int(small_length / 2) < 0:
            dim_span_lower = p
        else:
            dim_span_lower = int(small_length / 2)

        if p + int(small_length / 2) > full_length:
            dim_span_upper = full_length - p
        else:
            dim_span_upper = int(small_length / 2)

        spans.append((dim_span_lower, dim_span_upper + 1))
    return spans


def create_from_df(df: pd.DataFrame, header: Header):
    cube = DOK(shape=(header['NAXIS1'], header['NAXIS2'], header['NAXIS3']))

    for i, row in df.iterrows():
        if row.minor_radius_pixels < config['preprocessing']['min_axis_pixels']:
            continue

        small_dense_cube = dense_cube(row)

        spans = get_spans(cube.shape, small_dense_cube.shape, row[['x', 'y', 'z']].astype(np.int32))

        centers = [int(s / 2) for s in small_dense_cube.shape]

        small_dense_cube = small_dense_cube[tuple(starmap(lambda c, s: slice(c - s[0], c + s[1]), zip(centers, spans)))]

        cube[tuple(starmap(lambda c, s: slice(c - s[0], c + s[1]),
                           zip(row[['x', 'y', 'z']].astype(np.int32), spans)))] = small_dense_cube

    return COO(cube)


def create_from_files(file_type: str, save_to_disk=True, regenerate=False):
    fname = filename.processed.segmentmap(file_type)
    if os.path.exists(fname) and not regenerate:
        cube = load_npz(fname)
        logger.info('Loaded segmentmap from disk')
    else:
        logger.info('Computing segmentmap from truth catalogue...')
        df = pd.read_csv(filename.data.true(file_type), sep=' ')
        header = fits.getheader(filename.data.sky(file_type))
        df = prepare_df(df, header)
        cube = create_from_df(df, header)
        if save_to_disk:
            save_npz(fname, cube)
    return cube
