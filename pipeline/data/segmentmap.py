import os
from typing import Tuple
import pickle

import pandas as pd
from sparse import COO, save_npz, load_npz
from astropy.io.fits.header import Header
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from skimage import draw
from tqdm import tqdm

from pipeline.common import filename
from definitions import config, logger

SPEED_OF_LIGHT = 3e5
ALPHA = .2

segmap_config = config['segmentation']['target']

PADDING = 5
EPS = 1e-10


def prepare_df(df: pd.DataFrame, header: Header, do_filter=True, extended_radius=0):
    df = df.copy()
    wcs = WCS(header)
    df[['x', 'y', 'z']] = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)

    df['n_channels'] = header['RESTFREQ'] * df['w20'] / (SPEED_OF_LIGHT * header['CDELT3'])

    pixel_width_arcsec = abs(header['CDELT1']) * 3600
    df['major_radius_pixels'] = df['hi_size'] / (pixel_width_arcsec * 2) + extended_radius
    df['minor_radius_pixels'] = df['major_radius_pixels'] * np.sqrt(
        np.cos(np.deg2rad(df['i'])) ** 2 + (ALPHA ** 2) * np.sin(np.deg2rad(df['i'])) ** 2)

    if do_filter and config['segmentation']['filtering']['fraction']:
        df = df.sort_values(by=config['segmentation']['filtering']['power_measure'], ignore_index=False,
                            ascending=False)
        df = df.head(int(config['segmentation']['filtering']['fraction'] * len(df)))

    half_lengths = (df.major_radius_pixels, df.major_radius_pixels, df.n_channels / 2)

    full_cube_shape = (header['NAXIS1'], header['NAXIS2'], header['NAXIS3'])
    spans = get_spans(full_cube_shape, df[['x', 'y', 'z']], half_lengths)
    df = df.join(spans)

    for i, p in enumerate(['x', 'y', 'z']):
        df = df[(df[f'{p}_upper'] > 0) & (df[f'{p}_lower'] < full_cube_shape[i])]

    return df


def get_vz_range(x, y, p, row):
    p2 = p ** 2
    d = np.linalg.norm(np.stack((y, x)), axis=0)
    d2 = np.square(d)

    s = np.zeros(len(x))
    s[p < d] = np.sqrt(d2[p < d] - p2)

    xb = s * x - p * y
    yb = s * y + p * x

    xf = s * x + p * y
    yf = s * y - p * x

    def shift(xs, ys): return xs / np.linalg.norm(np.stack((xs, ys / np.cos(row.i))), axis=0)

    def backward_shift(cond): return shift(xb[cond], yb[cond])

    def forward_shift(cond): return shift(xf[cond], yf[cond])

    lower = np.zeros(len(x))

    pe = p + EPS

    lower[(x < pe) & (y < 0)] = forward_shift((x < pe) & (y < 0))
    lower[(x < pe) & (0 <= y)] = backward_shift((x < pe) & (0 <= y))
    lower[pe <= x] = (x[pe <= x] - p) / row.major_radius_pixels
    lower[(d < pe) | ((x < 0) & (np.abs(y) < pe))] = -1

    upper = np.zeros(len(x))
    upper[x < -pe] = (x[x < -pe] + p) / row.major_radius_pixels
    upper[(-pe <= x) & (y < 0)] = backward_shift((-pe <= x) & (y < 0))
    upper[(-pe <= x) & (0 <= y)] = forward_shift((-pe <= x) & (0 <= y))
    upper[(d < pe) | ((0 <= x) & (np.abs(y) < pe))] = 1

    return lower, upper


def full_cylinder_vz(x, y, p, row):
    return np.repeat(-1, len(x)), np.repeat(1, len(y))


def get_allocations(row: pd.Series, full_cube_shape: Tuple, padding=0, vz_range_method=get_vz_range):
    axes = row[['major_radius_pixels', 'minor_radius_pixels']].values.astype(np.float32)
    rotation_angle = np.deg2rad(row.pa - 90)
    if np.pi < rotation_angle:
        rotation_angle = rotation_angle - 2 * np.pi
    # Draw cross-section with base ellipse
    local_xs, local_ys = draw.ellipse(0, 0, axes[0] + 1 + padding, axes[1] + axes[1] / axes[0] + padding,
                                      rotation=rotation_angle)
    allocations = []
    if len(local_ys) > 0:
        rot_xs = np.cos(-rotation_angle) * local_xs - np.sin(-rotation_angle) * local_ys
        rot_ys = np.sin(-rotation_angle) * local_xs + np.cos(-rotation_angle) * local_ys
        lower, upper = vz_range_method(rot_xs, rot_ys, padding, row)

        xs = np.round(row.x + local_xs).astype(np.int32)
        ys = np.round(row.y + local_ys).astype(np.int32)

        pos, indices = np.unique(np.stack((xs, ys)), axis=1, return_index=True)
        lower_channel = np.round(row.z + row.n_channels * lower / 2).astype(np.int32)
        upper_channel = np.round(row.z + row.n_channels * upper / 2).astype(np.int32)
        for x, y, l, u in zip(xs[indices], ys[indices], lower_channel[indices], upper_channel[indices]):
            zs = np.arange(l, u + 1)
            allocations.extend([[x, y, z] for z in zs.astype(np.int32)])

    allocations = np.array(allocations)
    if len(allocations) > 0:
        for i in range(allocations.shape[1]):
            allocations = allocations[(allocations[:, i] > 0) & (allocations[:, i] < full_cube_shape[i])]
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


def create_from_df(df: pd.DataFrame, header: Header, fill_value=1., padding=0, vz_range_method=get_vz_range):
    df['fill_value'] = fill_value if fill_value is not None else df.index + 1

    full_cube_shape = (header['NAXIS1'], header['NAXIS2'], header['NAXIS3'])

    allocation_dict = {}
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc='Compute allocations'):
        row_allocations = get_allocations(row, full_cube_shape, padding, vz_range_method)
        if len(row_allocations) > 0:
            allocation_dict[i] = row_allocations

    df['n_allocations'] = [len(allocation_dict[i]) if i in allocation_dict.keys() else 0 for i, row in df.iterrows()]

    all_allocations = np.empty((int(df['n_allocations'].sum()), 5), dtype=np.int32)

    df = df.sort_values(by='n_allocations', ignore_index=False, ascending=False)

    c = 0
    allocations = dict()
    for i, row in tqdm(df[0 < df.n_allocations].iterrows(), total=sum(0 < df.n_allocations),
                       desc='Creating segmentmap from catalogue'):

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


def create_from_files(file_type: str, regenerate=False, save_to_disk=True, padding=0):
    cube, allocation_dict = None, None
    if not regenerate:
        cube, allocation_dict = from_processed(file_type)
    if cube is None or allocation_dict is None:
        logger.info('Computing segmentmap from truth catalogue...')
        df = pd.read_csv(filename.data.true(file_type), sep=' ', index_col='id')
        header = fits.getheader(filename.data.sky(file_type), ignore_blank=True)
        df = prepare_df(df, header)
        cube, allocation_dict = create_from_df(df, header, padding=padding)
        if save_to_disk:
            save_to_processed(file_type, cube, allocation_dict)

    return cube, allocation_dict
