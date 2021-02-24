from typing import Dict

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io.fits import header
import sparse


def create_datacube_set_dict(df: pd.DataFrame, hi_data: np.ndarray, segmentmap: sparse.COO, wcs: WCS, head: header,
                             prob_galaxy: float, cube_dim: np.array, empty_cube_dim: np.array) -> Dict:
    """
    :param df: truth catalogue values of the galaxies
    :param hi_data: h1 data cube
    :param segmentmap: sparse source map
    :param wcs: world coordinate system
    :param head: header information from data file
    :param prob_galaxy: proportion of data points containing a galaxy
    :param cube_dim: dimension of SMALL sampling cube (n*m*o)
    :param empty_cube_dim: dimension of BIG cube to sample from (p*q*r)
    :return: Dictionary with attribute as key
    """

    data = dict()
    positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int32)
    lower_freq, upper_freq = freq_boundary(df['central_freq'].values, df['w20'].values)
    upper_band = wcs.all_world2pix(
        np.concatenate((df[['ra', 'dec']].values, lower_freq.reshape(lower_freq.shape[0], 1)), axis=1), 0).astype(
        np.int32)[:, 2]
    lower_band = wcs.all_world2pix(
        np.concatenate((df[['ra', 'dec']].values, upper_freq.reshape(upper_freq.shape[0], 1)), axis=1), 0).astype(
        np.int32)[:, 2]
    pixel_width_arcsec = abs(head['CDELT1']) * 3600
    major_radius_pixels = np.ceil(df['hi_size'] / (pixel_width_arcsec * 2)).astype(np.int32)

    # Size of each cube with a source
    total_x_size = cube_dim[0] - 1 + major_radius_pixels
    total_y_size = cube_dim[1] - 1 + major_radius_pixels
    total_f_size = ((upper_band - lower_band) / 2 + cube_dim[2]).astype(np.int32)

    # Number of cubes
    n_sources = df.shape[0]

    # Create output dict
    data['image'] = list()
    data['segmentmap'] = list()
    data['position'] = list()
    data['dim'] = cube_dim
    data['ra'] = list()
    data['dec'] = list()
    data['hi_size'] = list()
    data['line_flux_integral'] = list()
    data['central_freq'] = list()
    data['pa'] = list()
    data['i'] = list()
    data['w20'] = list()

    x_max = hi_data.shape[0]
    y_max = hi_data.shape[1]
    f_max = hi_data.shape[2]

    # Sources Data
    for i in range(n_sources):
        x0, x1 = max(positions[i][0] - total_x_size[i], 0), min(positions[i][0] + total_x_size[i], x_max)
        y0, y1 = max(positions[i][1] - total_y_size[i], 0), min(positions[i][1] + total_y_size[i], y_max)
        f0, f1 = max(positions[i][2] - total_f_size[i], 0), min(positions[i][2] + total_f_size[i], f_max)

        segment = segmentmap[x0:x1, y0:y1, f0:f1].todense()
        if np.sum(segment) > 0:
            data['image'].append(hi_data[x0:x1, y0:y1, f0:f1])
            data['segmentmap'].append(segment)
            data['position'].append(np.array([[x0, y0, f0], [x1, y1, f1]]))
            data['ra'].append([df.loc[i, 'ra']])
            data['dec'].append([df.loc[i, 'dec']])
            data['hi_size'].append([df.loc[i, 'hi_size']])
            data['line_flux_integral'].append([df.loc[i, 'line_flux_integral']])
            data['central_freq'].append([df.loc[i, 'central_freq']])
            data['pa'].append([df.loc[i, 'pa']])
            data['i'].append([df.loc[i, 'i']])
            data['w20'].append([df.loc[i, 'w20']])

    data['index'] = len(data['image'])
    n_empty_cubes = int((1 - prob_galaxy) / prob_galaxy * data['index'])

    # Empty Data
    counter = 0
    xyf_max = [x_max - empty_cube_dim[0], y_max - empty_cube_dim[1], f_max - empty_cube_dim[2]]
    while counter < n_empty_cubes:
        corner = (np.random.random(3) * xyf_max).astype(np.int32)
        o_corner = corner + empty_cube_dim
        segment = segmentmap[corner[0]:o_corner[0], corner[1]:o_corner[1], corner[2]:o_corner[2]].todense()
        if np.sum(segment) == 0:
            counter += 1
            data['image'].append(hi_data[corner[0]:o_corner[0], corner[1]:o_corner[1], corner[2]:o_corner[2]])
            data['position'].append(np.array([corner, o_corner]))

    data['segmentmap'].append(np.zeros(data['dim']))
    data['ra'].append([np.nan])
    data['dec'].append([np.nan])
    data['hi_size'].append([np.nan])
    data['line_flux_integral'].append([np.nan])
    data['central_freq'].append([np.nan])
    data['pa'].append([np.nan])
    data['i'].append([np.nan])
    data['w20'].append([np.nan])
    return data


def freq_boundary(central_freq, w20):
    rest_freq = 1.420e9
    c = 3e5
    bw = rest_freq * w20 / c
    upper_freq = central_freq - bw / 2
    lower_freq = central_freq + bw / 2
    return lower_freq, upper_freq

