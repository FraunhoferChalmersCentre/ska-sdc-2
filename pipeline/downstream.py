from typing import Dict

import numpy as np
from astropy.wcs import WCS
from numpy.linalg import LinAlgError
from sofia import linker, parametrisation, readoptions, functions
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from definitions import config, ROOT_DIR

catParNamesBase = (
    "id", "x_geo", "y_geo", "z_geo", "x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "n_pix",
    "snr_min", "snr_max", "snr_sum", "x_p", "y_p", "z_p", "x_n", "y_n", "z_n", "snr_sum_p", "snr_sum_n", "snr_mean",
    "snr_std", "snr_rms", "w20", "w50", "w20_cfd", "w50_cfd", "n_x", "n_y", "n_chan", "n_los", "fill_frac")

catParUnitsBase = (
    "-", "pix", "pix", "chan", "pix", "pix", "chan", "pix", "pix", "pix", "pix", "chan", "chan", "-", "-", "-", "-",
    "pix", "pix", "chan", "pix", "pix", "chan", "-", "-", "-", "-", "-", "chan", "chan", "chan", "chan", "pix", "pix",
    "chan", "-", "-")

catParFormtBase = (
    "%10i", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%7i", "%7i", "%8i",
    "%12.3e", "%12.3e", "%12.3e", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%12.3e", "%12.3e",
    "%12.3e",
    "%12.3e", "%12.3e", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%7i", "%7i", "%7i", "%7i", "%5.3f")


def estimate_axes(mask: np.ndarray, parameters: dict):
    mask_2d = np.clip(mask.sum(axis=0), 0., 1.)
    positions = np.argwhere(mask_2d > 0)
    if len(positions) < 3:
        return None, None
    try:
        pca = IncrementalPCA(n_components=2, batch_size=30).fit(positions)

        return pca.explained_variance_[0], pca.explained_variance_[1]
    except LinAlgError:
        return parameters["merge"]["minSizeX"], parameters["merge"]["minSizeY"]


def estimate_angle(mask: np.ndarray):
    positions = np.argwhere(mask != 0)
    if len(positions) < 4:
        return None
    try:
        pca = IncrementalPCA(n_components=3, batch_size=30).fit(positions)

        angle = np.rad2deg(np.arctan2(pca.components_[0, 1], pca.components_[0, 2]))

        if pca.components_[0, 0] > 0:
            angle += 180

        angle -= 90
        angle = angle % 360

        return angle

    except LinAlgError:
        return 0


def remove_non_reliable(objects, mask, catParNames, catParFormt, catParUnits):
    reliable = list(np.array(objects)[:, 0].astype(int))
    # reliable = list(np.array(objects)[np.array(objects)[:, catParNamesBase.index('snr_sum')] > 0, 0].astype(int))  # select all positive sources

    objects, catParNames, catParUnits, catParFormt = remove_cols(objects, catParNames, catParFormt, catParUnits)
    # Make sure that reliable is sorted
    relList = list(reliable)
    relList.sort()
    reliable = np.array(relList)

    # Remove non-reliable sources in the objects array
    relObjects = []
    for rr in reliable:
        relObjects.append([len(relObjects) + 1] + list(objects[rr - 1]))
    relObjects = np.array(relObjects)
    objects = relObjects

    catParNames = list(catParNames)
    catParNames.insert(1, "id_old")

    catParFormt = list(catParFormt)
    catParFormt.insert(1, "%10i")

    catParUnits = list(catParUnits)
    catParUnits.insert(1, "-")

    # In the mask file
    mask *= -1
    index = 1
    catParNames = np.array(catParNames)
    for rr in reliable:
        objrr = objects[objects[:, 1] == rr][0]
        Xmin = int(objrr[catParNames == "x_min"])
        Ymin = int(objrr[catParNames == "y_min"])
        Zmin = int(objrr[catParNames == "z_min"])
        Xmax = int(objrr[catParNames == "x_max"])
        Ymax = int(objrr[catParNames == "y_max"])
        Zmax = int(objrr[catParNames == "z_max"])
        mask[Zmin:Zmax + 1, Ymin:Ymax + 1, Xmin:Xmax + 1][
            mask[Zmin:Zmax + 1, Ymin:Ymax + 1, Xmin:Xmax + 1] == -rr] = index
        index += 1
    mask[mask < 0] = 0
    catParNames = tuple(catParNames)

    newRel = []
    for i in range(0, len(objects)):
        newRel.append(i + 1)
    reliable = np.array(newRel)
    NRdet = objects.shape[0]

    return relObjects, tuple(catParNames), tuple(catParUnits), tuple(catParFormt), mask


def remove_cols(objects, catParNames, catParFormt, catParUnits):
    objects, catParNames, catParUnits, catParFormt = np.array(objects), list(catParNames), list(catParUnits), list(
        catParFormt)

    removecols = ["fill_frac", "snr_min", "snr_max", "snr_sum", "x_p", "y_p", "z_p", "x_n", "y_n", "z_n", "snr_sum_p",
                  "snr_sum_n", "snr_mean", "snr_std", "snr_rms", "w20", "w50", "w20_cfd", "w50_cfd", "n_pos", "n_neg",
                  "n_x", "n_y"]

    for remcol in removecols:
        if remcol in catParNames:
            index = catParNames.index(remcol)
            del (catParNames[index])
            del (catParUnits[index])
            del (catParFormt[index])
            objects = np.delete(objects, [index], axis=1)

    return [list(item) for item in list(objects)], tuple(catParNames), tuple(catParUnits), tuple(catParFormt)


def estimate_object_properties(cube: np.array, mask: np.array, dilated_mask: np.ndarray, df: pd.DataFrame,
                               parameters: dict):
    df['ell_maj'] = np.nan
    df['ell_min'] = np.nan
    df['ell_pa'] = np.nan
    df['est_flux'] = np.nan
    df['mask_size'] = np.nan
    for i, obj in df.iterrows():
        obj_bounds = tuple([slice(int(max(obj[p + '_min'] - 1, 0)), int(min(obj[p + '_max'] + 1, cube.shape[i])))
                            for i, p in enumerate(['z', 'y', 'x'])])

        object_dilated_mask = np.where(dilated_mask[obj_bounds] == obj.id, 1., 0.)
        object_mask = np.where(mask[obj_bounds] == obj.id, 1., 0.)

        df.loc[i, 'mask_size'] = object_mask.sum()
        df.loc[i, 'est_flux'] = (cube[obj_bounds] * object_dilated_mask).sum()

        angle = estimate_angle(object_mask)
        if angle is not None:
            df.loc[i, 'ell_pa'] = angle

        major, minor = estimate_axes(object_mask, parameters)
        if major is not None:
            df.loc[i, 'ell_maj'] = major
        if minor is not None:
            df.loc[i, 'ell_min'] = minor

    for attr in ['ell_maj', 'ell_min', 'ell_pa', 'est_flux']:
        df[attr] = df[attr].fillna(0)

    df['est_flux'] = df['est_flux'].clip(lower=0)

    return df


def extract_objects(cube: np.ndarray, mask: np.ndarray, parameters: Dict):
    if parameters["merge"]["positivity"]:
        mask[cube < 0.0] = 0

    objects, mask = linker.link_objects(cube.copy(), [], mask.copy(), parameters["merge"]["radiusX"],
                                        parameters["merge"]["radiusY"], parameters["merge"]["radiusZ"],
                                        parameters["merge"]["minSizeX"], parameters["merge"]["minSizeY"],
                                        parameters["merge"]["minSizeZ"], parameters["merge"]["maxSizeX"],
                                        parameters["merge"]["maxSizeY"], parameters["merge"]["maxSizeZ"],
                                        parameters["merge"]["minVoxels"], parameters["merge"]["maxVoxels"],
                                        parameters["merge"]["minFill"], parameters["merge"]["maxFill"],
                                        parameters["merge"]["minIntens"], parameters["merge"]["maxIntens"])

    if len(objects) == 0:
        return mask, mask, pd.DataFrame()

    objects, catParNames, catParUnits, catParFormt, mask = remove_non_reliable(objects, mask, catParNamesBase,
                                                                               catParFormtBase, catParUnitsBase)

    if len(objects) == 0:
        return mask, mask, pd.DataFrame()

    dilated_mask, objects = parametrisation.dilate(cube.copy(), mask.copy(), objects, catParNames, parameters)

    if len(objects) == 0:
        return mask, mask, pd.DataFrame()

    df = pd.DataFrame(objects, columns=catParNames)

    return mask, dilated_mask, df


def compute_challenge_metrics(df, header, position, padding):
    if padding is None:
        padding = np.zeros(len(position[0]))
    wcs = WCS(header)

    if len(df) > 0:

        shift = np.array([pos + pad for pos, pad in zip(position[0], padding)])

        for m in ['min', 'max']:
            for i, p in enumerate(['x', 'y', 'z']):
                df[f'{p}_{m}_s'] = df[f'{p}_{m}'] + shift[i]

        df.loc[:, ['ra', 'dec', 'central_freq']] = wcs.all_pix2world(
            np.array(df[['x_geo', 'y_geo', 'z_geo']] + shift, dtype=np.float32), 0)

        if 'n_chan' in df.columns:
            df.loc[:, 'w20'] = df['n_chan'] * config['constants']['speed_of_light'] * header['CDELT3'] / header[
                'RESTFREQ']

        if 'est_flux' in df.columns:
            df.loc[:, 'line_flux_integral'] = df['est_flux'] * header['CDELT3'] / (
                    np.pi * (7 / 2.8) ** 2 / (4 * np.log(2)))

        if 'ell_maj' in df.columns:
            df.loc[:, 'hi_size'] = df['ell_maj'] * 2.8
            calibration_params = config['downstream']['calibration']['hi_size']
            df.loc[:, 'hi_size'] = calibration_params['intercept'] + calibration_params['coefficient'] * df['hi_size']

        if 'ell_maj' and 'ell_min' in df.columns:
            df.loc[:, 'i'] = np.rad2deg(
                np.arccos(np.sqrt(((df['ell_min'] / df['ell_maj']) ** 2 - .2 ** 2) / (1 - .2 ** 2))))
            df.loc[:, 'i'] = df['i'].fillna(45)

        if 'ell_pa' in df.columns:
            df.loc[:, 'pa'] = df['ell_pa'].fillna(0)

    return df


def filter_df(df: pd.DataFrame):
    for attr in ['x', 'y', 'z']:
        df = df[(df[attr] > df[f'{attr}_min']) & (df[attr] < df[f'{attr}_max'])]

    return df


def parametrise_sources(header, input_cube, mask, position, parameters: Dict = None, padding=None, min_intensity=0,
                        max_intensity=np.inf, return_mask=False):
    if parameters is None:
        parameters = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])
    if mask.sum() == 0.:
        return pd.DataFrame()

    input_cube, mask, position = tuple(
        map(lambda t: t.squeeze().detach().cpu().numpy(), (input_cube, mask, position)))

    obj_mask, dilated_mask, df = extract_objects(input_cube, mask, parameters)

    if len(df) == 0:
        return df

    df = estimate_object_properties(input_cube, obj_mask, dilated_mask, df, parameters)

    df = compute_challenge_metrics(df, header, position, padding)

    df = df[(df['line_flux_integral'] > min_intensity) & (df['line_flux_integral'] < max_intensity)]

    if return_mask:
        return df, obj_mask

    return df
