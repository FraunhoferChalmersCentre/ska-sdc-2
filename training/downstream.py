import numpy as np
from sofia import readoptions, linker, parametrisation, wcs_coordinates
import pandas as pd

import definitions

default_file = definitions.ROOT_DIR + '/training/SoFiA_parameters.txt'
Parameters = readoptions.readPipelineOptions(default_file)

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

# --------------------------------------------------------------------------------------
# ### ALERT: Temporary list of allowed parameters for reliability calculation.
# ### This is necessary due to a bug in the linker that may produce wrong source parameters
# ### in some cases. NOTE: This list should be replaced with the original one again once the
# ### linker has been fixed. Ensure that catParNames_tmp is replaced with catParNames again
# ### in the for loops below as well!
# --------------------------------------------------------------------------------------
catParNames_tmp = ("n_pix", "n_chan", "n_los", "snr_min", "snr_max", "snr_sum", "snr_mean")


def remove_non_reliable(objects, mask, catParNames, catParFormt, catParUnits):
    reliable = list(np.array(objects)[np.array(objects)[:, catParNamesBase.index('snr_sum')] > 0, 0].astype(
        int))  # select all positive sources

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
    objects, catParNames, catParUnits, catParFormt = np.array(objects), list(catParNames), list(
        catParUnits), list(catParFormt)

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


def extract_sources(cube: np.ndarray, mask: np.ndarray, dunits):
    objects, mask = linker.link_objects(cube, list(), mask, Parameters["merge"]["radiusX"],
                                        Parameters["merge"]["radiusY"], Parameters["merge"]["radiusZ"],
                                        Parameters["merge"]["minSizeX"], Parameters["merge"]["minSizeY"],
                                        Parameters["merge"]["minSizeZ"], Parameters["merge"]["maxSizeX"],
                                        Parameters["merge"]["maxSizeY"], Parameters["merge"]["maxSizeZ"],
                                        Parameters["merge"]["minVoxels"], Parameters["merge"]["maxVoxels"],
                                        Parameters["merge"]["minLoS"], Parameters["merge"]["maxLoS"],
                                        Parameters["merge"]["minFill"], Parameters["merge"]["maxFill"],
                                        Parameters["merge"]["minIntens"], Parameters["merge"]["maxIntens"])

    if len(objects) == 0:
        return pd.DataFrame()

    objects, catParNames, catParUnits, catParFormt, mask = remove_non_reliable(objects, mask, catParNamesBase,
                                                                               catParFormtBase, catParUnitsBase)

    mask, objects = parametrisation.dilate(cube, mask, objects, catParNames, Parameters)

    np_Cube, mask, objects, catParNames, catParFormt, catParUnits = parametrisation.parametrise(
        cube, mask, objects, catParNames, catParFormt, catParUnits, Parameters, dunits)

    return pd.DataFrame(objects, columns=catParNames)
