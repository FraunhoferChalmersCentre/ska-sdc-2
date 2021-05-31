import torch
from itertools import starmap
from pipeline.segmentation.base import BaseSegmenter
import numpy as np


def coordinates_expand(dim, upper_left, cube_shape, padding):
    coord_start = ((dim - 2 * padding) * upper_left).astype(np.int32)
    coord_end = (coord_start + dim).astype(np.int32)

    keep = True

    for i, (e, s, d) in enumerate(zip(coord_end, cube_shape, dim)):
        if e < s < e + d:
            coord_end[i] = s
        elif e > s:
            keep = False

    return coord_start, coord_end, keep


def coordinates(dim, upper_left, cube_shape, padding):
    coord_start = ((dim - 2 * padding) * upper_left).astype(np.int32)
    coord_end = (coord_start + dim).astype(np.int32)

    ext_padding = np.zeros(3, dtype=np.int32)

    for i, (e, s) in enumerate(zip(coord_end, cube_shape)):
        ext_padding[i] = e - min(e, s)
        coord_end[i] = min(e, s)

    return coord_start, coord_end, ext_padding


def _partition_indexing(cube_shape, dim, padding, max_batch_size=None):
    if np.any(dim < 2 * padding):
        raise ValueError('Padding has to be less than half dimension')
    effective_shape = tuple(starmap(lambda s, p: s - 2 * p, zip(dim, padding)))
    patches_each_dim = tuple(starmap(lambda e, c, p: np.ceil((c - 2 * p) / e),
                                     zip(effective_shape, cube_shape, padding)))

    meshes = np.meshgrid(*map(np.arange, patches_each_dim))
    upper_lefts = np.stack(list(map(np.ravel, meshes)))
    n_evaluations = upper_lefts.shape[1]
    if max_batch_size is not None:
        batch_size = min(max_batch_size, n_evaluations)
    else:
        batch_size = n_evaluations

    n_index = int(np.ceil(float(n_evaluations) / batch_size))
    indexes_partition = np.array_split(np.arange(n_evaluations), n_index)
    return upper_lefts, indexes_partition


def partition_overlap(cube_shape, dim, padding, max_batch_size=None):
    upper_lefts, indexes_partition = _partition_indexing(cube_shape, dim, padding, max_batch_size)

    overlap_slices_partition = list()
    overlaps_partition = list()
    for indexes in indexes_partition:
        overlap_slices = list()
        overlaps = list()
        for i, index in enumerate(indexes):
            c_start, c_end, overlap = coordinates(dim, upper_lefts[:, index], cube_shape, padding)
            overlaps.append(overlap)
            overlap_slices.append(list(starmap(lambda s, e, o: slice(s - o, e), zip(c_start, c_end, overlap))))
        overlap_slices_partition.append(overlap_slices)
        overlaps_partition.append(overlaps)
    return overlap_slices_partition, overlaps_partition


def partition_expanding(cube_shape, dim, padding):
    upper_lefts, indexes_partition = _partition_indexing(cube_shape, dim, padding)

    exp_slices_partition = list()
    for indexes in indexes_partition:
        exp_slices = list()
        for i, index in enumerate(indexes):
            c_start, c_end, keep = coordinates_expand(dim, upper_lefts[:, index], cube_shape, padding)
            if keep:
                exp_slices.append(list(starmap(lambda s, e: slice(s, e), zip(c_start, c_end))))
        if len(exp_slices) > 0:
            exp_slices_partition.append(exp_slices)
    return exp_slices_partition


def _slice_add(slice_1, slice_2):
    return [slice(s1.start + s2.start + s2.stop, s1.stop + 2 * s2.stop) for s1, s2 in zip(slice_1, slice_2)]


def cube_evaluation(cube, dim, padding, position, overlap_slices, overlaps, model: BaseSegmenter):
    model_input = torch.empty(len(overlap_slices), 1, *dim)
    frequency_channels = torch.empty((len(overlap_slices), 2))

    padding_slices = list()

    for i, ovs in enumerate(overlap_slices):
        model_input[i, 0] = cube[ovs]
        frequency_channels[i, :] = torch.tensor([position[0, -1] + ovs[-1].start,
                                                 position[0, -1] + ovs[-1].stop])
        padd_slices = [slice(int(p + o), int(- p)) for o, p in zip(overlaps[i], padding)]
        padding_slices.append(padd_slices)

    model.eval()
    with torch.no_grad():
        model_out = model(model_input, frequency_channels)
    out = torch.empty(len(overlap_slices), 1, *dim)
    out[:, :, :, :, :] = model_out.detach().clone()
    del model_out
    torch.cuda.empty_cache()

    outputs = [m[0][p] for m, p in zip(out, padding_slices)]
    efficient_slices = [_slice_add(s, p) for s, p in zip(overlap_slices, padding_slices)]
    return outputs, efficient_slices


def connect_outputs(cube, outputs, efficient_slices, padding):
    eval_shape = tuple(starmap(lambda s, p: int(s - 2 * p), zip(cube.shape, padding)))
    eval_cube = torch.empty(eval_shape)
    for out, sli in zip(outputs, efficient_slices):
        eval_cube[sli] = out
    return eval_cube
