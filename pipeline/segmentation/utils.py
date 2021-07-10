import os
import pickle
from operator import itemgetter

import pandas as pd
import torch
from astropy.io import fits
from astropy.wcs import WCS
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from spectral_cube import SpectralCube
from sparse import COO

from pipeline.data import splitting
from pipeline.data.segmentmap import create_from_df, prepare_df
from pipeline.segmentation.base import BaseSegmenter
from pipeline.segmentation.training import EquiBatchBootstrapSampler
from pipeline.traversing.memory import max_batch_size
from pipeline.traversing.traverser import CubeCache, EvaluationTraverser
from pipeline.common import filename
from pipeline.common import filehandling
from definitions import config, ROOT_DIR
from pipeline.segmentation.convert2Dto3D import Conv3dConverter
import segmentation_models_pytorch as smp

from pipeline.data.generating import get_hi_shape

MODEL_INPUT_DIM = np.array([64, 64, 64])
CNN_PADDING = np.array([8, 8, 8])
SOFIA_PADDING = np.array([10, 10, 75])
DESIRED_DIM = np.array([739, 739, 512])


def generate_validation_input_cube(val_dataset_path):
    fits_file = val_dataset_path + '/input_cube.fits'
    if not os.path.isfile(fits_file):
        validation_split = config['segmentation']['validation']['split']
        split_point = int(get_hi_shape(filename.data.sky(config['segmentation']['size']))[0] * validation_split)
        full_fits_file = filename.data.sky(config['segmentation']['size'])
        data_cube = fits.getdata(full_fits_file, ignore_blank=True)
        header = fits.getheader(full_fits_file, ignore_blank=True)

        reduction_split = config['segmentation']['validation']['reduction']
        reduction_point = int(get_hi_shape(filename.data.sky(config['segmentation']['size']))[1] * reduction_split)

        wcs = WCS(header)[:, reduction_point:, split_point:]
        input_cube = SpectralCube(data_cube[:, reduction_point:, split_point:], wcs, header=header)

        input_cube.write(fits_file)

        return input_cube.shape, input_cube.header
    else:
        header = fits.getheader(fits_file, ignore_blank=True)
        shape = np.fromiter(map(lambda i: header['NAXIS{}'.format(i)], range(3, 0, -1)), dtype=np.int32)
        return shape, header


def generate_validation_segmentmap(val_dataset_path, header):
    inner_slices = [slice(p, -p) for p in CNN_PADDING]
    inner_slices.reverse()

    df = pd.read_csv(filename.data.true(config['segmentation']['size']), sep=' ')
    df = prepare_df(df, header, do_filter=False, extended_radius=config['scoring']['extended_radius'])

    segmap_name = val_dataset_path + '/segmentmap.npz'
    if not os.path.isfile(segmap_name):
        segmentmap, _ = create_from_df(df, header, fill_value=None)
        segmentmap = segmentmap.todense().T[inner_slices]
        np.savez(segmap_name, segmentmap)
    else:
        segmentmap = np.load(segmap_name)['arr_0'].astype(np.float32)
    return segmentmap


def generate_validation_catalogue(val_dataset_path, segmentmap):
    df_name = val_dataset_path + '/df.txt'
    if not os.path.isfile(df_name):
        df = pd.read_csv(filename.data.true(config['segmentation']['size']), sep=' ')
        df = df.loc[np.unique(segmentmap).astype(np.int32), :]
        df.to_csv(df_name, sep=' ', index_label='id')
    else:
        df = pd.read_csv(df_name, sep=' ', index_col='id')

    return df


def generate_validation_set(device='cuda'):
    val_dataset_path = filename.processed.validation_dataset(config['segmentation']['size'],
                                                        100 * config['segmentation']['validation']['reduction'])

    cube_shape, header = generate_validation_input_cube(val_dataset_path)

    # EvaluationTraverser
    desired_dim = np.array([min(d, s - 2 * c) for d, s, c in zip(DESIRED_DIM, np.flip(cube_shape), CNN_PADDING)])

    base_segmenter = get_base_segmenter()
    base_segmenter.to(device)
    torch.cuda.empty_cache()

    mbatch = max_batch_size(base_segmenter.model, MODEL_INPUT_DIM, config['traversing']['gpu_memory_max'])

    fits_file = val_dataset_path + '/input_cube.fits'

    evaluator = EvaluationTraverser(base_segmenter, fits_file, MODEL_INPUT_DIM, desired_dim, CNN_PADDING,
                                    SOFIA_PADDING, mbatch)

    segmentmap = generate_validation_segmentmap(val_dataset_path, header)

    df_true = generate_validation_catalogue(val_dataset_path, segmentmap)

    wcs = WCS(header)[[slice(c,-c) for c in CNN_PADDING]]

    return {'segmentmap': COO.from_numpy(segmentmap), 'df_true': df_true, 'evaluator': evaluator, 'wcs': wcs}


def get_data(only_validation=False, robust_validation=False):
    size = config['segmentation']['size']

    split_point = int(get_hi_shape(filename.data.sky(size))[0] * .8)

    if robust_validation:
        robust_validation_set = generate_validation_set()

    directory = filename.processed.dataset(size)
    dataset = filehandling.read_splitted_dataset(directory, limit_files=config['segmentation']['limit_files'])

    training_set, simple_validation_set, split_point = splitting.train_val_split(dataset, split_point=split_point)

    if robust_validation:
        validation_set = robust_validation_set
    else:
        validation_set = simple_validation_set

    if only_validation:
        return validation_set
    else:
        return training_set, validation_set


def get_model():
    modelname = config['segmentation']['model_name']
    model = smp.Unet(encoder_name=modelname, encoder_weights='imagenet', in_channels=1, classes=1,
                     decoder_use_batchnorm=True)
    # Convert pretrained 2D model to 3D
    Conv3dConverter(model, -1, (32, 1, 32, 32, 32))
    return model


def get_checkpoint_callback():
    model_id = filename.models.new_id()
    if config['segmentation']['robust_validation']:
        checkpoint_callback = ModelCheckpoint(monitor='score',
                                              save_top_k=10,
                                              dirpath=filename.models.directory,
                                              filename=config['segmentation']['model_name'] + '-' + str(
                                                  model_id) + '-{epoch:02d}-{score:.2f}',
                                              mode='max',
                                              period=10)
    else:
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              save_top_k=10,
                                              dirpath=filename.models.directory,
                                              filename=config['segmentation']['model_name'] + '-' + str(
                                                  model_id) + '-{epoch:02d}-{val_loss:.2f}',
                                              mode='min',
                                              period=1)
    return checkpoint_callback


def get_state_dict(file):
    state_dict = torch.load(ROOT_DIR + '/saved_models/{}'.format(file))['state_dict']
    state_dict = {k: v for k, v in state_dict.items() if k.startswith('model')}
    return state_dict


def get_random_vis_id(dataset, min_allocated=300, random_state=np.random.RandomState()):
    vis_id = None
    shape = get_hi_shape(filename.data.sky(config['segmentation']['size']))

    pos = dataset.get_attribute('position')
    alloc = dataset.get_attribute('allocated_voxels')
    while vis_id is None:
        random_id = random_state.randint(0, dataset.get_attribute('index'))
        candidate = True
        for r in pos[random_id].squeeze():

            for c, s in zip(r, shape):
                if torch.eq(c, 0) or torch.eq(c, s):
                    candidate = False

        if candidate and alloc[random_id].shape[0] > min_allocated:
            vis_id = random_id

    return vis_id


def get_statistics():
    if os.path.isfile(ROOT_DIR + "/saved_models/statistic.p"):
        scale, mean, std = itemgetter('scale', 'mean', 'std')(
            pickle.load(open(ROOT_DIR + "/saved_models/statistic.p", "rb")))
    else:
        header = fits.getheader(filename.data.sky(config['segmentation']['size']), ignore_blank=True)
        cc = CubeCache(filename.data.sky(config['segmentation']['size']))
        scale, mean, std = cc.comp_statistics(np.arange(header['NAXIS3']))
        pickle.dump({'scale': scale, 'mean': mean, 'std': std}, open(ROOT_DIR + "/saved_models/statistic.p", 'wb'))

    return scale, mean, std


def get_base_segmenter():
    scale, mean, std = get_statistics()
    model = get_model()
    return BaseSegmenter(model, scale, mean, std)


def get_equibatch_samplers(training_set, validation_set, robust_validation, epoch_merge=1):
    intensities = np.ones(len(training_set))
    train_source_bs_end = int(
        config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['training_end'])
    train_source_bs = int(
        config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['training_end'])
    train_noise_bs = config['segmentation']['batch_size'] - train_source_bs_end
    train_source_bs_start = int(
        config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['training_start'])
    train_sampler = EquiBatchBootstrapSampler(training_set.get_attribute('index'), len(training_set),
                                              train_source_bs, train_noise_bs, source_bs_start=train_source_bs_start,
                                              intensities=intensities,
                                              n_samples=epoch_merge * len(training_set),
                                              anneal_interval=config['segmentation']['anneal_interval'])

    if robust_validation:
        val_sampler = None
    else:
        val_source_bs = int(
            config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['validation'])
        val_noise_bs = config['segmentation']['batch_size'] - train_source_bs
        val_intensities = np.array([np.prod(a.shape) for a in validation_set.get_attribute('image')])
        val_sampler = EquiBatchBootstrapSampler(validation_set.get_attribute('index'), len(validation_set),
                                                val_source_bs, val_noise_bs, source_bs_start=None,
                                                intensities=val_intensities,
                                                n_samples=epoch_merge * len(validation_set),
                                                random_seed=100, batch_size_noise=0)

    return train_sampler, val_sampler
