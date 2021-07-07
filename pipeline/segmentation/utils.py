import os
import pickle
from operator import itemgetter

import pandas as pd
import torch
from torch import nn
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

MODEL_INPUT_DIM = np.array([128, 128, 128])
CNN_PADDING = np.array([8, 8, 8])
SOFIA_PADDING = np.array([0, 0, 0])


class DummyModel(nn.Module):
    def forward(self, x):
        return x


def save_input_cube(fits_file, split_point):
    if os.path.isfile(fits_file):
        os.remove(fits_file)
    full_fits_file = filename.data.sky(config['segmentation']['size'])
    data_cube = fits.getdata(full_fits_file)
    header = fits.getheader(full_fits_file)

    wcs = WCS(header)[:, :, split_point:]
    input_cube = SpectralCube(data_cube[:, :, split_point:], wcs, header=header)

    input_cube.write(fits_file)

    return input_cube.shape, input_cube.header


def generate_validation_cube(split_point, device='cuda'):
    fits_file = filename.processed.validation_dataset(config['segmentation']['size'],
                                                      config['segmentation']['model_name']) + '/input_cube.fits'
    cube_shape, header = save_input_cube(fits_file, split_point)
    # EvaluationTraverser
    desired_dim = np.flip(cube_shape) - 2 * CNN_PADDING

    base_segmenter = get_base_segmenter()
    base_segmenter.to(device)
    torch.cuda.empty_cache()

    mbatch = max_batch_size(base_segmenter.model, MODEL_INPUT_DIM, config['traversing']['gpu_memory_max'])

    evaluator = EvaluationTraverser(base_segmenter, fits_file, MODEL_INPUT_DIM, desired_dim, CNN_PADDING,
                                    SOFIA_PADDING, mbatch)

    inner_slices = [slice(p, -p) for p in CNN_PADDING]
    inner_slices.reverse()

    df = pd.read_csv(filename.data.true(config['segmentation']['size']), sep=' ')
    df = prepare_df(df, header, do_filter=False, extended_radius=config['scoring']['extended_radius'])

    segmap_name = filename.processed.validation_dataset(config['segmentation']['size'],
                                                        config['segmentation']['model_name']) + '/segmap.npz'
    if not os.path.isfile(segmap_name):
        segmap, _ = create_from_df(df, header, fill_value=None)
        segmap = segmap.todense().T[inner_slices]
        np.savez(segmap_name, segmap)
    else:
        segmap = np.load(segmap_name)['arr_0'].astype(np.float32)

    df_name = filename.processed.validation_dataset(config['segmentation']['size'],
                                                        config['segmentation']['model_name']) + '/df.txt'
    if not os.path.isfile(df_name):
        df = pd.read_csv(filename.data.true(config['segmentation']['size']), sep=' ')
        df = df.loc[np.unique(segmap).astype(np.int32), :]
        df.to_csv(df_name, sep=' ', index_label='id')
    else:
        df = pd.read_csv(df_name, sep=' ', index_col='id')

    return {'segmentmap': COO.from_numpy(segmap), 'df_true': df, 'evaluator': evaluator, 'header': header}


def get_data(only_validation=False, robust_validation=False):
    size = config['segmentation']['size']

    directory = filename.processed.dataset(size)
    dataset = filehandling.read_splitted_dataset(directory, limit_files=config['segmentation']['limit_files'])

    split_point = int(get_hi_shape(filename.data.sky(size))[0] * .8)
    training_set, validation_set, split_point = splitting.train_val_split(dataset, split_point=split_point)

    if robust_validation:
        validation_set = generate_validation_cube(int(split_point))
    else:
        print(len(training_set), len(validation_set))

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
        val_source_bs = int(config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['validation'])
        val_noise_bs = config['segmentation']['batch_size'] - train_source_bs
        val_intensities = np.array([np.prod(a.shape) for a in validation_set.get_attribute('image')])
        val_sampler = EquiBatchBootstrapSampler(validation_set.get_attribute('index'), len(validation_set),
                                                val_source_bs, val_noise_bs, source_bs_start=None,
                                                intensities=val_intensities, n_samples=epoch_merge * len(validation_set),
                                                random_seed=100, batch_size_noise=0)

    return train_sampler, val_sampler
