import os
import pickle
from operator import itemgetter

import torch
from astropy.io import fits
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from pipeline.segmentation.base import BaseSegmenter
from pipeline.segmentation.training import EquiBatchBootstrapSampler
from pipeline.traversing.traverser import CubeCache
from pipeline.common import filename
from pipeline.common import filehandling
from definitions import config, ROOT_DIR
from pipeline.segmentation.convert2Dto3D import Conv3dConverter
import segmentation_models_pytorch as smp

from pipeline.data.generating import get_hi_shape


def get_data(only_validation=False, robust_validation=False):
    size = config['segmentation']['size']

    directory = filename.processed.dataset(size)
    dataset = filehandling.read_splitted_dataset(directory, limit_files=config['segmentation']['limit_files'])

    import numpy as np
    from pipeline.data import splitting
    from pipeline.data.splitting import TrainingItemGetter, ValidationItemGetter

    random_state = np.random.RandomState(5)
    val_item_getter = ValidationItemGetter() if robust_validation else TrainingItemGetter()
    training_set, validation_set = splitting.train_val_split(dataset, .8, random_state=random_state,
                                                             train_filter=config['segmentation']['filtering'][
                                                                 'training'],
                                                             validation_item_getter=val_item_getter)
    print(len(training_set), len(validation_set))
    if only_validation:
        return validation_set
    else:
        return training_set, validation_set


def get_model():
    modelname = config['segmentation']['model_name']
    model = smp.Unet(encoder_name=modelname, encoder_weights='imagenet', in_channels=1, classes=1,
                     decoder_channels=[256, 128, 64, 32], encoder_depth=4, decoder_use_batchnorm=True)
    # Convert pretrained 2D model to 3D
    Conv3dConverter(model, -1)
    return model


def get_checkpoint_callback():
    model_id = filename.models.new_id()
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          save_top_k=1,
                                          dirpath=filename.models.directory,
                                          filename=config['segmentation']['model_name'] + '-' + str(
                                              model_id) + '-{epoch:02d}-{val_loss:.2f}',
                                          mode='min',
                                          period=10 if config['segmentation']['robust_validation'] else None)
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


def get_equibatch_samplers(training_set, validation_set, epoch_merge=1):
    intensities = np.array([np.prod(a.shape) for a in training_set.get_attribute('image')])
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
                                              n_samples=epoch_merge * len(training_set))

    val_source_bs = int(config['segmentation']['batch_size'] * config['segmentation']['source_fraction']['validation'])
    val_noise_bs = config['segmentation']['batch_size'] - train_source_bs
    val_intensities = np.array([np.prod(a.shape) for a in validation_set.get_attribute('image')])
    val_sampler = EquiBatchBootstrapSampler(validation_set.get_attribute('index'), len(validation_set),
                                            val_source_bs, val_noise_bs, source_bs_start=None,
                                            intensities=val_intensities, n_samples=epoch_merge * len(validation_set),
                                            random_seed=100, batch_size_noise=0)

    return train_sampler, val_sampler
