import os
from definitions import config


def true_data(types):
    return os.path.join(config['path']['data'], 'sky_dev_{}truthcat.txt'.format(types[-1].replace('s', '')))


def sky_data(types):
    if types == 'eval':
        name = os.path.join(config['path']['data'], 'sky_eval.fits')
    else:
        name = os.path.join(config['path']['data'], 'sky_{}dev.fits'.format(types[-1].replace('s', '')))
    return name


def cont_data(types):
    if types == 'eval':
        name = os.path.join(config['path']['data'], 'cont_eval.fits')
    else:
        name = os.path.join(config['path']['data'], 'cont_{}dev.fits'.format(types[-1].replace('s', '')))
    return name


def readme(types):
    if types == 'eval':
        name = os.path.join(config['path']['data'], 'README_eval.txt')
    else:
        name = os.path.join(config['path']['data'], 'README_{}dev.txt'.format(types[-1].replace('s', '')))
    return name
