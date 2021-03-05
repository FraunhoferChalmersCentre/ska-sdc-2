import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from astropy.io import fits


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
    
from utils import filename


def power_transform(t: np.ndarray, base=100, percentile=99.9):
    with tqdm(total=t.shape[0]) as pbar:
        pbar.set_description('Apply power transform')
        for i in range(t.shape[0]):
            pbar.update(1)
            minimum = t[i].min()
            smooth_max = np.percentile(t[i], percentile)
            t[i] = (t[i] - minimum) / (smooth_max  - minimum)
            t[i] = (np.power(base, t[i]) - 1) / base
    return t


def minmax_transform(t: np.ndarray):
    with tqdm(total=t.shape[0]) as pbar:
        pbar.set_description('Apply power transform')
        for i in range(t.shape[0]):
            pbar.update(1)
            lower = np.percentile(t[i], .1)
            upper = np.percentile(t[i], 99.9)
            t[i] = np.clip(t[i], lower, upper)
            t[i] = (t[i] - lower) / (upper  - lower)
    return t


transforms = {'power': power_transform, 'minmax': minmax_transform}
types = ['dev_s', 'dev_l', 'eval']
parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='T', nargs='*', default=transforms.keys(),  help='type of transform to apply ({})'.format(','.join(transforms.keys())))
parser.add_argument('--size', metavar='S', nargs='*', default=types,  help='size of sky to transform') 

args = parser.parse_args()

for t in args.type:
    if t not in transforms.keys():
        raise NotImplementedError()
    for s in args.size:
        header = fits.getheader(filename.data.sky(s))
        data = fits.getdata(filename.data.sky(s))
        transformed = transforms[t](data)
        fits.writeto(filename.data.transformed(s, t), transformed,header=header, overwrite=True)
        print(filename.data.transformed(s, t)  + ' was saved to disk.')
    
        