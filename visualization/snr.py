from astropy.wcs import WCS
from astropy.stats import SigmaClip, median_absolute_deviation
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')

wcs = WCS(header)
positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)

HI_rest_frequency = 1.420e9
speed_of_light = 3e5
bandwidth = 30e3
n_bands = HI_rest_frequency * df.w20 / (speed_of_light * bandwidth)
expected_signal = df.line_flux_integral * 2.8**2 / (n_bands * bandwidth * (df.hi_size / 2) ** 2 * np.pi)

starts = positions[:, 2] - n_bands / 2
ends = positions[:, 2] + n_bands / 2

if os.path.exists('stds.npy'):
    stds = []
    for i, band in enumerate(hi_data):
        #clipped = SigmaClip(maxiters=100)(band)
        #std = clipped.std()
        std = median_absolute_deviation(band)
        print(i, std)
        stds.append(std)

    stds = np.array(stds)
    np.save('stds.npy', stds)
else:
    stds = np.load('stds.npy')

noise = np.array([np.mean(stds[int(s):int(e)]) for s, e in zip(starts.values, ends.values)])

fig, axes = plt.subplots(1, 2)
axes[0].hist(expected_signal / noise, bins=30)

snrs = np.linspace(.1, 5, 20)
detectable = np.array([np.sum(expected_signal / noise > snr) for snr in snrs])
axes[1].plot(snrs, detectable)
ax2 = axes[1].twinx()
ax2.plot(snrs, detectable / len(noise))
plt.show()
