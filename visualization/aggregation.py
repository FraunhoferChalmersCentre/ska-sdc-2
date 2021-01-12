from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)

positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)

# for i in range(10):
#    sample = np.random.randint(0, im_data.shape[1])
#    plt.plot(np.arange(len(im_data)), im_data[:, sample, sample], alpha=.1)
sizes = df['line_flux_integral'].copy().values
sizes.sort()
#freq = df['central_freq'][df['line_flux_integral'] == sizes[-1]].values[0]
freq = np.random.choice(df[df['line_flux_integral'] > sizes[int(90 * len(sizes) / 100)]]['central_freq'])

d = positions[df['central_freq'] == freq].flatten()
band = d[2]
padding_w = min(hi_data.shape[1] - d[1], d[1], 100)
padding_h = min(hi_data.shape[2] - d[0], d[0], 20)
cropped = hi_data[:, d[1] - padding_w:d[1] + padding_w, d[0] - padding_h:d[0] + padding_h]
# d = positions[np.abs(positions[:, 2] - band) < 2]

n_bands = 10
fig, axes = plt.subplots(1, n_bands, sharex=True, sharey=True)
averaging_width = 100
for i, ax in enumerate(axes):
    channel = np.mean(cropped[band - 10*i: band + 10*i + 1, :, :], axis=0)
    ax.imshow(channel)
    # ax.scatter([100], d[:, 1], c='r', s=5, alpha=.5)
plt.show()

plt.figure()
span_center = 200
spectrum = cropped[band - span_center:band + span_center, padding_w, padding_h]
plt.plot(np.arange(band - span_center, band + span_center), spectrum)
plt.vlines(band, spectrum.min(), spectrum.max(), color='r')
plt.show()
