from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)
sizes = df['line_flux_integral'].copy().values
sizes.sort()
df = df[df['line_flux_integral'] > sizes[int(0. * len(sizes))]]
positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)

plt.style.use('dark_background')
plt.scatter(positions[:, 1], positions[:, 0], c=positions[:, 2], cmap='coolwarm')
plt.colorbar()
plt.axes().set_aspect('equal', 'datalim')
plt.show()