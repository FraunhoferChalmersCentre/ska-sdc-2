from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
cont = fits.getdata('../data/cont_dev.fits')  # Open the FITS file for reading

fig, axes = plt.subplots(3, 7)

for i in range(3):
    for j in range(7):
        axes[i, j].imshow(np.log(cont[i*7+j] - cont[i*7+j].min() + 1e-10))

plt.show()
