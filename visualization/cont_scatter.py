from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
df = df.sort_values(by=['line_flux_integral'], ignore_index=True, ascending=False)

cont = fits.getdata('../data/cont_dev.fits')
header = fits.getheader('../data/cont_dev.fits')


cont = cont.sum(axis=0)
#cont = cont.clip(cont.min(), np.percentile(cont.flatten(), 99.9))

wcs = WCS(header)
positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)

cont_flux = []
padding = 2
min_flux = 5

for i, row in df.iterrows():
    print(i)
    x = int(positions[i, 0])
    y = int(positions[i, 1])
    flux_sum = cont[x - padding:x + padding, y - padding:y + padding].sum()
    cont_flux.append(flux_sum)

cont_flux = np.array(cont_flux)
plt.scatter(df.line_flux_integral[df.line_flux_integral > min_flux], cont_flux[df.line_flux_integral > min_flux], alpha=.4)
plt.show()

plt.figure()
plt.imshow(cont, cmap='gray')
n_scatter = 1000
plt.scatter(positions[:n_scatter, 0], positions[:n_scatter, 1], s=10, alpha=.2, c='r')
plt.show()