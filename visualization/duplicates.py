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

xy_pos = positions[:, :2]
neighbours = []
neighbour_pos = []
for i, p in enumerate(xy_pos[:-1]):
    dist = xy_pos[i + 1:, :] - p
    normed_dist = np.linalg.norm(dist, axis=1)
    rel_ids = np.argwhere(normed_dist < 2)
    if len(rel_ids) > 0:
        for rel_id in rel_ids:
            neighbour_pos.append(p)
            neighbours.append([i, i + int(rel_id) + 1])

neighbours = np.array(neighbours)
neighbour_pos = np.array(neighbour_pos)

# freq_dist = []
# for n in neighbours:
#     freq_dist.append(abs(positions[n[0], 2] - positions[n[1], 2]))

# plt.scatter(neighbours[:, 1], freq_dist)
# plt.show()

plt.figure()
span_w = 100
r = np.random.randint(0, len(neighbours))

min_freq = int(np.argmin([positions[neighbours[r, 0], 2], positions[neighbours[r, 1], 2]]))
max_freq = int(np.argmax([positions[neighbours[r, 0], 2], positions[neighbours[r, 1], 2]]))
spectrum = hi_data[positions[neighbours[r, min_freq], 2] - span_w:positions[neighbours[r, max_freq], 2] + span_w,
           neighbour_pos[r, 0], neighbour_pos[r, 1]]

c = 1#20
spectrum = np.convolve(spectrum, np.ones(2 * c + 1))
plt.plot(
    np.arange(positions[neighbours[r, min_freq], 2] - span_w - c, positions[neighbours[r, max_freq], 2] + span_w + c),
    spectrum)
plt.vlines([positions[neighbours[r, min_freq], 2], positions[neighbours[r, max_freq], 2]], spectrum.min(),
           spectrum.max(),
           color='r')
plt.show()
