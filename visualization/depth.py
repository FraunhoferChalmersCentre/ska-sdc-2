from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)

positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)

key = 'line_flux_integral'
sizes = df[key].copy().values
sizes.sort()

# source_index = df[df[key] == sizes[-10]].index[0]
min_size = .999
max_size = 1.
candidates = df[(df[key] > sizes[int(min_size * len(sizes)) - 1]) & (df[key] < sizes[int(max_size * len(sizes)) - 1])]
source_index = np.random.choice(candidates.index)
freq = df.iloc[source_index]['central_freq']

d = positions[source_index].flatten()
center_band = d[2]
padding = 30
padding = min(hi_data.shape[1] - d[1], d[1], padding)
padding = min(hi_data.shape[2] - d[0], d[0], padding)
cropped = hi_data[:, d[1] - padding:d[1] + padding, d[0] - padding:d[0] + padding]

r = df.iloc[source_index]
rest_freq = 1.420e9
c = 3e5
center_speed = c * (rest_freq - freq) / rest_freq
lw = r.w20
bw = rest_freq * lw / c

upper_freq = r.central_freq - bw / 2
lower_freq = r.central_freq + bw / 2

start_band = max(0, wcs.all_world2pix([[r['ra'], r['dec'], upper_freq]], 0).astype(np.int)[0, -1])
end_band = min(len(cropped), wcs.all_world2pix([[r['ra'], r['dec'], lower_freq]], 0).astype(np.int)[0, -1])

fig, axes = plt.subplots(1, 3)
span_center = 50
start = max(0, center_band - span_center)
end = min(len(cropped), center_band + span_center)

axes[0].imshow(cropped[start:end, padding, :], extent=(d[1] - padding, d[1] + padding, start, end), cmap='gray')
axes[0].hlines(center_band, d[1] - padding, d[1] + padding, color='r', lw=2)
axes[0].hlines([start_band, end_band], d[1] - padding, d[1] + padding, color='g', lw=2)

axes[1].imshow(cropped[start:end, :, padding], extent=(d[0] - padding, d[0] + padding, start, end), cmap='gray')
axes[1].hlines(center_band, d[0] - padding, d[0] + padding, color='r', lw=2)
axes[1].hlines([start_band, end_band], d[0] - padding, d[0] + padding, color='g', lw=2)

axes[2].imshow(np.mean(cropped[center_band - 5: center_band + 5, :, :], axis=0), cmap='gray')
for ax in axes[:-1]:
    ax.set_xlabel('freq band')
    ax.set_ylabel('pixel')

plt.show()
