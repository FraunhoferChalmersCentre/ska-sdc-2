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
source_index = np.random.choice(df[df['line_flux_integral'] > sizes[int(.95 * len(sizes))]].index)
freq = df.iloc[source_index]['central_freq']

d = positions[source_index].flatten()
band = d[2]
padding_w = min(hi_data.shape[1] - d[1], d[1], 100)
padding_h = min(hi_data.shape[2] - d[0], d[0], 20)
cropped = hi_data[:, d[1] - padding_w:d[1] + padding_w, d[0] - padding_h:d[0] + padding_h]

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

averaging_width = 10
n_plots = 2 + np.ceil((end_band - start_band) / averaging_width).astype(int)
fig, axes = plt.subplots(1, n_plots, sharex=True, sharey=True)

for i, ax in enumerate(axes):
    k = start_band + averaging_width * i - averaging_width
    channel = np.mean(cropped[k: k + averaging_width, :, :], axis=0)
    ax.imshow(channel)
    ax.set_title('{}, {}'.format(k, k + averaging_width))
    # ax.scatter([100], d[:, 1], c='r', s=5, alpha=.5)
plt.show()

plt.figure()
span_center = 50
start = max(0, band - span_center)
end = min(len(cropped), band + span_center)
spectrum = cropped[start:end, padding_w, padding_h]
plt.plot(np.arange(start, end), spectrum)
plt.vlines(band, spectrum.min(), spectrum.max(), color='r')
plt.vlines([start_band, end_band], spectrum.min(), spectrum.max(), color='g')
plt.show()
