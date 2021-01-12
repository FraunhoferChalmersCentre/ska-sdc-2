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

#source_index = df[df[key] == sizes[-10]].index[0]
source_index = np.random.choice(df[df['line_flux_integral'] > sizes[int(.5 * len(sizes))]].index)
freq = df.iloc[source_index]['central_freq']


d = positions[source_index].flatten()
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

r = df.iloc[source_index]
rest_freq = 1.420e9
c = 3e5
center_speed = c * (rest_freq - freq) / rest_freq
lw = r.w20
bw = rest_freq * lw / c

upper_freq = r.central_freq - bw / 2
lower_freq = r.central_freq + bw / 2


galaxy_start = wcs.all_world2pix([[r['ra'], r['dec'], upper_freq]], 0).astype(np.int)[0]
galaxy_end = wcs.all_world2pix([[r['ra'], r['dec'], lower_freq]], 0).astype(np.int)[0]

plt.figure()
span_center = 200
start = max(0, band - span_center)
end = min(len(cropped), band + span_center)
spectrum = cropped[start:end, padding_w, padding_h]
plt.plot(np.arange(start, end), spectrum)
plt.vlines(band, spectrum.min(), spectrum.max(), color='r')
plt.vlines([galaxy_start[-1], galaxy_end[-1]], spectrum.min(), spectrum.max(), color='g')
plt.show()
