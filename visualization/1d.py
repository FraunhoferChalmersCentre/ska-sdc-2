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

sizes = df['line_flux_integral'].copy()
sorted_sizes = sizes.sort_values(ascending=True)

min_size = .99
max_size = 1.
index_pos = np.random.randint(int(min_size * len(sizes)), int(max_size * len(sizes)))
print(index_pos / len(sorted_sizes))
source_index = sorted_sizes.index[index_pos]
freq = df.iloc[source_index]['central_freq']

d = positions[source_index].flatten()
band = d[2]
padding_w = min(hi_data.shape[1] - d[1], d[1], 100)
padding_h = min(hi_data.shape[2] - d[0], d[0], 20)

r = df.iloc[source_index]
rest_freq = 1.420e9
c = 3e5
center_speed = c * (rest_freq - freq) / rest_freq
bw = rest_freq * r.w20 / c

upper_freq = r.central_freq - bw / 2
lower_freq = r.central_freq + bw / 2

start_band = max(0, wcs.all_world2pix([[r['ra'], r['dec'], upper_freq]], 0).astype(np.int)[0, -1])
end_band = min(len(hi_data), wcs.all_world2pix([[r['ra'], r['dec'], lower_freq]], 0).astype(np.int)[0, -1])
span_center = 50
start = max(0, start_band - span_center)
end = min(len(hi_data), end_band + span_center)

fig, axes = plt.subplots(2, 1, sharex=True)
w = 9

spectrum = hi_data[start:end, d[1], d[0]]
axes[0].plot(np.arange(start, end), spectrum)
axes[0].vlines(band, spectrum.min(), spectrum.max(), color='r')
axes[0].vlines([start_band, end_band], spectrum.min(), spectrum.max(), color='g')
axes[0].plot(np.arange(start + int(w / 2 - 1 / 2), end - int(w / 2 - 1 / 2)),
             np.convolve(spectrum, np.ones(w), 'valid') / w)

a = 1e3
bands = hi_data[start: end].copy()
transformed = hi_data[start: end].copy()
for i, b in enumerate(bands):
    upper_perc = np.percentile(b, 99)
    robust_b = np.where(b > upper_perc, upper_perc, b)
    transformed[i] = (transformed[i] - robust_b.min()) / (robust_b.max() - robust_b.min())
    transformed[i] = (np.power(a, transformed[i]) - 1) / a

spectrum = transformed[:, d[1], d[0]]
axes[1].plot(np.arange(start, end), spectrum)
axes[1].vlines(band, spectrum.min(), spectrum.max(), color='r')
axes[1].vlines([start_band, end_band], spectrum.min(), spectrum.max(), color='g')
axes[1].plot(np.arange(start + int(w / 2 - 1 / 2), end - int(w / 2 - 1 / 2)),
             np.convolve(spectrum, np.ones(w), 'valid') / w)
plt.show()
