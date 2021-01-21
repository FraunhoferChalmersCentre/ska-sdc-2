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

sizes = df['line_flux_integral'].copy() / df['w20']
sorted_sizes = sizes.sort_values(ascending=False)

index_pos = np.random.choice(np.arange(int(.5 * len(sizes))))
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
lw = r.w20
bw = rest_freq * lw / c

upper_freq = r.central_freq - bw / 2
lower_freq = r.central_freq + bw / 2

start_band = max(0, wcs.all_world2pix([[r['ra'], r['dec'], upper_freq]], 0).astype(np.int)[0, -1])
end_band = min(len(hi_data), wcs.all_world2pix([[r['ra'], r['dec'], lower_freq]], 0).astype(np.int)[0, -1])
span_center = 50
start = max(0, band - span_center)
end = min(len(hi_data), band + span_center)

bands = hi_data[start: end]
cropped = bands[:, d[1] - padding_w:d[1] + padding_w, d[0] - padding_h:d[0] + padding_h]

n_plots = 10
start = int(len(cropped) / 2 - n_plots / 2)
fig, axes = plt.subplots(2, n_plots + 1, sharex=True, sharey=True)

vmin = cropped[start:start+n_plots].min()
vmax = cropped[start:start+n_plots].max()

for i, ax in enumerate(axes[0, :-1]):
    ax.imshow(cropped[start + i], cmap='gray', vmin=vmin, vmax=vmax)

axes[0, -1].imshow(cropped[start:start+n_plots].sum(axis=0), cmap='gray')

a = 1e6
for i, b in enumerate(bands):
    robust_b = b.copy()
    upper_perc = np.percentile(robust_b, 99)
    robust_b = cropped[i]#np.where(robust_b > upper_perc, upper_perc, robust_b)
    cropped[i] = (cropped[i] - robust_b.min()) / (robust_b.max() - robust_b.min())
    cropped[i] = (np.power(a, cropped[i]) - 1) / a

vmin = cropped[start:start+n_plots].min()
vmax = cropped[start:start+n_plots].max()

for i, ax in enumerate(axes[1, :-1]):
    ax.imshow(cropped[start + i], cmap='gray', vmin=vmin, vmax=vmax)

axes[1, -1].imshow(cropped[start:start+n_plots].sum(axis=0), cmap='gray')
