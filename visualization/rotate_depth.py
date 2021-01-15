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
HI_rest_frequency = 1.420e9
speed_of_light = 3e5
bandwidth = 30e3
df['n_bands'] = HI_rest_frequency * df.w20 / (speed_of_light * bandwidth)
df['expected_signal'] = df.line_flux_integral * 2.8 ** 2 / (df.n_bands * bandwidth * (df.hi_size / 2) ** 2 * np.pi)

key = 'line_flux_integral'
sizes = df[key].copy().values
sizes.sort()

min_size = .999
max_size = 1.
candidates = df[(df[key] > sizes[int(min_size * len(sizes)) - 1]) & (df[key] < sizes[int(max_size * len(sizes)) - 1])]
source_index = np.random.choice(candidates.index)
freq = df.iloc[source_index]['central_freq']

print(np.argwhere(sizes == df.iloc[source_index].line_flux_integral).flatten() / len(sizes))

d = positions[source_index].flatten()
center_band = d[2]
padding = 30
padding = min(hi_data.shape[1] - d[1], d[1], padding)
padding = min(hi_data.shape[2] - d[0], d[0], padding)
cropped = hi_data[:, d[1] - padding:d[1] + padding + 1, d[0] - padding:d[0] + padding + 1]

r = df.iloc[source_index]

start_band = round(center_band - r.n_bands / 2)
end_band = round(center_band + r.n_bands / 2)

n_rotations = 4
angle = np.pi / n_rotations

fig, axes = plt.subplots(2, n_rotations + 2, figsize=(n_rotations * 5, 10))
freq_span_padding = max(int(.2 * (end_band - start_band)), 20)
start = max(0, start_band - freq_span_padding)
end = min(len(cropped), end_band + freq_span_padding)

vertical_indices = np.array([(padding - i, 0) for i in range(cropped.shape[1])]).astype(int)

axes[0, -1].imshow(np.mean(cropped[center_band - 5: center_band + 5, :, :], axis=0), cmap='gray')
axes[0, -1].plot([padding, r.hi_size / (2 * 2.8) * np.cos(np.deg2rad(r.pa))],
                 [padding, r.hi_size / (2 * 2.8) * np.sin(np.deg2rad(r.pa))])
axes[0, -1].set_yticks([])
axes[0, -1].set_xticks([])

stds = np.load('stds.npy')

vmin = np.inf
vmax = -np.inf
rotated_cross_sections = []
for i in range(n_rotations + 1):
    rotation_matrix = np.array([[np.cos(i * angle), -np.sin(i * angle)], [np.sin(i * angle), np.cos(i * angle)]])
    rotated = np.round([rotation_matrix @ v.T for v in vertical_indices]).astype(np.int) + padding - 1

    cross_section = cropped[start:end, rotated[:, 1], rotated[:, 0]]
    rotated_cross_sections.append(cross_section)
    vmin = min(vmin, cross_section.min())
    vmax = max(vmax, cross_section.max())

for i, cross_section in enumerate(rotated_cross_sections):
    axes[0, i].imshow(cross_section, cmap='gray', extent=(0, cross_section.shape[1], end, start), vmin=vmin, vmax=vmax)
    axes[0, i].hlines(center_band, 0, cross_section.shape[1], color='r', lw=4, alpha=.2)
    axes[0, i].hlines([start_band, end_band], 0, cross_section.shape[1], color='g', lw=4, alpha=.2)
    # axes[0, i].set_aspect(2)
    axes[0, i].set_title('{:.1f}'.format(np.rad2deg(i * angle)))

    center_line = cross_section[:, int(cross_section.shape[1] / 2)].copy()
    axes[1, i].plot(center_line, np.arange(start, end), label='signal')
    axes[1, i].hlines(center_band, center_line.min(), center_line.max(), color='r', lw=4, alpha=.2)
    axes[1, i].hlines([start_band, end_band], center_line.min(), center_line.max(), color='g', lw=4, alpha=.2)
    # axes[1, i].set_xticks([])
    axes[1, i].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # axes[1, i].vlines(r.expected_signal, start_band, end_band, label='expected', color='orange')
    axes[1, i].vlines(np.mean(center_line), start_band, end_band, label='avg', color='black')
    axes[1, i].plot(stds[start:end], np.arange(start, end), label='noise std', color='magenta')
    axes[1, i].plot(2 * stds[start:end], np.arange(start, end), label='2 x noise std', color='cyan')
    axes[1, i].invert_yaxis()
    if i > 0:
        axes[0, i].set_yticks([])
        axes[1, i].set_yticks([])

print('pixels', (df.iloc[source_index].hi_size / 2) ** 2 * np.pi / (2.8 ** 2))

axes[1, -2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', )
axes[0, 0].set_ylabel('freq band')
axes[-1, -1].set_visible(False)
plt.show()
