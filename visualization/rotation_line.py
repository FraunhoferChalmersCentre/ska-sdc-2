from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
df['major'] = df.hi_size / 2.8
df['minor'] = np.sqrt(np.cos(np.deg2rad(df.i)) ** 2 + .2 ** 2 / (1 - .2 ** 2)) * df.major
df = df[(df.major > 1) & (df.minor > 1)].reset_index()
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)

positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)
HI_rest_frequency = 1.420e9
speed_of_light = 3e5
bandwidth = 30e3
df['n_bands'] = HI_rest_frequency * df.w20 / (speed_of_light * bandwidth)
df['expected_signal'] = df.line_flux_integral * 2.8 ** 2 / (df.n_bands * bandwidth * (df.hi_size / 2) ** 2 * np.pi)

sizes = df['line_flux_integral']
sorted_sizes = sizes.sort_values(ascending=True)

min_size = .5
max_size = 1.
padding = 20
extended_padding = int(1.5 * padding)

index_pos = 0
source_index = None
d = None
while source_index is None or not extended_padding < d[0] < hi_data.shape[
    2] - extended_padding or not extended_padding < d[1] < hi_data.shape[1] - extended_padding:
    index_pos = np.random.randint(int(min_size * len(sizes)), int(max_size * len(sizes)))
    source_index = sorted_sizes.index[index_pos]
    d = positions[source_index].flatten()

print(index_pos / len(sorted_sizes))
r = df.iloc[source_index]
center_band = d[2]

start_band = round(center_band - r.n_bands / 2)
end_band = round(center_band + r.n_bands / 2)

freq_span_padding = max(int(.2 * (end_band - start_band)), 20)
start = max(0, start_band - freq_span_padding)
end = min(len(hi_data), end_band + freq_span_padding)

for i in range(start, end):
    hi_data[i] = (hi_data[i] - hi_data[i].min()) / (np.percentile(hi_data[i], 99.5) - hi_data[i].min())
    hi_data[i] = (np.power(1000, hi_data[i]) - 1) / 1000

twice_padded = hi_data[start:end, d[1] - extended_padding:d[1] + extended_padding + 1,
               d[0] - extended_padding:d[0] + extended_padding + 1].copy()
angle = np.deg2rad(-r.pa)
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

relative_indices = np.array(
    [[(i - padding, j - padding) for j in range(2 * padding + 1)] for i in range(2 * padding + 1)]).astype(int)
rot_indices = np.round([[rotation_matrix @ v.T for v in r] for r in relative_indices]).astype(
    np.int) + extended_padding

cropped_rotated = np.array([twice_padded[:, r[:, 1], r[:, 0]].T for r in rot_indices]).T

cropped = hi_data[start:end, d[1] - padding:d[1] + padding + 1, d[0] - padding:d[0] + padding + 1].copy()

n_rotations = 4
angle = np.pi / n_rotations

fig, axes = plt.subplots(2, n_rotations + 1, figsize=(n_rotations * 5, 10), sharey=True, sharex=True)

vertical_indices = np.array([(0, i - padding) for i in range(cropped.shape[1])]).astype(int)

major = r.hi_size / (2 * 2.8)
minor = np.sqrt(np.cos(np.deg2rad(r.i)) ** 2 + .2 ** 2 / (1 - .2 ** 2)) * r.hi_size / (2 * 2.8)

for row, d in enumerate([cropped, cropped_rotated]):

    rotated_cross_sections = []

    for i in range(n_rotations + 1):
        rotation_matrix = np.array([[np.cos(i * angle), -np.sin(i * angle)], [np.sin(i * angle), np.cos(i * angle)]])
        rotated = np.round([rotation_matrix @ v.T for v in vertical_indices]).astype(np.int) + padding - 1

        cross_section = d[:, rotated[:, 1], rotated[:, 0]]
        rotated_cross_sections.append(cross_section)

    rotated_cross_sections = np.array(rotated_cross_sections)
    vmax = np.percentile(rotated_cross_sections, 99.9)

    for i, cross_section in enumerate(rotated_cross_sections):
        axes[row, i].imshow(cross_section, cmap='gray', extent=(0, cross_section.shape[1], start, end), vmax=vmax)
        axes[row, i].hlines(center_band, 0, cross_section.shape[1], color='r', lw=4, alpha=.3)
        axes[row, i].hlines([start_band, end_band], 0, cross_section.shape[1], color='g', lw=4, alpha=.3)

        axes[row, i].set_title('{:.1f}'.format(np.rad2deg(i * angle)))

        if i > 0:
            axes[row, i].set_yticks([])

        if row > 0:
            dist = np.linalg.norm([major * np.cos(i * angle), minor * np.sin(i * angle)])
            axes[row, i].vlines([cross_section.shape[1] / 2 - dist,
                                 cross_section.shape[1] / 2 + dist], start_band,
                                end_band, color='b', alpha=.3, lw=4)

    axes[row, 0].set_ylabel('freq band')

print('inclination', r.i, 'pa', r.pa, 'major', major, 'minor', minor)

plt.show()
