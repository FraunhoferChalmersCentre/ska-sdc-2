from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat_v1.1.txt', sep=' ')
df['major'] = df.hi_size / 2.8
df['minor'] = np.sqrt(np.cos(np.deg2rad(df.i)) ** 2 + .2 ** 2 / (1 - .2 ** 2)) * df.major
df = df[(df.major > 1) & (df.minor > 1)].reset_index()
hi_data = fits.getdata('../data/sky_dev.fits').T  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)

positions = np.round(wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0)).astype(int)
HI_rest_frequency = 1.420e9
speed_of_light = 3e5
bandwidth = 30e3
df['n_bands'] = HI_rest_frequency * df.w20 / (speed_of_light * bandwidth)
df['expected_signal'] = df.line_flux_integral * 2.8 ** 2 / (df.n_bands * bandwidth * (df.hi_size / 2) ** 2 * np.pi)
df[['x', 'y', 'z']] = positions.astype(np.int)

df = df.sort_values(by='line_flux_integral', ignore_index=True)

min_size = .5
max_size = 1.
padding = 20
extended_padding = int(1.5 * padding)

index = 0
row = None
while row is None or not extended_padding < row.x < hi_data.shape[
    0] - extended_padding or not extended_padding < row.y < hi_data.shape[1] - extended_padding:
    index = np.random.randint(int(min_size * len(df)), int(max_size * len(df)))
    row = df.loc[index]

print(index/len(df))

x, y, z = tuple(map(int, row[['x', 'y', 'z']]))

start_band = np.round(z - row.n_bands / 2).astype(np.int)
end_band = np.round(z + row.n_bands / 2).astype(np.int)

freq_span_padding = max(int(.2 * (end_band - start_band)), 20)
start = max(0, start_band - freq_span_padding)
end = min(hi_data.shape[-1], end_band + freq_span_padding)

for i in range(start, end):
    hi_data[:, :, i] = (hi_data[:, :, i] - hi_data[:, :, i].min()) / (
                np.percentile(hi_data[:, :, i], 99.5) - hi_data[:, :, i].min())
    hi_data[:, :, i] = (np.power(100, hi_data[:, :, i]) - 1) / 100

twice_padded = hi_data[x - extended_padding:x + extended_padding + 1, y - extended_padding:y + extended_padding + 1,
               start:end].copy()
angle = np.deg2rad(-row.pa)
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

relative_indices = np.array(
    [[(i - padding, j - padding) for j in range(2 * padding + 1)] for i in range(2 * padding + 1)]).astype(int)
rot_indices = np.round([[rotation_matrix @ v.T for v in r] for r in relative_indices]).astype(np.int) + extended_padding

cropped_rotated = np.array([twice_padded[r[:, 0], r[:, 1], :] for r in rot_indices])

cropped = hi_data[x - padding:x + padding + 1, y - padding:y + padding + 1, start:end].copy()

n_rotations = 4
angle = np.pi / n_rotations

fig, axes = plt.subplots(2, n_rotations + 1, figsize=(n_rotations * 5, 10), sharey=True, sharex=True)

vertical_indices = np.array([(0, i - padding) for i in range(cropped.shape[1])]).astype(int)

major = row.hi_size / (2 * 2.8)
minor = np.sqrt((np.cos(np.deg2rad(row.i)) ** 2) * (1 - .2 ** 2) + .2 ** 2) * major

for j, cube in enumerate([cropped, cropped_rotated]):

    rotated_cross_sections = []

    for i in range(n_rotations + 1):
        rotation_matrix = np.array([[np.cos(i * angle), -np.sin(i * angle)], [np.sin(i * angle), np.cos(i * angle)]])
        rotated = np.round([rotation_matrix @ v.T for v in vertical_indices]).astype(np.int) + padding

        cross_section = cube[rotated[:, 1], rotated[:, 0], :]
        rotated_cross_sections.append(cross_section)

    rotated_cross_sections = np.array(rotated_cross_sections)
    vmax = np.percentile(rotated_cross_sections, 99.9)

    for i, cross_section in enumerate(rotated_cross_sections):
        axes[j, i].imshow(cross_section.T, cmap='gray', extent=(0, cross_section.shape[0], start, end), vmax=vmax)
        axes[j, i].hlines(z, 0, cross_section.shape[0], color='r', lw=4, alpha=.3)
        axes[j, i].hlines([start_band, end_band], 0, cross_section.shape[0], color='g', lw=4, alpha=.3)

        axes[j, i].set_title('{:.1f}'.format(np.rad2deg(i * angle)))

        if i > 0:
            axes[j, i].set_yticks([])

        if j > 0:
            dist = np.linalg.norm([major * np.cos(i * angle), minor * np.sin(i * angle)])
            axes[j, i].vlines([cross_section.shape[0] / 2 - dist,
                               cross_section.shape[0] / 2 + dist], start_band,
                              end_band, color='b', alpha=.3, lw=4)

    axes[j, 0].set_ylabel('freq band')

print('inclination', row.i, 'pa', row.pa, 'major', major, 'minor', minor)

plt.show()
