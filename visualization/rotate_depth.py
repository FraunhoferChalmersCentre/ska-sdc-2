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


sizes = df['line_flux_integral']
sorted_sizes = sizes.sort_values(ascending=True)

min_size = .5
max_size = 1.
index_pos = np.random.randint(int(min_size * len(sizes)), int(max_size * len(sizes)))
print(index_pos / len(sorted_sizes))
source_index = sorted_sizes.index[index_pos]
r = df.iloc[source_index]

d = positions[source_index].flatten()
center_band = d[2]
padding = 30
padding = min(hi_data.shape[1] - d[1], d[1], padding)
padding = min(hi_data.shape[2] - d[0], d[0], padding)

start_band = round(center_band - r.n_bands / 2)
end_band = round(center_band + r.n_bands / 2)

freq_span_padding = max(int(.2 * (end_band - start_band)), 20)
start = max(0, start_band - freq_span_padding)
end = min(len(hi_data), end_band + freq_span_padding)

cropped = hi_data[start:end, d[1] - padding:d[1] + padding + 1, d[0] - padding:d[0] + padding + 1].copy()
transformed = cropped.copy()

a = 1e3
for i, b in enumerate(hi_data[start:end].copy()):
    upper_perc = np.percentile(b, 99)
    robust_b = np.where(b > upper_perc, upper_perc, b)
    transformed[i] = (transformed[i] - robust_b.min()) / (robust_b.max() - robust_b.min())
    transformed[i] = (np.power(a, transformed[i]) - 1) / a

n_rotations = 10
angle = np.pi / n_rotations

fig, axes = plt.subplots(2, n_rotations + 1, figsize=(n_rotations * 5, 10), sharey=True, sharex=True)


vertical_indices = np.array([(padding - i, 0) for i in range(cropped.shape[1])]).astype(int)

for row, d in enumerate([cropped, transformed]):
    vmin = np.inf
    vmax = -np.inf
    rotated_cross_sections = []
    for i in range(n_rotations + 1):
        rotation_matrix = np.array([[np.cos(i * angle), -np.sin(i * angle)], [np.sin(i * angle), np.cos(i * angle)]])
        rotated = np.round([rotation_matrix @ v.T for v in vertical_indices]).astype(np.int) + padding - 1

        cross_section = d[:, rotated[:, 1], rotated[:, 0]]
        rotated_cross_sections.append(cross_section)

    vmin = np.min(rotated_cross_sections)
    vmax = np.percentile(rotated_cross_sections, 99.9)

    for i, cross_section in enumerate(rotated_cross_sections):
        axes[row, i].imshow(cross_section, cmap='gray', extent=(0, cross_section.shape[1], start, end), vmin=vmin, vmax=vmax)
        axes[row, i].hlines(center_band, 0, cross_section.shape[1], color='r', lw=4, alpha=.3)
        axes[row, i].hlines([start_band, end_band], 0, cross_section.shape[1], color='g', lw=4, alpha=.3)
        # axes[0, i].set_aspect(2)
        axes[row, i].set_title('{:.1f}'.format(np.rad2deg(i * angle)))

        if i > 0:
            axes[row, i].set_yticks([])

    axes[row, 0].set_ylabel('freq band')

print('pixels', (df.iloc[source_index].hi_size / 2) ** 2 * np.pi / (2.8 ** 2))

plt.tight_layout()
plt.show()
