# %%
import sparse
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform

from pipeline.common import filename
from pipeline.data.segmentmap import create_from_files, prepare_df

mpl.rcParams['text.usetex'] = True
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

ALPHA = .2
HI_REST_FREQ = 1.420e9
SPEED_OF_LIGHT = 3e5

PADDING = 0

# %%

from spectral_cube import SpectralCube
from astropy.wcs import WCS

size = 'dev_s'
hi_data = fits.getdata(filename.data.sky(size), ignore_blank=True).T
header = fits.getheader(filename.data.sky(size), ignore_blank=True)
df = pd.read_csv(filename.data.true(size), sep=' ', index_col='id')
df = df.sort_values(by='line_flux_integral', ignore_index=False, ascending=False)

wcs = WCS(header)

# %%
from pipeline.data.segmentmap import create_from_df
from definitions import config

test_dataset_path = filename.processed.test_dataset(config['traversing']['checkpoint'])
df = prepare_df(df, header)
sourcemap, allocation_dict = create_from_df(df, header, padding=1)

# %%
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def rotated_cross_sections(cube: np.ndarray, x: float, y: float, pa: float, margin: int, start: int, end: int,
                           n_rotations: int = 3, binary=False, max_angle=180, shift=1):
    padded = cube[int(x - margin):int(x + margin + 1), int(y - margin):int(y + margin + 1), start:end].astype(
        np.float32)
    if isinstance(cube, sparse.SparseArray):
        padded = padded.todense()

    angle = max_angle / (n_rotations + 1e-7)

    cross_sections = []

    for i in range(n_rotations + 1):
        rotated = np.array(
            [transform.rotate(padded[:, :, c], - pa + 90 + i * angle, order=1) for c in range(padded.shape[-1])]).T
        this_cross_section = np.mean(rotated[int(margin - shift):int(margin + shift + 1), :, :], axis=0)
        if binary:
            this_cross_section = this_cross_section.astype(np.bool)

        cross_sections.append(this_cross_section)

    return np.array(cross_sections)


def plot_pv(df, hi_data, sourcemap, index=None, padding=10, shift=1, max_angle=90, n_rotations=1):
    extended_padding = int(1.5 * padding)
    row = None if index is None else df.loc[index]
    while row is None or not extended_padding < row.x < hi_data.shape[
        0] - extended_padding or not extended_padding < row.y < hi_data.shape[1] - extended_padding:
        index = np.random.choice(df.index)
        row = df.loc[index]

    x, y, z = tuple(map(int, row[['x', 'y', 'z']]))

    start_band = np.round(z - row.n_channels / 2).astype(int)
    end_band = np.round(z + row.n_channels / 2).astype(int)

    margin = row.major_radius_pixels + padding

    start = max(0, start_band - padding)
    end = min(hi_data.shape[-1], end_band + padding)

    data_cross_sections = rotated_cross_sections(hi_data, x, y, row.pa, margin, start, end, n_rotations, shift=shift,
                                                 max_angle=max_angle)
    segmentmap_cross_sections = rotated_cross_sections(sourcemap, x, y, row.pa, margin, start, end, n_rotations,
                                                       shift=shift, max_angle=max_angle)

    fig, axes = plt.subplots(2, n_rotations + 2, figsize=(10, 5.5))

    for j, cross_sections in enumerate([data_cross_sections, segmentmap_cross_sections]):

        vmax = np.percentile(cross_sections, 99.9)

        for i, cross_section in enumerate(cross_sections):
            coords_range = wcs.all_pix2world(
                [[0, 0, start], [cross_section.shape[1], cross_section.shape[0], end]], 0)
            extent = coords_range[[0, 1, 0, 1], [0, 0, 2, 2]] - np.array(
                [np.mean(coords_range[:, 0]) - (-1) ** i * 2 * header['CDELT1'],
                 np.mean(coords_range[:, 0]) - (-1) ** i * 2 * header['CDELT1'],
                 0, 0])

            extent /= np.array([1, 1, 1e6, 1e6])
            aspect = 1e6 * abs((coords_range[0, 0] - coords_range[1, 0]) / (coords_range[0, 2] - coords_range[1, 2]))

            axes[j, i].imshow(cross_section.T,
                              cmap='gray',
                              vmax=vmax,
                              extent=extent)

            axes[j, i].set_aspect(aspect)
            angle = i * (max_angle / n_rotations)
            if np.isclose(angle, 0):
                axes[j, i].set_title('Major')
            elif np.isclose(angle, 90):
                axes[j, i].set_title('Minor')

            axes[j, i].set_ylabel('Frequency (MHz)')
            axes[j, i].set_xlabel('Offset (deg)')
            #axes[j, i].set_xticklabels(axes[j, i].get_xticks(), rotation=45, size=10)
            axes[j, i].set_yticklabels(axes[j, i].get_yticks().astype(int), rotation=45, size=10)
            axes[j, i].xaxis.set_major_formatter(FuncFormatter('{:.3f}'.format))

    coords_range = wcs.all_pix2world(
        [[int(x - margin), int(y - margin), start], [int(x + margin + 1), int(y + margin + 1), end]], 0)
    axes[0, -1].imshow(
        hi_data[int(x - margin):int(x + margin + 1), int(y - margin):int(y + margin + 1), start:end].sum(axis=2),
        cmap='gray', extent=coords_range[[0, 1, 0, 1], [0, 0, 1, 1]])
    axes[1, -1].imshow(
        sourcemap[int(x - margin):int(x + margin + 1), int(y - margin):int(y + margin + 1), start: end].todense().sum(
            axis=2), cmap='gray', extent=coords_range[[0, 1, 0, 1], [0, 0, 1, 1]])

    for ax in axes.ravel():
        ax.grid(True, color='c', alpha=.5)

    for ax in [axes[0, -1], axes[1, -1]]:
        ax.set_title('Moment 0')
        ax.set_xlabel(r'RA (deg)')
        ax.set_ylabel(r'DEC (deg)')
        #ax.set_xticklabels(ax.get_xticks(), rotation=45, size=10)
        ax.set_yticklabels(ax.get_yticks(), rotation=45, size=10)
        ax.xaxis.set_major_formatter(FuncFormatter('{:.3f}'.format))
        ax.yaxis.set_major_formatter(FuncFormatter('{:.3f}'.format))

    axes[0, 0].annotate(r'\mbox{H\,\small I} data', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=axes[0, 0].yaxis.label, textcoords='offset points',
                        size=20, ha='right', va='center')
    axes[1, 0].annotate('Target mask', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                        size=20, ha='right', va='center')

    plt.subplots_adjust(left=.21, wspace=.5, hspace=.3)
    plt.savefig('target_comparison.pdf')
    plt.show()

plot_pv(df.head(2).tail(1), hi_data, sourcemap, shift=2, max_angle=90)

# %%

fluxes = np.geomspace(df.line_flux_integral.min(), df.line_flux_integral.max(), 20)
bin_dict = {i: np.argmax(r.line_flux_integral < fluxes) for i, r in df.iterrows()}

fig, ax = plt.subplots()

ax2 = ax.twinx()

ax.hist(df.line_flux_integral, bins=fluxes)
paddings = np.arange(0, 3)
sums = []
for p in paddings:
    bins = np.zeros(len(fluxes))
    _, allocations = create_from_files('dev_s', regenerate=True, padding=p)
    for k, v in allocations.items():
        bins[bin_dict[k]] += len(v)

    ax2.plot(fluxes, bins, c=plt.cm.viridis(p / paddings.max()))

ax.set_xscale('log')
plt.show()
