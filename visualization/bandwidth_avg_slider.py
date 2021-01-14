from matplotlib.widgets import Slider, Button
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(0)
percentage = 0.9
avg_width = 20
dynamic_scale = True

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_dev.fits')
wcs = WCS(header)

positions = wcs.all_world2pix(df[['ra', 'dec', 'central_freq']], 0).astype(np.int)

key = 'line_flux_integral'
sizes = df[key].copy().values
sizes.sort()


def get_source(_df, _positions, _hi_data, _wcs, _percentage, _sizes):
    source_index = np.random.choice(_df[_df['line_flux_integral'] > _sizes[int(_percentage * len(_sizes))]].index)
    _freq = _df.iloc[source_index]['central_freq']

    d = _positions[source_index].flatten()

    padding_w = min(_hi_data.shape[1] - d[1], d[1], 20)
    padding_h = min(_hi_data.shape[2] - d[0], d[0], 20)
    _cropped = hi_data[:, d[1] - padding_w:d[1] + padding_w, d[0] - padding_h:d[0] + padding_h]

    r = _df.iloc[source_index]
    rest_freq = 1.420e9
    c = 3e5
    lw = r.w20
    bw = rest_freq * lw / c

    _upper_freq = r.central_freq - bw / 2
    _lower_freq = r.central_freq + bw / 2

    _start_band = wcs.all_world2pix([[r['ra'], r['dec'], _upper_freq]], 0).astype(np.int)[0, -1]
    _end_band = wcs.all_world2pix([[r['ra'], r['dec'], _lower_freq]], 0).astype(np.int)[0, -1]
    return _freq, _cropped, _upper_freq, _lower_freq, _start_band, _end_band


freq, cropped, upper_freq, lower_freq, start_band, end_band = get_source(df, positions, hi_data, wcs, percentage, sizes)

fig = plt.figure()
ax = fig.add_subplot(111)

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.035, 0.65, 0.03], facecolor=axcolor)
axavg = plt.axes([0.25, 0.005, 0.65, 0.03], facecolor=axcolor)
dfreq = (lower_freq - upper_freq)/(end_band - start_band + 1)
sfreq = Slider(axfreq, 'Freq', upper_freq-avg_width/2*dfreq, lower_freq+avg_width/2*dfreq, valinit=freq, valstep=dfreq)
savg = Slider(axavg, 'Avg', 1, end_band-start_band, valinit=avg_width, valstep=1)

central_band = int((start_band+end_band)/2)
channel = np.mean(cropped[int(central_band-avg_width/2): int(central_band+avg_width/2), :, :], axis=0)
im = ax.imshow(channel)


def update(val):
    frequency = sfreq.val
    averaging_w = int(savg.val)
    pos = int(end_band + averaging_w / 2 - int(lower_freq + averaging_w / 2 * dfreq - frequency) / dfreq)
    cl = np.mean(cropped[int(pos-averaging_w/2): int(pos+averaging_w/2), :, :], axis=0)
    im.set_data(cl)
    fig.canvas.draw_idle()


sfreq.on_changed(update)
savg.on_changed(update)

resetax = plt.axes([0.8, 0.075, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    savg.reset()


button.on_clicked(reset)

sampleax = plt.axes([0.8, 0.115, 0.1, 0.04])
button_sample = Button(sampleax, 'Sample', color=axcolor, hovercolor='0.975')


def sample(event):
    global end_band
    global cropped
    global lower_freq
    freq, cropped, upper_freq, lower_freq, start_band, end_band = get_source(df, positions, hi_data, wcs, percentage,
                                                                             sizes)
    global dfreq
    dfreq = (lower_freq - upper_freq) / (end_band - start_band + 1)
    sfreq.valmin = upper_freq-avg_width/2*dfreq
    sfreq.valmax = lower_freq+avg_width/2*dfreq
    sfreq.valinit = freq
    savg.valmax = end_band-start_band
    savg.valinit = avg_width
    sfreq.ax.set_xlim(sfreq.valmin, sfreq.valmax)
    savg.ax.set_xlim(savg.valmin, savg.valmax)
    sfreq.val = sfreq.valmin
    savg.val = savg.valmin
    sfreq.valstep = dfreq
    central_band = int((start_band + end_band) / 2)
    cl = np.mean(cropped[int(central_band - avg_width / 2): int(central_band + avg_width / 2), :, :], axis=0)
    im.set_data(cl)

    if dynamic_scale:
        fig_new = plt.figure()
        ax_new = fig_new.add_subplot(111)
        im_new = ax_new.imshow(cl)
        im.set_clim(im_new.get_clim())

    fig.canvas.draw_idle()
    reset(1)


button_sample.on_clicked(sample)

plt.show()
