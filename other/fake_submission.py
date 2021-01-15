from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd

# hi_data = fits.getdata('../data/sky_dev.fits')  # Open the FITS file for reading
header = fits.getheader('../data/sky_eval.fits')
wcs = WCS(header)

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
hi_min, hi_max = df.hi_size.min(), df.hi_size.max()
flux_min, flux_max = df.line_flux_integral.min(), df.line_flux_integral.max()
pa_min, pa_max = df.pa.min(), df.pa.max()
i_min, i_max = df.i.min(), df.i.max()
w20_min, w20_max = df.w20.min(), df.w20.max()

samples = 2000
attrs = {'id': [], 'ra': [], 'dec': [], 'hi_size': [], 'line_flux_integral': [], 'central_freq': [], 'pa': [], 'i': [],
         'w20': []}
for s in range(samples):
    f, x, y = tuple(map(lambda t: np.random.uniform(20, t - 60), [header['NAXIS{}'.format(i)] for i in range(1, 4)]))
    ra, dec, freq = wcs.all_pix2world([[f, x, y]], 0).flatten()

    attrs['id'].append(s)
    attrs['ra'].append(ra)
    attrs['dec'].append(dec)
    attrs['hi_size'].append(np.random.uniform(hi_min, hi_max))
    attrs['line_flux_integral'].append(np.random.uniform(flux_min, flux_max))
    attrs['central_freq'].append(np.round(freq))
    attrs['pa'].append(np.random.uniform(pa_min, pa_max))
    attrs['i'].append(np.random.uniform(i_min, i_max))
    attrs['w20'].append(np.random.uniform(i_min, i_max))

submission = pd.DataFrame.from_dict(attrs)
submission.to_csv('fake_submission.txt', sep=' ', index=False)
