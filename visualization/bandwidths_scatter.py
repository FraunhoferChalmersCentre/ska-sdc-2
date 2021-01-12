from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')


rest_freq = 1.420e9
c = 3e5
plt.hist(rest_freq * df.w20 / (c * 30e3), bins=30)
plt.show()
