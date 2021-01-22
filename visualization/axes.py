import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
df['major'] = df.hi_size / 2.8
df['minor'] = np.sqrt(np.cos(np.deg2rad(df.i)) ** 2 + .2 ** 2 / (1 - .2 ** 2)) * df.major
filtered = df[(df.major > 1) & (df.minor > 1)]

fig, axes = plt.subplots(2, 1)
axes[0].scatter(filtered.line_flux_integral, filtered.major, alpha=.1)
axes[1].scatter(filtered.line_flux_integral, filtered.minor, alpha=.1)
