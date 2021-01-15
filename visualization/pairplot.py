import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('../data/sky_dev_truthcat.txt', sep=' ')
df.line_flux_integral = np.log(df.line_flux_integral)
df.hi_size = np.log(df.hi_size)

sns.pairplot(df, plot_kws=dict(marker="+", linewidth=1, alpha=.1))
