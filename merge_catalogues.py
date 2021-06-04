import glob
from datetime import datetime

import pandas as pd
import numpy as np

catalogue_files = glob.glob('_n_parallel*_i_job*.txt')
dfs = [pd.read_csv(c, index_col=0, sep=' ') for c in catalogue_files]
merged_df = pd.concat(dfs)
merged_df.index = np.arange(merged_df.shape[0])
merged_df.to_csv('catalogue_{}.txt'.format(datetime.now().strftime("%Y%m%d-%H%M%S")), sep=' ', index_label='id')
