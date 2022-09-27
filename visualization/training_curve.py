import glob

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


def get_df(keyword):
    files = glob.glob(f'loss_files/*{keyword}_loss.csv')
    dfs = [pd.read_csv(t) for t in files]
    return pd.concat(dfs)


training_df = get_df('train')
val_df = get_df('val')
plt.plot(training_df.Step, training_df.Value, label='Training')
plt.plot(val_df.Step, val_df.Value, label='Validation')
plt.ylabel('Loss')
plt.xlabel('Gradient steps')
plt.legend()
plt.show()
