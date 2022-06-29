##
import numpy as np
import matplotlib.pyplot as plt

from definitions import config
from visualization.utils import get_pareto, get_params_df, get_predicted_dfs, get_truthcat

plt.rcParams['text.usetex'] = True

ALPHA = .5
MAX_RANK = np.inf

truthcat = get_truthcat()

predicted_dfs = np.array(get_predicted_dfs())

params_df = get_params_df(predicted_dfs, truthcat).reset_index(drop=True)

params_df = get_pareto(params_df)

linear_attrs = {'hi_size': 'Major axis', 'line_flux_integral': 'Line flux integral', 'w20': 'Line width'}

units = {'hi_size': 'arcsec', 'line_flux_integral': 'Jy Hz', 'w20': 'km/s'}

angle_attr = {'pa': 'Position angle', 'i': 'Inclination'}

##
fig, a = plt.subplots(2, int((len(linear_attrs) + len(angle_attr) + 1) / 2), figsize=(11, 7))
axes = [*a[0], *a[1]]

for i in range(len(predicted_dfs)):
    predicted_dfs[i] = predicted_dfs[i][predicted_dfs[i].match >= 0]

cmap = plt.get_cmap('winter').reversed()

predicted_dfs = predicted_dfs[params_df.best_rank < MAX_RANK]
params_df = params_df[params_df.best_rank < MAX_RANK]

axes[0].scatter(params_df.precision, params_df.recall, c=params_df.best_rank + 1, cmap=cmap, alpha=ALPHA)

axes[0].set_ylabel(r'Completeness')
axes[0].set_xlabel(r'Reliability')

iou = config['iou_threshold']
for l, ax in zip(linear_attrs.keys(), axes[1:]):
    maes = []
    ious = []
    for d in predicted_dfs:
        d_iou = d[iou < d.iou]
        maes.append(np.median(np.abs((d_iou[f'{l}_prediction'] - d_iou[f'{l}_target']) / d_iou[f'{l}_target'])))
        ious.append(d.iou.mean())
    print(l)
    pcm = ax.scatter(params_df.precision, maes, c=params_df.best_rank + 1, cmap=cmap, alpha=ALPHA)

    ax.set_title(linear_attrs[l])
    ax.set_xlabel(r"Reliability")
    ax.set_ylabel(f'Median relative error')

for l, ax in zip(angle_attr.keys(), axes[1 + len(linear_attrs):]):
    print(l)
    mae = []
    ious = []
    for d in predicted_dfs:
        d_iou = d[iou < d.iou]

        abs_error = np.mod((d_iou[f'{l}_prediction'] - d_iou[f'{l}_target']).values, 360)
        abs_error = np.minimum(abs_error, 360 - abs_error, abs_error)
        mae.append(np.median(abs_error))
        ious.append(d.iou.mean())

    pcm = ax.scatter(params_df.precision, mae, c=params_df.best_rank + 1, cmap=cmap, alpha=ALPHA)

    ax.set_title(angle_attr[l])
    ax.set_xlabel(r'Reliability')
    ax.set_ylabel(r'Median absolute error (deg)')

plt.subplots_adjust(wspace=.3, hspace=.3)

cbar = fig.colorbar(pcm, label='Best ranking', ax=axes, shrink=.6, pad=.01, fraction=.02)
cbar.solids.set(alpha=1)
cbar.ax.invert_yaxis()

plt.savefig('tradeoff_best.pdf')
plt.show()
