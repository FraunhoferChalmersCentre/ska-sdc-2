##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from definitions import config

from utils import get_pareto, get_params_df, get_predicted_dfs, get_truthcat

plt.rcParams['text.usetex'] = True

COST_MAX_FILTER = 0

predicted_cataloges = get_predicted_dfs()
truthcat = get_truthcat()
params_df = get_params_df(predicted_cataloges, truthcat)

params_df = get_pareto(params_df)

predicted_catalogues = {k: v for k, v in zip(params_df.index, predicted_cataloges)}
params_df = params_df[params_df.cost_max >= COST_MAX_FILTER]
predicted_catalogues = {k: predicted_catalogues[k] for k in params_df.index}

##
matched_detections = []
for i, row in params_df.iterrows():
    df = predicted_catalogues[i][config['iou_threshold'] <= predicted_catalogues[i].iou]
    matched_detections.append(
        pd.DataFrame({'match': df.match, 'catalog': i, 'precision': row.precision,
                      'line_flux_integral_prediction': df.line_flux_integral_prediction}))

all_catalogs = pd.concat(matched_detections)

truthcat = truthcat.assign(max_precision=0)
truthcat = truthcat.assign(line_flux_integral_prediction=0)
for i in np.unique(all_catalogs.match.values):
    truthcat.loc[int(i), 'line_flux_integral_prediction'] = all_catalogs[
        all_catalogs.match.values == int(i)].line_flux_integral_prediction.max()
    truthcat.loc[int(i), 'max_precision'] = all_catalogs[all_catalogs.match.values == int(i)].precision.max()

##

from matplotlib.gridspec import GridSpec

N_BINS = 50

filtered_truthcat = truthcat[0 < truthcat.max_precision]

fig = plt.figure()
gs = GridSpec(4, 4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[0, 0:3])
ax_hist_y = fig.add_subplot(gs[1:4, 3])

bins = np.logspace(np.log10(truthcat.line_flux_integral.min()), np.log10(truthcat.line_flux_integral.max()), N_BINS)
ax_hist_x.hist(filtered_truthcat.line_flux_integral, bins=bins,
               label=f'Detected sources ({len(filtered_truthcat)}, {100 * len(filtered_truthcat) / len(truthcat):.1f}\%)')
ax_hist_x.hist(truthcat.line_flux_integral, bins=bins, alpha=1, zorder=-1, label=f'All sources ({len(truthcat)})')
ax_hist_x.set_ylabel('Count')
ax_hist_x.legend()
ax_hist_x.tick_params(labelbottom=False)
ax_hist_x.set_xscale('log')

ax_scatter.scatter(filtered_truthcat.line_flux_integral, filtered_truthcat.max_precision, s=.1, c='k')

mp = filtered_truthcat.max_precision.copy().sort_values(ascending=False)
unique_mp = np.unique(mp.values)
unique_mp_index = np.array([np.argwhere(np.isclose(v, mp)).max() for v in unique_mp])

ax_hist_y.plot(unique_mp_index / len(mp), unique_mp)
ax_hist_y.set_xticks([0, .25, .5, .75, 1])
ax_hist_y.tick_params(axis='x', labelrotation=45)
ax_hist_y.tick_params(labelleft=False)
ax_hist_y.grid(True)
ax_hist_y.set_xlabel('Cumulative fraction detected')

ymin, ymax = ax_hist_y.get_ylim()
ys = np.linspace(ymin, ymax)

percents = 10
fluxes = np.percentile(filtered_truthcat.line_flux_integral, np.arange(0, 100 + percents, percents))
lower = np.zeros(len(fluxes) - 1)
upper = np.zeros(len(fluxes) - 1)
for i in range(len(fluxes) - 2):
    kde_data = filtered_truthcat[
        (fluxes[i] <= filtered_truthcat.line_flux_integral) & (filtered_truthcat.line_flux_integral <= fluxes[i + 1])][
        'max_precision']
    lower[i] = np.percentile(kde_data, 10)

    upper[i] = np.percentile(kde_data, 90)

lower[-1] = lower[-2]
upper[-1] = upper[-2]

ax_scatter.fill_between(np.repeat(fluxes, 2)[1:-1], np.repeat(lower, 2), np.repeat(upper, 2), zorder=-1, alpha=.5)

ax_scatter.set_xscale('log')
ax_scatter.set_ylim(ax_hist_y.get_ylim())
ax_scatter.set_xlim(ax_hist_x.get_xlim())
ax_scatter.xaxis.set_major_formatter(ScalarFormatter())
ax_scatter.set_xlabel('Line flux integral (Jy Hz)')
ax_scatter.set_ylabel('Max reliability')
ax_scatter.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
for text in ax_scatter.get_xminorticklabels():
    text.set_rotation(45)
    text.set_fontsize(7)
ax_scatter.tick_params(axis='x', labelrotation=45)
ax_scatter.grid(True, which='major')
ax_scatter.grid(True, which='minor')

plt.savefig('line_flux_reliability.pdf')
plt.show()
