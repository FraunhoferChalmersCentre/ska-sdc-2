##
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from definitions import config

from visualization.utils import get_pareto, get_params_df, get_predicted_dfs, get_truthcat

N_SPLITS = 30
FLUX_MAX_PERCENTILE = 100
COST_MAX_FILTER = 1 - 1e-10
ATTRIBUTE = 'line_flux_integral'

plt.rcParams['text.usetex'] = True

predicted_cataloges = get_predicted_dfs()
truthcat = get_truthcat()
params_df = get_params_df(predicted_cataloges, truthcat)
predicted_catalogues = {k: v for k, v in zip(params_df.index, get_predicted_dfs())}

params_df = get_pareto(params_df)

params_df = params_df[params_df.cost_max >= COST_MAX_FILTER]
predicted_catalogues = {k: predicted_catalogues[k] for k in params_df.index}

truthcat = get_truthcat()

##
flux_min = truthcat.loc[:, ATTRIBUTE].min()
flux_max = np.percentile(truthcat.loc[:, ATTRIBUTE], FLUX_MAX_PERCENTILE)
fluxes = np.logspace(np.log(flux_min), np.log(flux_max), N_SPLITS + 1, base=np.e)


##

def binned_flux(s): return np.array([
    np.sum((fluxes[i] < s) * ((s < fluxes[i + 1]) | np.isclose(s, fluxes[i + 1]))) for i in range(N_SPLITS)])


##
flux_wise_completeness = pd.DataFrame(columns=params_df.index, index=fluxes[1:])

for i, row in params_df.iterrows():
    # Find the line flux integral of true source
    matched_sources = predicted_catalogues[i][
        config['iou_threshold'] <= predicted_catalogues[i].iou].match.values.astype(np.int32)
    match_line_flux = truthcat.loc[matched_sources, ATTRIBUTE]
    # Get the number of sources by flux bin
    flux_wise_completeness[i] = binned_flux(match_line_flux.values)

flux_wise_completeness = flux_wise_completeness.T

##

fig, ax = plt.subplots()

all_fluxes = np.logspace(np.log(flux_min), np.log(flux_max), 100 + 1, base=np.e)
ax.hist(truthcat.loc[:, ATTRIBUTE], range=(0, flux_max), bins=all_fluxes[1:], color='k', alpha=.3)
ax.set_ylabel('Source count')
ax.set_xlabel('Line flux integral (Jy Hz)')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())

ax1 = ax.twinx()
ax1.set_ylim([0, 1])
ax1.set_ylabel('Completeness')

total_source_count = binned_flux(truthcat.loc[:, ATTRIBUTE])

params_df = params_df.assign(normalized_precision=(params_df.precision - params_df.precision.min()) / (
        params_df.precision.max() - params_df.precision.min()))

cmap = mpl.cm.viridis
for i, row in params_df.iterrows():
    ax1.plot(fluxes[1:], flux_wise_completeness.loc[i] / total_source_count, c=cmap(row.normalized_precision), alpha=.8)

# Add colorbar
norm = mpl.colors.Normalize(vmin=params_df.precision.min(), vmax=params_df.precision.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, label='Reliability', orientation='horizontal')

plt.tight_layout()
plt.savefig('line_flux_completeness.pdf')
plt.show()
