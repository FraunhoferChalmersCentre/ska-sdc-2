import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from definitions import config

eps = 1e-5

N = 1000


def get_pareto(df):
    df = df.assign(cost_max=0, alpha_max=0, pareto_sol=False)
    for alpha in np.linspace(eps, 1 - eps, N):
        relative_cost = alpha * df.recall + (1 - alpha) * df.precision
        ranking = relative_cost.sort_values(ascending=False).index
        relative_cost /= relative_cost.max()

        alpha_filter = df.cost_max < relative_cost

        df.loc[alpha_filter, 'cost_max'] = relative_cost[alpha_filter]
        df.loc[alpha_filter, 'best_rank'] = np.array(
            [(ranking == i).argmax() for i in df.index[alpha_filter]])
        df.loc[alpha_filter, 'alpha_max'] = alpha

        df.loc[relative_cost == 1, 'pareto_sol'] = True

    return df


def get_params_df(predicted_dfs, truthcat):
    trials = pickle.load(open(f"param_sets/{config['param_set']}.pb", "rb"))

    params_df = pd.DataFrame.from_dict(trials.results)
    params_df = pd.concat([params_df, pd.DataFrame.from_dict(trials.vals)], axis=1)
    params_df = params_df.loc[params_df.status == 'ok', :]

    params_df = params_df.assign(
        precision=[adjust_precision(predicted_dfs[i]) for i in range(len(params_df))])
    params_df = params_df.assign(
        recall=[adjust_recall(predicted_dfs[i], len(truthcat)) for i in range(len(params_df))])

    return params_df


def get_predicted_dfs():
    trials = pickle.load(open(f"param_sets/{config['param_set']}.pb", "rb"))
    dfs = []

    for i in range(len(trials.results)):
        if trials.results[i]['status'] == 'ok':
            if 'df' in trials.results[i].keys():
                dfs.append(pd.DataFrame.from_dict(trials.results[i]['df']))
            else:
                dfs.append(pd.DataFrame())

    return dfs


def adjust_precision(predicted_df):
    return sum(config['iou_threshold'] < predicted_df.iou) / len(predicted_df)


def adjust_recall(predicted_df, n_true):
    return sum(config['iou_threshold'] < predicted_df.iou) / n_true


def get_truthcat():
    return pd.read_csv(config['truth_catalogue'], sep=' ', index_col='id')
