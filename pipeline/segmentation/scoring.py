from datetime import datetime

import numpy as np
import pandas as pd
import sparse

from definitions import config, ROOT_DIR
from pipeline.common.filename import DirectoryFileName

LINEAR_SCATTER_ATTRS = ['central_freq', 'w20', 'line_flux_integral', 'hi_size']
ANGLE_SCATTER_ATTRS = ['pa', 'i']


def score_source(true_df: pd.DataFrame, matched_prediction_df: pd.DataFrame):
    predictions = {}

    for attr in LINEAR_SCATTER_ATTRS:
        if attr in matched_prediction_df.columns:
            predictions[attr] = [matched_prediction_df[attr].mean(), true_df[attr].mean()]
            matched_prediction_df.loc[:, attr + '_error'] = np.abs(matched_prediction_df[attr] - true_df[attr]) / \
                                                            true_df[attr]

    for attr in ANGLE_SCATTER_ATTRS:
        if attr in matched_prediction_df.columns:
            predictions[attr] = [matched_prediction_df[attr].mean(), true_df[attr].mean()]
            theta = np.deg2rad(matched_prediction_df[attr] - true_df[attr])
            matched_prediction_df.loc[:, attr + '_error'] = np.abs(np.rad2deg(np.arctan2(np.sin(theta), np.cos(theta))))

    scores = {}
    for attr, threshold in config['scoring']['threshold'].items():
        if attr + '_error' in matched_prediction_df.columns:
            attr_score = np.clip(threshold / matched_prediction_df[attr + '_error'], a_min=0, a_max=1).mean()
            scores[attr] = attr_score

    return len(matched_prediction_df), scores, predictions


def score_df(df_predicted: pd.DataFrame, df_true: pd.DataFrame, segmentmap: sparse.COO):
    total_points = 0

    n_matched = 0

    total_penalty = 0

    df_predicted['points'] = 0
    df_predicted['penalty'] = 0

    metrics = {}

    for attr in LINEAR_SCATTER_ATTRS + ANGLE_SCATTER_ATTRS:
        df_predicted[f'{attr}_score'] = 0
        df_predicted[f'{attr}_prediction'] = 0
        df_predicted[f'{attr}_target'] = 0

    if len(df_predicted) > 0:
        for i, row in df_predicted.iterrows():
            for attr in LINEAR_SCATTER_ATTRS + ANGLE_SCATTER_ATTRS:
                df_predicted.loc[i, f'{attr}_prediction'] = row[attr]
            try:
                match = segmentmap[int(row.x_geo), int(row.y_geo), int(row.z_geo)]

                matched, scores, predictions = score_source(df_true.loc[int(match)], df_predicted.loc[[i]])
            except (IndexError, KeyError):
                print('Index or key error')
                match = -1
            print(match)
            if match == 0:
                total_penalty += config['scoring']['fp_penalty']
                df_predicted.loc[i, 'penalty'] = config['scoring']['fp_penalty']
            elif match > 0:
                n_matched += 1

                for attr in LINEAR_SCATTER_ATTRS + ANGLE_SCATTER_ATTRS:
                    df_predicted.loc[i, f'{attr}_score'] = scores[attr]
                    df_predicted.loc[i, f'{attr}_target'] = predictions[attr][1]

                points = np.mean(list(scores.values()))
                df_predicted.loc[i, 'total_points'] = points
                total_points += points

        metrics['precision'] = n_matched / len(df_predicted)
        metrics['recall'] = n_matched / len(df_true)
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0

        for attr in LINEAR_SCATTER_ATTRS + ANGLE_SCATTER_ATTRS:
            attrs_scores = df_predicted.loc[df_predicted[f'{attr}_score'] > 0, f'{attr}_score']
            metrics[f'sdc2_score/{attr}'] = attrs_scores.mean()

        metrics['detected'] = len(df_predicted)
        metrics['matched'] = n_matched
        metrics['sdc2_penalty'] = total_penalty
        metrics['sdc2_points'] = total_points
        metrics['sdc2_score'] = total_points - total_penalty

    return metrics, df_predicted
