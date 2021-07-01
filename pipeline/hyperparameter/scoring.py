from typing import Dict

import pandas as pd
import numpy as np
import torch
from astropy.wcs import WCS
from torch import nn
from astropy.io.fits import Header
from sparse import COO
from torch.utils.tensorboard import SummaryWriter

from definitions import config
from pipeline.downstream import parametrise_sources
from pipeline.segmentation.scoring import LINEAR_SCATTER_ATTRS, ANGLE_SCATTER_ATTRS


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prediction_match(header, row, true_source):
    predicted_pos = np.array([row.ra, row.dec])
    true_pos = np.array([true_source['ra'], true_source['dec']]).flatten()

    line_width_freq = header['RESTFREQ'] * true_source['w20'] / config['constants']['speed_of_light']
    if np.linalg.norm(predicted_pos - true_pos) * 3600 < true_source['hi_size'] / 2 and np.abs(
            true_source['central_freq'] - row['central_freq']) < line_width_freq / 2:
        return True
    else:
        return False


def score_source(header, batch, matched, writer):
    true_attrs = {k: batch[k] for k in
                  ['ra', 'dec', 'central_freq', 'line_flux_integral', 'hi_size', 'w20', 'pa', 'i']}
    # matched = parametrized_df.loc[[prediction_match(header, r, batch) for i, r in parametrized_df.iterrows()]].copy()
    matched.loc[:, 'pos_error'] = np.sqrt(
        np.square(matched['ra'] - true_attrs['ra']) + np.square(matched['dec'] - true_attrs['dec'])) * 3600 / \
                                  true_attrs['hi_size']

    predictions = {}

    for attr in LINEAR_SCATTER_ATTRS:
        if attr in matched.columns:
            predictions[attr] = [matched[attr].mean(), true_attrs[attr].mean()]
            matched.loc[:, attr + '_error'] = np.abs(matched[attr] - true_attrs[attr]) / true_attrs[attr]

    for attr in ANGLE_SCATTER_ATTRS:
        if attr in matched.columns:
            predictions[attr] = [matched[attr].mean(), true_attrs[attr].mean()]
            theta = np.deg2rad(matched[attr] - true_attrs[attr])
            matched.loc[:, attr + '_error'] = np.abs(np.rad2deg(np.arctan2(np.sin(theta), np.cos(theta))))

    scores = {}
    for attr, threshold in config['scoring']['threshold'].items():
        if attr + '_error' in matched.columns:
            attr_score = np.clip(threshold / matched[attr + '_error'], a_min=0, a_max=1).mean()
            scores[attr] = attr_score
            writer.add_scalar(attr, attr_score)

    return matched, scores, predictions


def score_df(input_cube: torch.tensor, header: Header, model_out: torch.tensor, df_true: pd.DataFrame,
             segmentmap: np.ndarray, sofia_params: Dict, mask_threshold: float, min_intensity: float,
             writer: SummaryWriter):
    mask = torch.round(nn.Sigmoid()(model_out.to(torch.float32)) + 0.5 - mask_threshold).to(torch.float32)
    mask[mask > 1] = 1

    position = torch.zeros(2, 3)
    position[1] = torch.tensor(input_cube.shape)
    df_predicted = parametrise_sources(header, input_cube, mask, position, sofia_params, min_intensity=min_intensity)

    points = 0

    n_found = 0

    penalty = 0

    for i, row in df_predicted.iterrows():
        match = segmentmap[int(row.z_geo), int(row.y_geo), int(row.x_geo)]
        print(match)
        if match == 0:
            penalty += 1
            continue

        n_found += 1

        matched, scores, predictions = score_source(header, df_true.loc[int(match)], df_predicted.loc[[i]], writer)
        points += np.mean(list(scores.values()))

    precision = n_found / (n_found + penalty)
    recall = n_found / len(df_true)

    writer.add_scalar('n_found', n_found)
    writer.add_scalar('precision', precision)
    writer.add_scalar('recall', recall)
    writer.add_scalar('f1', 2 * (precision * recall) / (precision + recall))
    writer.add_scalar('avg_point', points / n_found)
    writer.add_scalar('penalty', penalty)
    writer.add_scalar('points', points)
    writer.add_scalar('score', points - penalty)

    return points - penalty
