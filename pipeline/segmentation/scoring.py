import numpy as np

from definitions import config

LINEAR_SCATTER_ATTRS = ['central_freq', 'w20', 'line_flux_integral', 'hi_size']
ANGLE_SCATTER_ATTRS = ['pa', 'i']


def prediction_match(header, row, batch):
    predicted_pos = np.array([row.ra, row.dec])
    true_pos = np.array([batch['ra'], batch['dec']]).flatten()

    line_width_freq = header['RESTFREQ'] * batch['w20'] / config['constants']['speed_of_light']
    if np.linalg.norm(predicted_pos - true_pos) * 3600 < batch['hi_size'] / 2 and np.abs(
            batch['central_freq'] - row['central_freq']) < line_width_freq / 2:
        return True
    else:
        return False


def score_source(header, batch, matched):
    true_attrs = {k: batch[k] for k in
                  ['ra', 'dec', 'central_freq', 'line_flux_integral', 'hi_size', 'w20', 'pa', 'i']}

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

    return len(matched), scores, predictions


def parametrisation_validation(header, parametrized_df, batch, has_source):
    if has_source:
        # There is a source in this
        for i, row in parametrized_df.iterrows():
            if prediction_match(header, row, batch):
                return True
        return False
    else:
        # No sources in this batch
        return len(parametrized_df) > 0
