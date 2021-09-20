import pickle
from datetime import datetime

import numpy as np
from astropy.io.fits import getheader
from hyperopt import hp, fmin, tpe
from hyperopt.fmin import generate_trials_to_calculate
from sofia import readoptions

from definitions import config, ROOT_DIR
from pipeline.common import filename
from pipeline.hyperparameter.tuning import PrecisionRecallTradeoffTuner

checkpoint = config['traversing']['checkpoint']

test_set_path = filename.processed.test_dataset(checkpoint)

sofia_params = readoptions.readPipelineOptions(ROOT_DIR + config['downstream']['sofia']['param_file'])

alphas = np.linspace(0, 1, 100)

n_trials = 10
performed = 0
factor = .99

space = {'radius_spatial': hp.uniform('radius_spatial', .5, 5),
         'radius_freq': hp.uniform('radius_freq', .5, 100),
         'min_size_spatial': hp.uniform('min_size_spatial', .5, 5),
         'min_size_freq': hp.uniform('min_size_freq', 1, 50),
         'max_size_spatial': hp.uniform('max_size_spatial', 5, 30),
         'max_size_freq': hp.uniform('max_size_freq', 50, 300),
         'min_voxels': hp.uniform('min_voxels', 1, 300),
         'dilation_max_spatial': hp.uniform('dilation_max_spatial', .5, 5),
         'dilation_max_freq': hp.uniform('dilation_max_freq', .5, 20),
         'mask_threshold': hp.uniform('mask_threshold', 1e-2, 1),
         'min_intensity': hp.uniform('min_intensity', 0, 30),
         'max_intensity': hp.uniform('max_intensity', 200, 1000)
         }

init_values = [{'radius_spatial': 1,
                'radius_freq': 1,
                'min_size_spatial': 1,
                'min_size_freq': 1,
                'max_size_spatial': 30,
                'max_size_freq': 300,
                'min_voxels': 1,
                'dilation_max_spatial': sofia_params['parameters']['dilatePixMax'],
                'dilation_max_freq': sofia_params['parameters']['dilateChanMax'],
                'mask_threshold': 1e-2,
                'min_intensity': 0,
                'max_intensity': 100
                }]

trials = generate_trials_to_calculate(init_values)
header = getheader(filename.data.test_sky())

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
trials_log_file = ROOT_DIR + f'/hparam_logs/{timestamp}.pb'

for i, alpha in enumerate(alphas):
    name = f'{checkpoint}/{alpha:.2f}'

    if len(trials.results) > 0:
        for r in trials.results:
            r['loss'] = - (alpha * r['precision'] + (1 - alpha) * r['recall'])

    tuner = PrecisionRecallTradeoffTuner(alpha, config['hyperparameters']['threshold'], init_values[0]['min_intensity'],
                                         init_values[0]['max_intensity'], sofia_params, test_set_path, header,
                                         name=name)

    best = fmin(tuner.produce_score, space, algo=tpe.suggest, max_evals=performed + int(np.round(n_trials)),
                trials=trials, trials_save_file=trials_log_file)
    performed += int(np.round(n_trials))
    n_trials = n_trials * factor