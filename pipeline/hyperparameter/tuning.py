from datetime import datetime

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import numpy as np

iteration = 0


class Tuner:
    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.iteration = 0

    def tuning_objective(self, args):
        try:

            with open("notebooks/hyperparams.txt", "a+") as file_object:
                file_object.write(str(self.iteration) + '\t' + str(args) + '\n')

            version = str(self.iteration) + datetime.now().strftime("_%Y%m%d_%H%M%S")
            name = 'hyperparam' + version

            self.segmenter.name = name

            self.segmenter.threshold = args['mask_threshold']
            self.segmenter.sofia_parameters['merge']['radiusX'] = int(np.round(args['radius_spatial']))
            self.segmenter.sofia_parameters['merge']['radiusY'] = int(np.round(args['radius_spatial']))
            self.segmenter.sofia_parameters['merge']['radiusZ'] = int(np.round(args['radius_freq']))
            self.segmenter.sofia_parameters['merge']['minSizeX'] = int(np.round(args['min_size_spatial']))
            self.segmenter.sofia_parameters['merge']['minSizeY'] = int(np.round(args['min_size_spatial']))
            self.segmenter.sofia_parameters['merge']['minSizeZ'] = int(np.round(args['min_size_freq']))
            self.segmenter.sofia_parameters['merge']['minVoxels'] = int(np.round(args['min_voxels']))
            self.segmenter.sofia_parameters['parameters']['dilatePixMax'] = int(np.round(args['dilation_max_spatial']))
            self.segmenter.sofia_parameters['parameters']['dilateChanMax'] = int(np.round(args['dilation_max_freq']))

            logger = TensorBoardLogger("tb_logs", name="hyperparam", version=version)
            logger.log_metrics({'hparams/' + k: v for k, v in args.items()})
            trainer = pl.Trainer(gpus=0, logger=logger)
            results = trainer.validate(self.segmenter)[0]

            points = results['point_epoch']

            self.iteration += 1

            return - points
        except Exception as err:
            print(err)
            return float('inf')
