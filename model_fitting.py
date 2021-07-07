from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_toolbelt import losses
from astropy.io import fits
import numpy as np

from definitions import config, ROOT_DIR
from pipeline.segmentation.training import TrainSegmenter
from pipeline.common import filename
from pipeline.segmentation.utils import get_data, get_checkpoint_callback, get_random_vis_id, get_base_segmenter, \
    get_equibatch_samplers

training_set, validation_set = get_data(robust_validation=config['segmentation']['robust_validation'])
base_segmenter = get_base_segmenter()
checkpoint_callback = get_checkpoint_callback()

train_sampler, val_sampler = get_equibatch_samplers(training_set, validation_set,
                                                                 config['segmentation']['robust_validation'])

segmenter = TrainSegmenter(base_segmenter,
                           loss_fct=losses.JointLoss(losses.DiceLoss(mode='binary'),
                                                     losses.SoftBCEWithLogitsLoss(), 1.0, 1.0),
                           training_set=training_set,
                           validation_set=validation_set,
                           header=fits.getheader(filename.data.sky(config['segmentation']['size'])),
                           optimizer=torch.optim.Adam(base_segmenter.parameters(), lr=1e-3),
                           vis_id=get_random_vis_id(training_set, random_state=np.random.RandomState(10)),
                           batch_size=config['segmentation']['batch_size'],
                           threshold=config['hyperparameters']['threshold'],
                           random_mirror=config['segmentation']['augmentation'],
                           random_rotation=config['segmentation']['augmentation'],
                           robust_validation=config['segmentation']['robust_validation'],
                           train_sampler=train_sampler,
                           val_sampler=val_sampler,
                           check_val_every_n_epoch=10
                           )

logger = TensorBoardLogger("tb_logs",
                           name=config['segmentation']['model_name'],
                           version=datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                           )
trainer = pl.Trainer(max_epochs=100000, gpus=1, logger=logger, callbacks=[checkpoint_callback])

trainer.fit(segmenter)
