from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_toolbelt import losses
from astropy.io import fits
import numpy as np

from definitions import config, ROOT_DIR
from pipeline.data.ska_dataset import TrainingItemGetter
from pipeline.segmentation.training import TrainSegmenter
from pipeline.common import filename
from pipeline.segmentation.utils import get_data, get_checkpoint_callback, get_random_vis_id, get_base_segmenter, \
    get_equibatch_samplers, get_full_validator
from pipeline.segmentation.validation import SimpleValidator

training_set, validation_set = get_data(full_set_validation=False, validation_item_getter=TrainingItemGetter())
base_segmenter = get_base_segmenter()

loss_fct = losses.JointLoss(losses.DiceLoss(mode='binary'), losses.SoftBCEWithLogitsLoss(), 1.0, 1.0)
validator = SimpleValidator(base_segmenter, {'val_loss': loss_fct})
checkpoint_callback = get_checkpoint_callback(period=config['segmentation']['validation']['interval'])

train_sampler, val_sampler = get_equibatch_samplers(training_set, validation_set, only_training=False)

segmenter = TrainSegmenter(base_segmenter,
                           loss_fct=loss_fct,
                           training_set=training_set,
                           validation_set=validation_set,
                           header=fits.getheader(filename.data.sky(config['segmentation']['size'])),
                           validator=validator,
                           optimizer=torch.optim.Adam(base_segmenter.parameters(), lr=1e-3),
                           vis_id=get_random_vis_id(training_set, random_state=np.random.RandomState(10)),
                           batch_size=config['segmentation']['batch_size'],
                           threshold=config['hyperparameters']['threshold'],
                           random_mirror=config['segmentation']['augmentation'],
                           random_rotation=config['segmentation']['augmentation'],
                           train_sampler=train_sampler,
                           val_sampler=val_sampler
                           )

logger = TensorBoardLogger("tb_logs",
                           name=config['segmentation']['model_name'],
                           version=datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                           )
trainer = pl.Trainer(max_epochs=100000, gpus=1, logger=logger, callbacks=[checkpoint_callback],
                     check_val_every_n_epoch=config['segmentation']['validation']['interval'])

trainer.fit(segmenter)
