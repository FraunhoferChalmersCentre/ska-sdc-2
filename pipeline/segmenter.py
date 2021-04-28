from torch import nn
import pytorch_lightning as pl
import torch


class BaseSegmenter(pl.LightningModule):

    def __init__(self, model: nn.Module, scale, mean, std):
        super().__init__()
        self.model = model
        self.scale, self.mean, self.std = scale, mean, std

    def _get_model_input(self, image, frequency_channels):
        transformed_image = torch.empty(image.shape, device=self.device)
        start = frequency_channels[:, 0]
        stop = frequency_channels[:, 1]

        for i, (im, start_channel, end_channel) in enumerate(zip(image, start.int(), stop.int())):
            tr_im = im.T.clone()
            for j, (value_span, mu, sigma) in enumerate(
                    zip(self.scale[start_channel:end_channel], self.mean[start_channel:end_channel],
                        self.std[start_channel:end_channel])):
                tr_im[j] = (tr_im[j] - value_span[0]) / (value_span[1] - value_span[0])
                tr_im[j] = torch.clamp(tr_im[j], 0., 1.)
                tr_im[j] = (tr_im[j] - mu) / sigma
            transformed_image[i] = tr_im.T

        return transformed_image.float()

    def forward(self, image, frequency_channels):
        return self.model(self._get_model_input(image, frequency_channels))
