import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .pitch_estimator import PitchEstimator
from .posterior_encoder import PosteriorEncoder
from .speaker_embedding import SpeakerEmbedding
from .flow import Flow
from .text_encoder import TextEncoder
from .duration_predictors import StochasticDurationPredictor, DurationPredictor

from .crop import crop_features


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    padding_shape = [[0, 0], [1, 0], [0, 0]]
    padding = [item for sublist in padding_shape[::-1] for item in sublist]

    path = path - F.pad(path, padding)[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path


class Generator(nn.Module):
    # initialize from config
    def __init__(self, config):
        super().__init__()
        self.posterior_encoder = PosteriorEncoder(**config.posterior_encoder)
        self.decoder = Decoder(**config.decoder)
        self.flow = Flow(**config.flow)
        self.text_encoder = TextEncoder(**config.text_encoder)
        self.stochastic_duration_predictor = StochasticDurationPredictor(**config.stochastic_duration_predictor)
        self.duration_predictor = DurationPredictor(**config.duration_predictor)
        self.speaker_embedding = SpeakerEmbedding(**config.speaker_embedding)