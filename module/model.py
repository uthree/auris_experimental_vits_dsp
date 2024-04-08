import math

import torch
import torch.nn as nn

from .model_components.generator import Generator
from .model_components.discriminator import Discriminator
from .model_components.duration_discriminator import DurationDiscriminator


class VITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(config['generator'])
        self.discriminator = Discriminator(**config['discriminator'])
        self.duration_discriminator = DurationDiscriminator(**config['duration_discriminator'])
