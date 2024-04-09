import math

import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator

class VITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(config['generator'])
        self.discriminator = Discriminator(**config['discriminator'])

