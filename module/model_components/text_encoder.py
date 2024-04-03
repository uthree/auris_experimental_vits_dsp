import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Encoding from https://arxiv.org/abs/1706.03762
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
