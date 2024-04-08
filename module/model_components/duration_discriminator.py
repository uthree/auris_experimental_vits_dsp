import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXtLayer
from .normalization import LayerNorm


class DurationDiscriminator(nn.Module):
    def __init__(
            self,
            content_channels=192,
            speaker_embedding_dim=256,
            internal_channels=7,
            kernel_size=7,
            num_layers=4,
            ):
        super().__init__()
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.duration_input = nn.Conv1d(1, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mid_layers.append(ConvNeXtLayer(internal_channels, kernel_size))
        self.output_norm = LayerNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, 1, 1)

    def forward(self, x, x_mask, g, dur):
        x = self.speaker_input(g) + self.content_input(x) + self.duration_input(dur)
        x = x * x_mask
        x = self.input_norm(x)
        for layer in self.mid_layers:
            x = layer(x) * x_mask
        x = self.output_norm(x)
        x = self.output_layer(x)
        return x
