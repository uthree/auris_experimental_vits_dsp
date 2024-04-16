import torch
import torch.nn as nn
from .normalization import LayerNorm
from .convnext import ConvNeXtLayer

class DurationDiscriminator(nn.Module):
    def __init__(
            self,
            content_channels=192,
            speaker_embedding_dim=256,
            internal_channels=192,
            num_layers=3,
    ):
        super().__init__()
        self.text_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.duration_input = nn.Conv1d(1, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(internal_channels) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, 1, 1)

    def forward(
            self,
            text_encoded,
            text_mask,
            log_duration,
            spk,
    ):
        x = self.text_input(text_encoded) + self.duration_input(log_duration) + self.speaker_input(spk)
        x = self.input_norm(x)
        x = self.mid_layers(x) * text_mask
        x = self.output_norm(x)
        x = self.output_layer(x) * text_mask
        return x
