import torch
import torch.nn as nn
from .wn import WN


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 n_fft=3840,
                 internal_channels=192,
                 speaker_embedding_dim=256,
                 content_channels=192,
                 kernel_size=5,
                 dilation=1,
                 num_layers=16):
        super().__init__()

        self.input_channels = n_fft // 2 + 1
        self.pre = nn.Conv1d(self.input_channels, internal_channels, 1)
        self.wn = WN(internal_channels, kernel_size, dilation, num_layers, speaker_embedding_dim)
        self.post = nn.Conv1d(internal_channels, content_channels * 2)

    # x: [BatchSize, fft_bin, Length]
    # Output: [BatchSize, content_channels, Length]
    # where fft_bin = input_channels = n_fft // 2 + 1
    def forward(self, x, spk):
        x = self.pre(x)
        x = self.wn(x, spk)
        x = self.post(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(logvar)
        return z, mean, logvar
