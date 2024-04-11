import torch
import torch.nn as nn
from .wn import WN


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 n_fft=3840,
                 frame_size=960,
                 internal_channels=192,
                 speaker_embedding_dim=256,
                 content_channels=192,
                 kernel_size=5,
                 dilation=1,
                 num_layers=16):
        super().__init__()

        self.input_channels = n_fft // 2 + 1
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.pre = nn.Conv1d(self.input_channels, internal_channels, 1)
        self.wn = WN(internal_channels, kernel_size, dilation, num_layers, speaker_embedding_dim)
        self.post = nn.Conv1d(internal_channels, content_channels * 2, 1)

    # x: [BatchSize, fft_bin, Length]
    # x_length: [BatchSize]
    # g: [BatchSize, speaker_embedding_dim, 1]
    #
    # Outputs:
    #   z: [BatchSize, content_channels, Length]
    #   mean: [BatchSize, content_channels, Length]
    #   logvar: [BatchSize, content_channels, Length]
    #   z_mask: [BatchSize, 1, Length]
    #
    # where fft_bin = input_channels = n_fft // 2 + 1
    def forward(self, x, x_length, g):
        # generate mask
        max_length = x.shape[2]
        progression = torch.arange(max_length, dtype=x_length.dtype, device=x_length.device)
        z_mask = (progression.unsqueeze(0) < x_length.unsqueeze(1))
        z_mask = z_mask.unsqueeze(1).to(x.dtype)
        
        # pass network
        x = self.pre(x) * z_mask
        x = self.wn(x, z_mask, g) 
        x = self.post(x) * z_mask
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(logvar) * z_mask
        return z, mean, logvar, z_mask
