import torch
import torch.nn as nn

from .convnext import ConvNeXtLayer


# audio to latent feature without speaker infomation
class AudioEncoder(nn.Module):
    def __init__(
            self,
            n_fft=3840,
            internal_channels=192,
            content_channels=192,
            kernel_size=7,
            num_layers=4):
        super().__init__()
        self.input_channels = n_fft // 2 + 1
        self.pre = nn.Conv1d(self.input_channels, internal_channels, 1)
        self.mid_layers = nn.ModuleList(
                [ConvNeXtLayer(internal_channels, kernel_size) for _ in range(num_layers)])
        self.post = nn.Conv1d(internal_channels, content_channels, 1)

    # x: [BatchSize, fft_bin, Length]
    # x_length: [BatchSize]
    #
    # Outputs:
    #   x: [BatchSize, content_channels, Length]
    #   x_mask: [BatchSize, 1, Length]
    #
    # where fft_bin = input_channels = n_fft // 2 + 1
    def forward(self, x, x_length):
        # generate mask
        max_length = x.shape[2]
        progression = torch.arange(max_length, dtype=x_length.dtype, device=x_length.device)
        x_mask = (progression.unsqueeze(0) < x_length.unsqueeze(1))
        x_mask = x_mask.unsqueeze(1).to(x.dtype)

        # pass network
        x = self.pre(x) * x_mask
        for layer in self.mid_layers:
            x = layer(x) * x_mask
        x = self.post(x) * x_mask
        return x, x_mask
