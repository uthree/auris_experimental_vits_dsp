import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm


# WN module from https://arxiv.org/abs/1609.03499
class WNLayer(nn.Module):
    def __init__(self,
                 hidden_channels=192,
                 kernel_size=5,
                 dilation=1,
                 speaker_embedding_dim=256):
        super().__init__()
        self.speaker_in = weight_norm(
                nn.Conv1d(
                    speaker_embedding_dim,
                    hidden_channels, 1))
        padding = int((kernel_size * dilation - dilation) / 2)
        self.conv = weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels * 2,
                    kernel_size,
                    1,
                    padding,
                    dilation=dilation,
                    padding_mode='replicate'))
        self.out = weight_norm(
                nn.Conv1d(hidden_channels, hidden_channels * 2, 1))

    # x: [BatchSize, hidden_channels, Length]
    # x_mask: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Output: [BatchSize, hidden_channels, Length]
    def forward(self, x, x_mask, spk):
        res = x
        x = x + self.speaker_in(spk)
        x = self.conv(x)
        x_0, x_1 = torch.chunk(x, 2, dim=1)
        x = torch.tanh(x_0) * torch.sigmoid(x_1)
        x = self.out(x)
        out, skip = torch.chunk(x, 2, dim=1)
        out = (out + res) * x_mask
        skip = skip * x_mask
        return out, skip

    def remove_weight_norm(self, x):
        remove_weight_norm(self.speaker_in)
        remove_weight_norm(self.conv)
        remove_weight_norm(self.out)


class WN(nn.Module):
    def __init__(self,
                 hidden_channels=192,
                 kernel_size=5,
                 dilation=1,
                 num_layers=4,
                 speaker_embedding_dim=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                    WNLayer(hidden_channels, kernel_size, dilation, speaker_embedding_dim))

    # x: [BatchSize, hidden_channels, Length]
    # x_mask: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Output: [BatchSize, hidden_channels, Length]
    def forward(self, x, x_mask, spk):
        output = None
        for layer in self.layers:
            x, skip = layer(x, x_mask, spk)
            if output is None:
                output = skip
            else:
                output += skip
        return output
