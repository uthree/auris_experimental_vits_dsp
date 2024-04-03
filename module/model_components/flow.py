import torch
import torch.nn as nn

from .wn import WN


class Flip(nn.Module):
    def forward(self, x, *args, **kwargs):
        x = torch.flip(x, dim=1)
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 in_channels=192,
                 internal_channels=192,
                 speaker_embedding_dim=256,
                 kernel_size=5,
                 dilation=1,
                 num_layers=4):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, internal_channels, 1)
        self.wn = WN(internal_channels, kernel_size, dilation, num_layers, speaker_embedding_dim)
        self.post = nn.Conv1d(internal_channels, self.half_channels, 1)

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, spk, reverse=False):
        x_0, x_1 = torch.chunk(x, 2, dim=1)
        h = self.pre(x_0)
        h = self.wn(h, spk)
        x_1_mean = self.post(h)

        if not reverse:
            x_1 = x1_mean + x1
        else:
            x_1 = (x_1 - x_1_mean)

        x = torch.cat([x_0, x_1], dim=1)
        return x


class Flow(nn.Module):
    def __init__(self,
                 content_channels=192,
                 internal_channels=192,
                 speaker_embedding_dim=256,
                 ketnel_size=5,
                 dilation=1,
                 num_flows=4,
                 num_layers=4):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(num_flows):
            self.flows.append(
                    ResidualCouplingLayer(
                        content_channels,
                        internal_channels,
                        speaker_embedding_dim,
                        kernel_size,
                        dilation,
                        num_layers))
            self.flows.append(Flip())

    # z: [BatchSize, content_channels, Length]
    # spk: [Batchsize, speaker_embedding_dim, 1]
    # Output: [BatchSize, content_channels, Length]
    def forward(self, z, spk, reverse=False):
        if not reverse:
            for flow in self.flows:
                z = flow(z, spk, reverse=False)
        else:
            for flow in reversed(self.flows):
                z = flow(z, spk, reverse=True)
        return z
