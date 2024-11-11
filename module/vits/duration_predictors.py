import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import piecewise_rational_quadratic_transform
from .normalization import LayerNorm
from .convnext import ConvNeXtLayer


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class StochasticDurationPredictor(nn.Module):
    def __init__(
            self,
            in_channels=192,
            filter_channels=256,
            kernel_size=5,
            p_dropout=0.0,
            n_flows=4,
            speaker_embedding_dim=256
            ):
        super().__init__()
        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(Flip())

        self.pre = nn.Linear(in_channels, filter_channels)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.proj = nn.Linear(filter_channels, filter_channels)

        self.post_pre = nn.Linear(1, filter_channels)
        self.post_convs = DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_proj = nn.Linear(filter_channels, filter_channels)

        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(Flip())

        if speaker_embedding_dim != 0:
            self.cond = nn.Linear(speaker_embedding_dim, filter_channels)


    # x: [BatchSize, in_chanels, Length]
    # x_mask [BatchSize, 1, Length]
    # g: [BatchSize, speaker_embedding_dim, 1]
    # w: Optional, Training only shape=[BatchSize, 1, Length]
    # Output: [BatchSize, 1, Length]
    #
    # note: g is speaker embedding
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x.mT).mT
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g.mT).mT
        x = self.convs(x, x_mask)
        x = self.proj(x.mT).mT * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w.mT).mT
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w.mT).mT * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class ConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Linear(self.half_channels, filter_channels)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Linear(filter_channels, self.half_channels * (num_bins * 3 - 1))
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0.mT).mT
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h.mT).mT * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=reverse, tails="linear", tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class DDSConv(nn.Module):
    """
    Dilated and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding))
            self.linears.append(nn.Linear(channels, channels))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.linears[i](y.mT).mT
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


# TODO convert to class method
class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class DurationPredictor(nn.Module):
    def __init__(self,
                 content_channels=192,
                 internal_channels=256,
                 speaker_embedding_dim=256,
                 kernel_size=7,
                 num_layers=4):
        super().__init__()
        self.phoneme_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mid_layers.append(ConvNeXtLayer(internal_channels, kernel_size))
        self.output_norm = LayerNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, 1, 1)

    def forward(self, x, x_mask, g):
        x = x.detach()
        g = g.detach()
        x = (self.phoneme_input(x) + self.speaker_input(g)) * x_mask
        x = self.input_norm(x) * x_mask
        for layer in self.mid_layers:
            x = layer(x) * x_mask
        x = self.output_norm(x) * x_mask
        x = self.output_layer(x) * x_mask
        x = F.relu(x)
        return x
