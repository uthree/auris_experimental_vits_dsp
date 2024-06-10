import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class HarmonicOscillator(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            frame_size=480,
            num_harmonics=0,
            min_frequency=20.0,
            noise_scale=0.03
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.min_frequency = min_frequency
        self.num_harmonics = num_harmonics
        self.noise_scale = noise_scale

    def forward(self, f0):
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
        mul = (torch.arange(self.num_harmonics+1, device=f0.device) + 1).unsqueeze(0).unsqueeze(2)
        fs = f0 * mul
        uv = (f0 >= self.min_frequency).to(torch.float)
        integrated = torch.cumsum(fs / self.sample_rate, dim=2)
        theta = 2 * math.pi * (integrated % 1)
        source = torch.sin(theta) * uv + torch.randn_like(theta) * self.noise_scale
        return source


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), 1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.gelu(x)
            xt = c1(xt)
            xt = F.gelu(xt)
            xt = c2(xt)
            x = x + xt
        return x
    

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.gelu(x)
            xt = c1(xt)
            x = x + xt
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            content_channels=192,
            speaker_embedding_dim=256,
            sample_rate=24000,
            frame_size=480,
            num_harmonics=0,
            upsample_initial_channels=512,
            resblock_type="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[24, 20, 4, 4],
            upsample_rates=[12, 10, 2, 2]
        ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.oscillator = HarmonicOscillator(sample_rate, frame_size, num_harmonics=num_harmonics)
        self.conv_pre = weight_norm(nn.Conv1d(content_channels, upsample_initial_channels, 7, 1, 3))
        self.speaker_condition = weight_norm(nn.Conv1d(speaker_embedding_dim, upsample_initial_channels, 1))
        ch_last = upsample_initial_channels//(2**(self.num_upsamples))
        self.source_pre = weight_norm(nn.Conv1d(num_harmonics+1, ch_last, 7, 1, 3))
        self.ups = nn.ModuleList()
        downs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            downs.append(weight_norm(nn.Conv1d(c2, c1, k, u, p)))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
        self.downs = nn.ModuleList(reversed(downs))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, g, f0):
        s = self.oscillator(f0)

        skips = []
        s = self.source_pre(s)
        for i in range(self.num_upsamples):
            skips.append(s)
            s = F.gelu(s)
            s = self.downs[i](s)
        skips = list(reversed(skips))

        x = self.conv_pre(x) + self.speaker_condition(g) + s
        for i in range(self.num_upsamples):
            x = F.gelu(x)
            x = self.ups[i](x)
            x = x + skips[i]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.gelu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x