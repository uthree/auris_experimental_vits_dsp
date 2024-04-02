import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm, remove_weight_norm
from .utils import get_padding, init_weights


# Oscillate harmonic signal
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
# uv: [BatchSize, 1, Frames], voiced = 1.0, unvocied = 0.0
#
# frame_size: int
# num_harmonics: int
# min_frequency: float
# noise_scale: float
#
# Output: [BatchSize, NumHarmonics, Length]
#
# length = Frames * frame_size
def oscillate_harmonics(f0,
                        uv,
                        frame_size=960,
                        sample_rate=48000,
                        num_harmonics=0):
    N = f0.shape[0]
    C = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(C, device=device) + 1).unsqueeze(0).unsqueeze(2).expand(N, C, Lw)

    # change length to wave's
    fs = F.interpolate(f0, Lw, mode='linear') * mul

    # unvoiced / voiced mask
    uv = F.interpolate(uv, Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    theta = 2 * math.pi * (I % 1) # convert to radians

    harmonics = torch.sin(theta) * uv

    return harmonics.to(device)


# Oscillate aperiodic signal
#
# fft_bin = n_fft // 2 + 1
# kernels: [BatchSize, fft_bin, Frames]
#
# Output: [BatchSize, 1, Frames * frame_size]
def oscillate_aperiodic_signal(kernels, frame_size=960, n_fft=3840):
    device = kernels.device
    N = kernels.shape[0]
    Lf = kernels.shape[2] # frame length
    Lw = Lf * frame_size # waveform length
    dtype = kernels.dtype()

    gaussian_noise = torch.randn(N, Lw, device=device, dtype=torch.float)
    kernels = kernels.to(torch.float) # to fp32

    # calculate convolution in fourier-domain
    # Since the input is an aperiodic signal such as Gaussian noise,
    # there is no need to consider the phase on the kernel side.
    w = torch.hann_window(n_fft, dtype=torch.float, device=device)
    noise_stft = torch.stft(gaussian_noise, n_fft, frame_size, window=w)[:, :, 1:]
    y_stft = noise_stft * kernels # In fourier domain, Multiplication means convolution.
    y_stft = F.pad(y_stft, [1, 0]) # pad
    y = torch.istft(y_stft, n_fft, frame_size, window=w)
    y = y.unsqueeze(1)
    y = y.to(dtype)
    return y


class FiLM(nn.Module):
    def __init__(self, in_channels, condition_channels):
        super().__init__()
        self.to_shift = weight_norm(nn.Conv1d(condition_channels, in_channels, 1))
        self.to_scale = weight_norm(nn.Conv1d(condition_channels, in_channels, 1))

    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift

    def remove_weight_norm(self):
        remove_weight_norm(self.to_shift)
        remove_weight_norm(self.to_scale)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ConvNeXt v2 + GeGLU
class ConvNeXtLayer(nn.Module):
    def __init__(self, channels=512, kernel_size=7, mlp_mul=3):
        super().__init__()
        padding = kernel_size // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, groups=channels)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.grn = GRN(channels * mlp_mul)
        self.c4 = nn.Conv1d(channels * mlp_mul, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x) * F.gelu(self.c3(x))
        x = self.grn(x)
        x = self.c4(x)
        x = x + res
        return x


# this model estimates fundamental frequency (f0) from latent content
class PitchEstimator(nn.Module):
    def __init__(self,
                 content_channels=192,
                 internal_channels=512,
                 num_classes=512,
                 kernel_size=7,
                 num_layers=6,
                 mlp_mul=3,
                 min_frequency=20.0,
                 classes_per_octave=48,
                 ):
        super().__init__()
        self.content_channels = content_channels
        self.num_classes = num_classes
        self.classes_per_octave = classes_per_octave
        self.min_frequency = min_frequency

        self.input_layer = nn.Conv1d(content_channels, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXt(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, num_classes, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.mid_layers(x)
        x = self.output_norm(x)
        logits = self.output_layer(x)
        return logits

    def freq2id(self, f):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        nc = self.num_classes
        return torch.ceil(torch.clamp(cpo * torch.log2(f / fmin), 0, nc-1)).to(torch.long)

    def id2freq(self, ids):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        nc = self.num_classes

        x = ids.to(torch.float)
        x = fmin * (2 ** (x / cpo))
        x[x <= self.f0_min] = 0
        return x

    def infer(self, x, k=4):
        logits = self.forward(x)
        probs, indices = torch.topk(logits, k, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.freq2id(indices)
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.min_frequency] = 0
        return f0


class SourceNet(nn.Module):
    def __init__(self,
                 content_channels=192,
                 internal_channels=512,
                 sample_rate=48000,
                 frame_size=960,
                 n_fft=3840,
                 num_harmonics=30,
                 kernel_size=7,
                 num_layers=6,
                 mlp_mul=3):
        super().__init__()
        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.f0_input = nn.Conv1d(1, internal_channels)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXt(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.to_amps = nn.Conv1d(internal_channels, num_harmonics + 1, 1)
        self.to_kernels = nn.Conv1d(internal_channels, n_fft // 2 + 1, 1)

    def forward(self, x, f0):
        x = self.content_input(x) + self.f0_input(torch.log(F.relu(f0) + 1e-6))
        x = self.input_norm(x)
        x = self.mid_layers(x)
        x = output_norm(x)
        amps = self.to_amps(x)
        kernels = self.to_kernels(x)
        return amps, kernels


class ResBlock1(nn.Module):
    def __init__(self, channels, condition_channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])
        self.films = nn.ModuleList([])

        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs1.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
            self.convs2.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, 1, padding_mode='replicate')))
            self.films.append(
                    FiLM(channels, condition_channels))

    def forward(self, x, c):
        for c1, c2, film in zip(self.convs1, self.convs2, self.films):
            res = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = film(x, c)
            x = x + res
        return x

    def remove_weight_norm(self):
        for c1, c2, film in zip(self.convs1, self.convs2, self.films):
            remove_weight_norm(c1)
            remove_weight_norm(c2)
            film.remove_weight_norm()


class MRF(nn.Module):
    def __init__(self,
                 channels,
                 condition_channels,
                 resblock_type='1',
                 kernel_sizes=[3, 7, 11],
                 dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.num_blocks = len(kernel_sizes)
        if resblock_type == '1':
            block = ResBlock1
        for k, d in zip(kernel_sizes, dilations):
            self.blocks.append(block(channels, condition_channels, k, d))

    def forward(self, x, c):
        xs = None
        for block in self.blocks:
            if xs is None:
                xs = block(x, c)
            else:
                xs += block(x, c)
        return xs / self.num_blocks

    def remove_weight_norm(self):
        for block in self.blocks:
            block.remove_weight_norm()


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 condition_channels,
                 factor,
                 resblock_type='1',
                 kernel_sizes=[3, 7, 11],
                 dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.MRF = MRF(out_channels, condition_channels, resblock_type, kernel_sizes, dilations)
        self.up_conv = weight_norm(
                nn.ConvTranspose1d(in_channels, out_channels, factor*2, factor))
        self.pad_left = factor // 2
        self.pad_right = factor - pad_left

    def forward(self, x, c):
        x = F.leaky_relu(x, 0.1)
        x = self.up_conv(x)
        x = x[:, :, self.pad_left:-self.pad_right]
        x = self.mrf(x, c)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.up_conv)
        self.MRF.remove_weight_norm()


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 factor,
                 dilations=[[1, 2], [4, 8]],
                 kernel_size=3):
        super().__init__()
        self.convs = nn.ModuleList([])
        for ds in dilations:
            cs = nn.ModuleList([])
            for d in ds:
                padding = get_padding(kernel_size, d)
                cs.append(
                        weight_norm(
                            nn.Conv1d(in_channels, in_channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
            self.convs.append(cs)
        pad_left = factor // 2
        pad_right factor - pad_left
        self.pad = nn.ReplicationPad1d([pad_left, pad_right])
        self.output_conv = weignt_norm(nn.Conv1d(in_channels, out_channels, factor*2, factor))

    def forward(self, x):
        for block in self.convs:
            res = x
            for c in block:
                x = F.leaky_relu(x, 0.1)
                x = c(x)
            x = x + res
        x = self.pad(x)
        x = self.output_conv(x)
        return x

    def remove_weight_norm(self):
        for block in self.convs:
            for c in block:
                remove_weight_norm(c)
        remove_weight_norm(self.output_conv)


class FilterNet(nn.Module):
    def __init__(self,
                 content_channels=192,
                 channels=[512, 256, 128, 64, 32],
                 resblock_type='1',
                 factors=[5, 4, 4, 4, 3],
                 up_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 up_kernel_sizes=[3, 7, 11],
                 down_dilations=[[1, 2], [4, 8]],
                 num_harmonics=30,
                 ):
        super().__init__()
        # input layer
        self.content_input = weight_norm(nn.Conv1d(input_channels, channels[0], 1))
        self.f0_input = weight_norm(nn.Conv1d(1, channels[0], 1))

        # downsamples
        self.downs = nn.ModuleList([])
        self.downs.append(weight_norm(nn.Conv1d(num_harmonics + 2, channels[-1], 1)))
        cs = list(reversed(channels[1:]))
        ns = cs[1:] + [channels[0]]
        fs = list(reversed(factors[1:]))
        for c, n, f, in zip(cs, ns, fs):
            self.downs.append(DownBlock(c, n, f, down_dilations))

        # upsamples
        self.ups = nn.ModuleList([])
        cs = channels
        ns = channels[1:] + [channels[-1]]
        fs = factors
        for c, n, f in zip(cs, ns, fs):
            self.ups.append(UpBlock(c, n, c, f))
        self.output_layer = weight_norm(
                nn.Conv1d(channels[-1], 1, 7, 1, 3, padding_mode='replicate'))

    def forward(self, content, f0, source):
        x = self.content_input(content) + self.f0_input(torch.log(F.relu(f0) + 1e-6))

        skips = []
        for down in self.downs:
            source = down(source)
            skips.append(source)

        for up, s in zip(self.ups, reversed(skips)):
            x = up(x, s)
        x = self.output_layer(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.content_in)
        remove_weight_norm(self.output_layer)
        for down in self.downs:
            down.remove_weight_norm()
        for up in self.ups:
            up.remove_weight_norm()


class Decoder(nn.Module):
    def __init__(self,
                 sample_rate=48000,
                 frame_size=960,
                 n_fft=3840,
                 content_channels=192,
                 pe_internal_channels=512,
                 pe_num_layers=6,
                 source_internal_channels=512,
                 source_num_layers=6,
                 num_harmonics=30,
                 filter_channels=[512, 256, 128, 64, 32],
                 filter_factors=[5, 4, 4, 4, 3],
                 filter_resblock_type='1',
                 filter_down_dilations=[[1, 2], [4, 8]],
                 filter_up_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 filter_up_kernel_sizes=[3, 7, 11]):
        super().__init__()
        self.pitch_estimator = PitchEstimator(
                content_channels,
                pe_internal_channels,
                num_layers=pe_num_layers)
        self.source_net = SourceNet(
                content_channels,
                source_internal_channels,
                source_num_layers,
                frame_size,
                n_fft,
                num_harmonics)
        self.filter_net = FilterNet(
                content_channels,
                filter_channels,
                filter_resblock_type,
                filter_factors,
                filter_up_dilations,
                filter_up_kernel_sizes,
                filter_down_dilations,
                num_harmonics)
