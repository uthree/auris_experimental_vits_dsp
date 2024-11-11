import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm
from module.utils.common import get_padding, init_weights
from .normalization import LayerNorm
from .convnext import ConvNeXtLayer
from .feature_retrieval import match_features


# Oscillate harmonic signal
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
#
# frame_size: int
# sample_rate: float or int
# min_frequency: float
# num_harmonics: int
#
# Output: [BatchSize, NumHarmonics+1, Length]
#
# length = Frames * frame_size
@torch.amp.autocast('cuda', enabled=False)
def oscillate_harmonics(
        f0,
        frame_size=960,
        sample_rate=48000,
        num_harmonics=0,
        min_frequency=20.0
    ):
    N = f0.shape[0]
    C = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(C, device=device) + 1).unsqueeze(0).unsqueeze(2)

    # change length to wave's
    fs = F.interpolate(f0, Lw, mode='linear') * mul

    # unvoiced / voiced mask
    uv = (f0 > min_frequency).to(torch.float)
    uv = F.interpolate(uv, Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    theta = 2 * math.pi * (I % 1) # convert to radians

    harmonics = torch.sin(theta) * uv

    return harmonics.to(device)


# Oscillate full harmonics signal using the BLIT (transformed expression)
# Adapted from: https://github.com/TylorShine/FIR-SVC/blob/04fd374b9de84f93688888f4dab89aa689a7f84d/modules/vocoder.py#L285
# 
# Inputs ---
# f0: [BatchSize, 1, Frames]
#
# frame_size: int
# sample_rate: float or int
# min_frequency: float
#
# Output: [BatchSize, 1, Length]
#
# length = Frames * frame_size
@torch.amp.autocast("cuda", enabled=False)
def oscillate_full_harmonics(
        f0,
        frame_size=960,
        sample_rate=48000,
        min_frequency=20.0,
    ):
    
    N, C, Lf = f0.shape
    Lw = Lf * frame_size
    
    # print(f0)
    
    # change length to wave's
    fs = F.interpolate(f0, Lw, mode='linear')
    
    # unvoiced / voiced mask
    uv = (f0 > min_frequency).to(torch.float)
    uv = F.interpolate(uv, Lw, mode='nearest')
    
    pix = torch.cumsum(math.pi*fs/sample_rate, dim=2) % (math.pi)

    a = torch.round(sample_rate/torch.clamp(fs, min=20.)*0.5).flatten(1).unsqueeze(1)*2. + 1.
    sinpix = torch.sin(pix)
    harmonic_source = torch.where(pix < 1e-8, 1., torch.sin(a*pix) / (a*sinpix)) * uv

    return harmonic_source


# Oscillate noise via gaussian noise and equalizer
#
# fft_bin = n_fft // 2 + 1
# kernels: [BatchSize, fft_bin, Frames]
#
# Output: [BatchSize, 1, Frames * frame_size]
@torch.amp.autocast('cuda', enabled=False)
def oscillate_noise(kernels, frame_size=960, n_fft=3840):
    device = kernels.device
    N = kernels.shape[0]
    Lf = kernels.shape[2] # frame length
    Lw = Lf * frame_size # waveform length
    dtype = kernels.dtype

    # gaussian_noise = torch.randn(N, Lw, device=device, dtype=torch.float) # gaussian noise makes a 'pulsey' result
    uniform_noise = torch.rand(N, Lw, device=device, dtype=torch.float)     # so use uniform noise instead
    kernels = kernels.to(torch.float) # to fp32

    # calculate convolution in fourier-domain
    # Since the input is an aperiodic signal such as Gaussian noise,
    # there is no need to consider the phase on the kernel side.
    w = torch.hann_window(n_fft, dtype=torch.float, device=device)
    noise_stft = torch.stft(uniform_noise, n_fft, frame_size, window=w, return_complex=True)[:, :, 1:]
    y_stft = noise_stft * kernels # In fourier domain, Multiplication means convolution.
    y_stft = F.pad(y_stft, [1, 0]) # pad
    y = torch.istft(y_stft, n_fft, frame_size, window=w)
    y = y.unsqueeze(1)
    y = y.to(dtype)
    return y


# this model estimates fundamental frequency (f0) and energy from z
class PitchEnergyEstimator(nn.Module):
    def __init__(self,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 internal_channels=256,
                 num_classes=512,
                 kernel_size=7,
                 num_layers=6,
                 mlp_mul=3,
                 min_frequency=20.0,
                 max_frequency=1600.0,
                 ):
        super().__init__()
        self.content_channels = content_channels
        self.num_classes = num_classes
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXtLayer(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.to_f0_logits = nn.Conv1d(internal_channels, num_classes, 1)
        self.to_energy = nn.Conv1d(internal_channels, 1, 1)
        
        self.log2_fmax_fmin = math.log2(self.max_frequency/self.min_frequency)

    # z_p: [BatchSize, content_channels, Length]
    # spk: [BatchSize, speaker_embedding_dim, Length]
    def forward(self, z_p, spk):
        x = self.content_input(z_p) + self.speaker_input(spk)
        x = self.input_norm(x)
        x = self.mid_layers(x)
        x = self.output_norm(x)
        logits, energy = self.to_f0_logits(x), self.to_energy(x)
        energy = F.elu(energy) + 1.0
        return logits, energy

    # f: [<Any shape allowed>]
    def freq2id(self, f):
        fmin = self.min_frequency
        fmax = self.max_frequency
        # cpo = self.classes_per_octave
        nc = self.num_classes
        # return torch.ceil(torch.clamp(cpo * torch.log2(f / fmin), 0, nc-1)).to(torch.long)
        return torch.ceil(nc * torch.log2(torch.clamp(f, fmin, fmax)/fmin)/self.log2_fmax_fmin).to(torch.long)
    
    # ids: [<Any shape allowed>]
    def id2freq(self, ids):
        fmin = self.min_frequency
        fmax = self.max_frequency
        # cpo = self.classes_per_octave
        nc = self.num_classes
        x = ids.to(torch.float)
        # x = fmin * (2 ** (x / cpo))
        x = fmin * ((fmax/fmin) ** (x / nc))
        x[x <= self.min_frequency] = 0
        return x
    
    def logit2freq(self, logits, k=4, threshold=0.03):
        probs, indices = torch.topk(logits, k, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.id2freq(indices)
        confidence = probs.max(1, keepdim=True).values
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.min_frequency] = 0
        f0 = torch.masked_fill(f0, confidence <= threshold, 0)
        return f0
    
    # z_p: [BatchSize, content_channels, Length]
    # spk: [BatchSize, speaker_embedding_dim, Length]
    # Outputs:
    #   f0: [BatchSize, 1, Length]
    #   energy: [BatchSize, 1, Length]
    def infer(self, z_p, spk, k=4, threshold=0.03):
        logits, energy = self.forward(z_p, spk)
        f0 = self.logit2freq(logits, k, threshold)
        return f0, energy


class FiLM(nn.Module):
    def __init__(self, in_channels, condition_channels):
        super().__init__()
        self.to_shift = weight_norm(nn.Conv1d(condition_channels, in_channels, 1))
        self.to_scale = weight_norm(nn.Conv1d(condition_channels, in_channels, 1))

    # x: [BatchSize, in_channels, Length]
    # c: [BatchSize, condition_channels, Length]
    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift

    def remove_weight_norm(self):
        remove_weight_norm(self.to_shift)
        remove_weight_norm(self.to_scale)


class SourceNet(nn.Module):
    def __init__(
                self,
                sample_rate=48000,
                n_fft=3840,
                frame_size=960,
                content_channels=192,
                speaker_embedding_dim=256,
                internal_channels=512,
                num_harmonics=30,
                kernel_size=7,
                num_layers=6,
                mlp_mul=3
            ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics

        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.energy_input = nn.Conv1d(1, internal_channels, 1)
        self.f0_input = nn.Conv1d(1, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXtLayer(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.to_amps = nn.Conv1d(internal_channels, num_harmonics + 1, 1)
        self.to_kernels = nn.Conv1d(internal_channels, n_fft // 2 + 1, 1)

    # x: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # energy: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Outputs:
    #   amps: [BatchSize, num_harmonics+1, Length * frame_size]
    #   kernels: [BatchSize, n_fft //2 + 1, Length]
    def amps_and_kernels(self, x, f0, energy, spk):
        x = self.content_input(x) + self.speaker_input(spk) + self.f0_input(torch.log(F.relu(f0) + 1e-6))
        x = x * self.energy_input(energy)
        x = self.input_norm(x)
        x = self.mid_layers(x)
        x = self.output_norm(x)
        amps = F.elu(self.to_amps(x)) + 1.0
        kernels = F.elu(self.to_kernels(x)) + 1.0
        return amps, kernels


    # x: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Outputs:
    #   dsp_out: [BatchSize, 1, Length * frame_size]
    #   source: [BatchSize, 1, Length * frame_size]
    def forward(self, x, f0, energy, spk):
        amps, kernels = self.amps_and_kernels(x, f0, energy, spk)

        # oscillate source signals
        harmonics = oscillate_harmonics(f0, self.frame_size, self.sample_rate, self.num_harmonics)
        amps = F.interpolate(amps, scale_factor=self.frame_size, mode='linear')
        harmonics = harmonics * amps
        noise = oscillate_noise(kernels, self.frame_size, self.n_fft)
        source = torch.cat([harmonics, noise], dim=1)
        dsp_out = torch.sum(source, dim=1, keepdim=True)

        return dsp_out, source


# HiFi-GAN's ResBlock1
class ResBlock1(nn.Module):
    def __init__(self, channels, condition_channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])
        self.films = nn.ModuleList([])

        for d in dilations:
            padding = get_padding(kernel_size, 1)
            self.convs1.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
            padding = get_padding(kernel_size, d)
            self.convs2.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, 1, padding_mode='replicate')))
            self.films.append(
                    FiLM(channels, condition_channels))
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.films.apply(init_weights)

    # x: [BatchSize, channels, Length]
    # c: [BatchSize, condition_channels, Length]
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


# HiFi-GAN's ResBlock2
class ResBlock2(nn.Module):
    def __init__(self, channels, condition_channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.films = nn.ModuleList([])
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
            self.films.append(FiLM(channels, condition_channels))
        self.convs.apply(init_weights)
        self.films.apply(init_weights)

    # x: [BatchSize, channels, Length]
    # c: [BatchSize, condition_channels, Length]
    def forward(self, x, c):
        for conv, film in zip(self.convs, self.films):
            res = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = film(x, c)
            x = x + res
        return x

    def remove_weight_norm(self):
        for conv, film in zip(self.convs, self.films):
            conv.remove_weight_norm()
            film.remove_weight_norm()


# TinyVC's Block (from https://github.com/uthree/tinyvc)
class ResBlock3(nn.Module):
    def __init__(self, channels, condition_channels, kernel_size=3, dilations=[1, 3, 9, 27]):
        super().__init__()
        assert len(dilations) == 4, "Resblock 3's len(dilations) should be 4."
        self.convs = nn.ModuleList([])
        self.films = nn.ModuleList([])
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                    weight_norm(
                        nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
        for _ in range(2):
            self.films.append(
                    FiLM(channels, condition_channels))

    # x: [BatchSize, channels, Length]
    # c: [BatchSize, condition_channels, Length]
    def forward(self, x, c):
        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.convs[0](x)
        x = F.leaky_relu(x, 0.1)
        x = self.convs[1](x)
        x = self.films[0](x, c)
        x = x + res

        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.convs[2](x)
        x = F.leaky_relu(x, 0.1)
        x = self.convs[3](x)
        x = self.films[1](x, c)
        x = x + res

        return x

    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)
        for film in self.films:
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
        elif resblock_type == '2':
            block = ResBlock2
        elif resblock_type == '3':
            block = ResBlock3
        for k, d in zip(kernel_sizes, dilations):
            self.blocks.append(block(channels, condition_channels, k, d))

    # x: [BatchSize, channels, Length]
    # c: [BatchSize, condition_channels, Length]
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
                 dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 interpolation='conv'):
        super().__init__()
        self.MRF = MRF(out_channels, condition_channels, resblock_type, kernel_sizes, dilations)
        self.interpolation = interpolation
        self.factor = factor
        if interpolation == 'conv':
            self.up_conv = weight_norm(
                    nn.ConvTranspose1d(in_channels, out_channels, factor*2, factor))
            self.pad_left = factor // 2
            self.pad_right = factor - self.pad_left
        elif interpolation == 'linear':
            self.up_conv = weight_norm(nn.Conv1d(in_channels, out_channels, 1))

    # x: [BatchSize, in_channels, Length]
    # c: [BatchSize, condition_channels, Length(upsampled)]
    # Output: [BatchSize, out_channels, Length(upsampled)]
    def forward(self, x, c):
        x = F.leaky_relu(x, 0.1)
        if self.interpolation == 'conv':
            x = self.up_conv(x)
            x = x[:, :, self.pad_left:-self.pad_right]
        elif self.interpolation == 'linear':
            x = self.up_conv(x)
            x = F.interpolate(x, scale_factor=self.factor)
        x = self.MRF(x, c)
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
                 kernel_size=3,
                 interpolation='conv'):
        super().__init__()
        pad_left = factor // 2
        pad_right = factor - pad_left
        self.pad = nn.ReplicationPad1d([pad_left, pad_right])
        self.interpolation = interpolation
        self.factor = factor
        if interpolation == 'conv':
            self.input_conv = weight_norm(nn.Conv1d(in_channels, out_channels, factor*2, factor))
        elif interpolation == 'linear':
            self.input_conv = weight_norm(nn.Conv1d(in_channels, out_channels, 1))

        self.convs = nn.ModuleList([])
        for ds in dilations:
            cs = nn.ModuleList([])
            for d in ds:
                padding = get_padding(kernel_size, d)
                cs.append(
                        weight_norm(
                            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, dilation=d, padding_mode='replicate')))
            self.convs.append(cs)


    # x: [BatchSize, in_channels, Length]
    # Output: [BatchSize, out_channels, Length]
    def forward(self, x):
        if self.interpolation == 'conv':
            x = self.pad(x)
            x = self.input_conv(x)
        elif self.interpolation == 'linear':
            x = self.pad(x)
            x = F.avg_pool1d(x, self.factor*2, self.factor) # approximation of linear interpolation
            x = self.input_conv(x)
        for block in self.convs:
            res = x
            for c in block:
                x = F.leaky_relu(x, 0.1)
                x = c(x)
            x = x + res
        return x

    def remove_weight_norm(self):
        for block in self.convs:
            for c in block:
                remove_weight_norm(c)
        remove_weight_norm(self.output_conv)


class FilterNet(nn.Module):
    def __init__(self,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 channels=[512, 256, 128, 64, 32],
                 resblock_type='1',
                 factors=[5, 4, 4, 4, 3],
                 up_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 up_kernel_sizes=[3, 7, 11],
                 up_interpolation='conv',
                 down_dilations=[[1, 2], [4, 8]],
                 down_kernel_size=3,
                 down_interpolation='conv',
                 num_harmonics=30,
                 ):
        super().__init__()
        # input layer
        self.content_input = weight_norm(nn.Conv1d(content_channels, channels[0], 1))
        self.speaker_input = weight_norm(nn.Conv1d(speaker_embedding_dim, channels[0], 1))
        self.energy_input = weight_norm(nn.Conv1d(1, channels[0], 1))
        self.f0_input = weight_norm(nn.Conv1d(1, channels[0], 1))

        # downsamples
        self.downs = nn.ModuleList([])
        self.downs.append(weight_norm(nn.Conv1d(num_harmonics + 2, channels[-1], 1)))
        cs = list(reversed(channels[1:]))
        ns = cs[1:] + [channels[0]]
        fs = list(reversed(factors[1:]))
        for c, n, f, in zip(cs, ns, fs):
            self.downs.append(DownBlock(c, n, f, down_dilations, down_kernel_size, down_interpolation))

        # upsamples
        self.ups = nn.ModuleList([])
        cs = channels
        ns = channels[1:] + [channels[-1]]
        fs = factors
        for c, n, f in zip(cs, ns, fs):
            self.ups.append(UpBlock(c, n, c, f, resblock_type, up_kernel_sizes, up_dilations, up_interpolation))
        self.output_layer = weight_norm(
                nn.Conv1d(channels[-1], 1, 7, 1, 3, padding_mode='replicate'))

    # content: [BatchSize, content_channels, Length(frame)]
    # f0: [BatchSize, 1, Length(frame)]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # source: [BatchSize, num_harmonics+2, Length(Waveform)]
    # Output: [Batchsize, 1, Length * frame_size]
    def forward(self, content, f0, energy, spk, source):
        x = self.content_input(content) + self.speaker_input(spk) + self.f0_input(torch.log(F.relu(f0) + 1e-6))
        x = x * self.energy_input(energy)

        skips = []
        for down in self.downs:
            source = down(source)
            skips.append(source)

        for up, s in zip(self.ups, reversed(skips)):
            x = up(x, s)
        x = self.output_layer(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.content_input)
        remove_weight_norm(self.output_layer)
        remove_weight_norm(self.speaker_input)
        for down in self.downs:
            down.remove_weight_norm()
        for up in self.ups:
            up.remove_weight_norm()


def pitch_estimation_loss(logits, label):
    num_classes = logits.shape[1]
    device = logits.device
    weight = torch.ones(num_classes, device=device)
    weight[0] = 1e-2
    return F.cross_entropy(logits, label, weight)


class Decoder(nn.Module):
    def __init__(self,
                 sample_rate=48000,
                 frame_size=960,
                 n_fft=3840,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 pe_internal_channels=256,
                 pe_num_layers=6,
                 source_internal_channels=512,
                 source_num_layers=6,
                 num_harmonics=30,
                 filter_channels=[512, 256, 128, 64, 32],
                 filter_factors=[5, 4, 4, 3, 2],
                 filter_resblock_type='1',
                 filter_down_dilations=[[1, 2], [4, 8]],
                 filter_down_kernel_size=3,
                 filter_down_interpolation='conv',
                 filter_up_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 filter_up_kernel_sizes=[3, 7, 11],
                 filter_up_interpolation='conv'):
        super().__init__()
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.pitch_energy_estimator = PitchEnergyEstimator(
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                num_layers=pe_num_layers,
                internal_channels=pe_internal_channels
                )
        self.source_net = SourceNet(
                sample_rate=sample_rate,
                frame_size=frame_size,
                n_fft=n_fft,
                num_harmonics=num_harmonics,
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                internal_channels=source_internal_channels,
                num_layers=source_num_layers,
                )
        self.filter_net = FilterNet(
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                channels=filter_channels,
                resblock_type=filter_resblock_type,
                factors=filter_factors,
                up_dilations=filter_up_dilations,
                up_kernel_sizes=filter_up_kernel_sizes,
                up_interpolation=filter_up_interpolation,
                down_dilations=filter_down_dilations,
                down_kernel_size=filter_down_kernel_size,
                down_interpolation=filter_down_interpolation,
                num_harmonics=num_harmonics
                )
    # training pass
    #
    # content: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    #
    # Outputs:
    #   f0_logits [BatchSize, num_f0_classes, Length]
    #   estimated_energy: [BatchSize, 1, Length]
    #   dsp_out: [BatchSize, Length * frame_size]
    #   output: [BatchSize, Length * frame_size]
    def forward(self, content, f0, energy, spk):
        # estimate energy, f0
        f0_logits, estimated_energy = self.pitch_energy_estimator(content, spk)
        f0_label = self.pitch_energy_estimator.freq2id(f0).squeeze(1)
        loss_pe = pitch_estimation_loss(f0_logits, f0_label)
        loss_ee = (estimated_energy - energy).abs().mean()
        loss = loss_pe * 45.0 + loss_ee * 45.0
        loss_dict = {
            "Pitch Estimation": loss_pe.item(),
            "Energy Estimation": loss_ee.item()
        }

        # source net
        dsp_out, source = self.source_net(content, f0, energy, spk)
        dsp_out = dsp_out.squeeze(1)

        # GAN output
        output = self.filter_net(content, f0, energy, spk, source)
        output = output.squeeze(1)

        return dsp_out, output, loss, loss_dict

    # inference pass
    #
    # content: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # Output: [BatchSize, 1, Length * frame_size]
    # energy: [BatchSize, 1, Length]
    # f0: [BatchSize, 1, Length]
    # reference: None or [BatchSize, content_channels, NumReferenceVectors]
    # alpha: float 0 ~ 1.0
    # k: int
    def infer(self, content, spk, energy=None, f0=None, reference=None, alpha=0, k=4, metrics='cos'):
        if energy is None or f0 is None:
            f0_est, energy_est = self.pitch_energy_estimator.infer(content, spk)
        if energy is None:
            energy = energy_est
        if f0 is None:
            f0 = f0_est

        # run feature retrieval if got reference vectors
        if reference is not None:
            content = match_features(content, reference, k, alpha, metrics)

        dsp_out, source = self.source_net(content, f0, energy, spk)

        # filter network
        output = self.filter_net(content, f0, energy, spk, source)
        return output
    
    def estimate_pitch_energy(self, content, spk):
        f0, energy = self.pitch_energy_estimator.infer(content, spk)
        return f0, energy

    
class FirSourceNet(nn.Module):
    def __init__(
                self,
                sample_rate=48000,
                n_fft=3840,
                frame_size=960,
                content_channels=192,
                speaker_embedding_dim=256,
                internal_channels=512,
                kernel_size=7,
                num_layers=6,
                mlp_mul=3
            ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.frame_size = frame_size

        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.energy_input = nn.Conv1d(1, internal_channels, 1)
        self.f0_input = nn.Conv1d(1, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXtLayer(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.to_amps = nn.Conv1d(internal_channels, 1, 1)
        self.to_kernels = nn.Conv1d(internal_channels, n_fft // 2 + 1, 1)

    # x: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # energy: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Outputs:
    #   amps: [BatchSize, 1, Length * frame_size]
    #   kernels: [BatchSize, n_fft //2 + 1, Length]
    def amps_and_kernels(self, x, f0, energy, spk):
        x = self.content_input(x) + self.speaker_input(spk) + self.f0_input(torch.log(F.relu(f0) + 1e-6))
        x = x + self.energy_input(energy)
        x = self.input_norm(x)
        x = self.mid_layers(x)
        x = self.output_norm(x)
        amps = F.elu(self.to_amps(x)) + 1.0
        kernels = F.elu(self.to_kernels(x)) + 1.0
        return amps, kernels


    # x: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # Outputs:
    #   dsp_out: [BatchSize, 1, Length * frame_size]
    def forward(self, x, f0, energy, spk):
        amps, kernels = self.amps_and_kernels(x, f0, energy, spk)

        # oscillate source signals
        harmonics = oscillate_full_harmonics(f0, self.frame_size, self.sample_rate)
        amps = F.interpolate(amps, scale_factor=self.frame_size, mode='linear')
        harmonics = harmonics * amps
        noise = oscillate_noise(kernels, self.frame_size, self.n_fft)
        source = torch.cat([harmonics, noise], dim=1)
        # source = torch.cat([harmonics, harmonics], dim=1)
        dsp_out = torch.sum(source, dim=1, keepdim=True)

        return dsp_out, dsp_out
    
    
class FirFilterNet(nn.Module):
    def __init__(self,
                 n_window=1024,
                 frame_size=960,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 internal_channels=512,
                 kernel_size=7,
                 num_layers=6,
                 mlp_mul=3
                 ):
        super().__init__()
        self.n_window = n_window
        self.frame_size = frame_size
        # input layer
        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.energy_input = nn.Conv1d(1, internal_channels, 1)
        self.f0_input = nn.Conv1d(1, internal_channels, 1)
        self.mid_layers = nn.Sequential(
                *[ConvNeXtLayer(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.output_layer = nn.Conv1d(internal_channels, n_window, 7, 1, 3, padding_mode='replicate')

    # content: [BatchSize, content_channels, Length(frame)]
    # f0: [BatchSize, 1, Length(frame)]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    # source: [BatchSize, 1, Length(Waveform)]
    # Output: [Batchsize, 1, Length * frame_size]
    def forward(self, content, f0, energy, spk, source):
        x = self.content_input(content) + self.speaker_input(spk) + self.f0_input(torch.log(F.relu(f0) + 1e-6))
        x = x + self.energy_input(energy)
        
        x = self.mid_layers(x)
        x = self.output_norm(x)
        filt = self.output_layer(x).transpose(1, 2)
        
        # filter source
        B, C, L = source.shape
        source_t = source.reshape(B, -1, self.frame_size)
        source_t = F.pad(source_t, (self.n_window, self.n_window-1)).unsqueeze(1)
        bi, bj, c, d = source_t.shape
        source_t = source_t.transpose(1, 0).view(bj, bi * c, d)
        filt_v = filt.view(bi, -1, filt.shape[2])
        filt = filt_v.reshape(bi*filt_v.shape[1], 1, -1)
        source = F.conv1d(source_t, filt, groups=bi*filt_v.shape[1]).view(bj, bi, filt_v.shape[1], -1).squeeze(0)
        
        # overlap and add
        output_size = L + self.n_window
        source = F.fold(
            source.transpose(2, 1),
            output_size=(1, output_size),
            kernel_size=(1, self.frame_size + self.n_window),
            stride=(1, self.frame_size)).squeeze(1)

        return source[:, :, :L]

    def remove_weight_norm(self):
        pass


class FirDecoder(nn.Module):
    def __init__(self,
                 sample_rate=48000,
                 frame_size=960,
                 n_fft=3840,
                 n_window=1024,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 pe_internal_channels=256,
                 pe_num_layers=6,
                 source_internal_channels=512,
                 source_num_layers=6,
                 filter_internal_channels=512,
                 filter_num_layers=6):
        super().__init__()
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.n_window = n_window
        self.sample_rate = sample_rate
        self.pitch_energy_estimator = PitchEnergyEstimator(
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                num_layers=pe_num_layers,
                internal_channels=pe_internal_channels
                )
        self.source_net = FirSourceNet(
                sample_rate=sample_rate,
                n_fft=n_fft,
                frame_size=frame_size,
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                internal_channels=source_internal_channels,
                num_layers=source_num_layers,
                )
        self.filter_net = FirFilterNet(
                n_window=n_window,
                frame_size=frame_size,
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                internal_channels=filter_internal_channels,
                num_layers=filter_num_layers,
                )
    # training pass
    #
    # content: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    #
    # Outputs:
    #   f0_logits [BatchSize, num_f0_classes, Length]
    #   estimated_energy: [BatchSize, 1, Length]
    #   dsp_out: [BatchSize, Length * frame_size]
    #   output: [BatchSize, Length * frame_size]
    def forward(self, content, f0, energy, spk):
        # estimate energy, f0
        f0_logits, estimated_energy = self.pitch_energy_estimator(content, spk)
        f0_label = self.pitch_energy_estimator.freq2id(f0).squeeze(1)
        loss_pe = pitch_estimation_loss(f0_logits, f0_label)
        estimated_f0 = self.pitch_energy_estimator.logit2freq(f0_logits)
        loss_pe_freq = F.l1_loss(torch.log2(estimated_f0 + 1.), torch.log2(f0 + 1.))
        loss_ee = (estimated_energy - energy).abs().mean()
        loss = loss_pe * 45.0 + loss_pe_freq + loss_ee * 45.0
        loss_dict = {
            "Pitch Logits Estimation": loss_pe.item(),
            "Pitch Estimation": loss_pe_freq.item(),
            "Energy Estimation": loss_ee.item()
        }

        # source net
        dsp_out, source = self.source_net(content, f0, energy, spk)
        dsp_out = dsp_out.squeeze(1)

        # GAN output
        output = self.filter_net(content, f0, energy, spk, source)
        output = output.squeeze(1)

        return dsp_out, output, loss, loss_dict

    # inference pass
    #
    # content: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # Output: [BatchSize, 1, Length * frame_size]
    # energy: [BatchSize, 1, Length]
    # f0: [BatchSize, 1, Length]
    # reference: None or [BatchSize, content_channels, NumReferenceVectors]
    # alpha: float 0 ~ 1.0
    # k: int
    def infer(self, content, spk, energy=None, f0=None, reference=None, alpha=0, k=4, metrics='cos'):
        if energy is None or f0 is None:
            f0_est, energy_est = self.pitch_energy_estimator.infer(content, spk)
        if energy is None:
            energy = energy_est
        if f0 is None:
            f0 = f0_est
            
        print(f0)

        # run feature retrieval if got reference vectors
        if reference is not None:
            content = match_features(content, reference, k, alpha, metrics)

        dsp_out, source = self.source_net(content, f0, energy, spk)

        # filter network
        output = self.filter_net(content, f0, energy, spk, source)
        return output
    
    def estimate_pitch_energy(self, content, spk):
        f0, energy = self.pitch_energy_estimator.infer(content, spk)
        return f0, energy


class PitchEnergyDecoder(nn.Module):
    def __init__(self,
                 sample_rate=48000,
                 frame_size=960,
                 n_fft=3840,
                 n_window=1024,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 pe_internal_channels=256,
                 pe_num_layers=6):
        super().__init__()
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.n_window = n_window
        self.sample_rate = sample_rate
        self.pitch_energy_estimator = PitchEnergyEstimator(
                content_channels=content_channels,
                speaker_embedding_dim=speaker_embedding_dim,
                num_layers=pe_num_layers,
                internal_channels=pe_internal_channels
                )
        
    # training pass
    #
    # content: [BatchSize, content_channels, Length]
    # f0: [BatchSize, 1, Length]
    # spk: [BatchSize, speaker_embedding_dim, 1]
    #
    # Outputs:
    #   f0_logits [BatchSize, num_f0_classes, Length]
    #   estimated_energy: [BatchSize, 1, Length]
    #   dsp_out: [BatchSize, Length * frame_size]
    #   output: [BatchSize, Length * frame_size]
    def forward(self, content, f0, energy, spk):
        # estimate energy, f0
        f0_logits, estimated_energy = self.pitch_energy_estimator(content, spk)
        f0_label = self.pitch_energy_estimator.freq2id(f0).squeeze(1)
        loss_pe = pitch_estimation_loss(f0_logits, f0_label)
        # estimated_f0, estimated_energy = self.pitch_energy_estimator.infer(content, spk)
        estimated_f0 = self.pitch_energy_estimator.logit2freq(f0_logits)
        loss_pe_freq = F.l1_loss(torch.log2(estimated_f0 + 1.), torch.log2(f0 + 1.))
        loss_ee = (estimated_energy - energy).abs().mean()
        loss = loss_pe * 45.0 + loss_pe_freq + loss_ee * 45.0
        loss_dict = {
            "Pitch Logits Estimation": loss_pe.item(),
            "Pitch Estimation": loss_pe_freq.item(),
            "Energy Estimation": loss_ee.item()
        }
        
        return None, None, loss, loss_dict
    
    def estimate_pitch_energy(self, content, spk):
        f0, energy = self.pitch_energy_estimator.infer(content, spk)
        return f0, energy
