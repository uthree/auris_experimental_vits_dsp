import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, channels=32, channels_mul=2, num_layers=4, max_channels=256, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.parametrizations.weight_norm if use_spectral_norm == False else nn.utils.parametrizations.spectral_norm
        
        k = kernel_size
        s = stride
        c = channels

        convs = [nn.Conv2d(1, c, (k, 1), (s, 1), (get_padding(5, 1), 0))]
        for i in range(num_layers):
            c_next = min(c * channels_mul, max_channels)
            convs.append(nn.Conv2d(c, c_next, (k, 1), (s, 1), (get_padding(5, 1), 0)))
            c = c_next
        self.convs = nn.ModuleList([norm_f(c) for c in convs])
        self.post = norm_f(nn.Conv2d(c, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad))
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(
            self,
            periods=[1, 2, 5, 7, 11],
            channels=32,
            channels_mul=2,
            max_channels=256,
            num_layers=4
            ):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for i, p in enumerate(periods):
            self.sub_discs.append(DiscriminatorP(p,
                                                 channels=channels,
                                                 channels_mul=channels_mul,
                                                 max_channels=max_channels,
                                                 num_layers=num_layers,
                                                 use_spectral_norm=(i==0)))

    def forward(self, x):
        x = x.unsqueeze(1)
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class DiscriminatorR(nn.Module):
    def __init__(self, resolution=128, channels=32, num_layers=6, use_spectral_norm=True):
        super().__init__()
        norm_f = nn.utils.parametrizations.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm
        self.convs = nn.ModuleList([norm_f(nn.Conv2d(1, channels, (7, 3), (1, 1), (3, 1)))])
        self.hop_size = resolution
        self.n_fft = resolution * 4
        for _ in range(num_layers):
            self.convs.append(norm_f(nn.Conv2d(channels, channels, (7, 3), (2, 1), (2, 1))))
        self.post = norm_f(nn.Conv2d(channels, 1, 3, 1, 1))

    def spectrogram(self, x):
        # spectrogram
        w = torch.hann_window(self.n_fft).to(x.device)
        x = torch.stft(x, self.n_fft, self.hop_size, window=w, return_complex=True).abs()
        return x

    def forward(self, x):
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        feats = []
        for l in self.convs:
            x = l(x)
            F.leaky_relu(x, 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x, feats


class MultiResolutionDiscriminator(nn.Module):
    def __init__(
            self,
            resolutions=[128, 256, 512],
            channels=32,
            num_layers=6,
            ):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for i, r in enumerate(resolutions):
            self.sub_discs.append(DiscriminatorR(r, channels, num_layers, (i==0)))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator(**config.mpd)
        self.MRD = MultiResolutionDiscriminator(**config.mrd)

    # x: [BatchSize, Length(waveform)]
    def forward(self, x):
        mpd_logits, mpd_feats = self.MPD(x)
        mrd_logits, mrd_feats = self.MRD(x)
        return mpd_logits + mrd_logits, mpd_feats + mrd_feats
