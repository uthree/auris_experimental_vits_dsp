import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            scales=[16, 32, 64, 128, 256, 512]
            ):
        super().__init__()
        self.scales = scales

    def forward(self, x, y):
        x = x.to(torch.float)
        y = y.to(torch.float)

        loss = 0
        num_scales = len(self.scales)
        for s in self.scales:
            hop_length = s
            n_fft = s * 4
            window = torch.hann_window(n_fft, device=x.device)
            x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
            y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()

            x_spec[x_spec.isnan()] = 0
            x_spec[x_spec.isinf()] = 0
            y_spec[y_spec.isnan()] = 0
            y_spec[y_spec.isinf()] = 0

            loss += ((x_spec - y_spec) ** 2).mean() + (safe_log(x_spec) - safe_log(y_spec)).abs().mean()
        return loss / num_scales


class LogMelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            sample_rate=48000,
            n_fft=2048,
            hop_length=512,
            n_mels=80
            ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate,
                n_fft,
                hop_length=hop_length,
                n_mels=n_mels)
    
    def forward(self, x, y):
        x = x.to(torch.float)
        y = y.to(torch.float)

        x = safe_log(self.to_mel(x))
        y = safe_log(self.to_mel(y))

        x[x.isnan()] = 0
        x[x.isinf()] = 0
        y[y.isnan()] = 0
        y[y.isinf()] = 0

        return (x - y).abs().mean()


# 1 = fake, 0 = real
def discriminator_adversarial_loss(real_outputs, fake_outputs):
    loss = 0
    n = min(len(real_outputs), len(fake_outputs))
    for dr, df in zip(real_outputs, fake_outputs):
        dr = dr.float()
        df = df.float()
        real_loss = (dr ** 2).mean()
        fake_loss = ((df - 1) ** 2).mean()
        loss += real_loss + fake_loss
    return loss / n


def generator_adversarial_loss(fake_outputs):
    loss = 0
    n = len(fake_outputs)
    for dg in fake_outputs:
        dg = dg.float()
        loss += (dg ** 2).mean()
    return loss / n


def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    n = min(len(fmap_real), len(fmap_fake))
    for r, f in zip(fmap_real, fmap_fake):
        f = f.float()
        r = r.float()
        loss += (f - r).abs().mean()
    return loss * (2 / n) 
