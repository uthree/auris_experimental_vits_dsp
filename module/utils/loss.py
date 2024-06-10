import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)

def multiscale_stft_loss(x, y, scales=[16, 32, 64, 128, 256, 512]):
    x = x.to(torch.float)
    y = y.to(torch.float)

    loss = 0
    num_scales = len(scales)
    for s in scales:
        hop_length = s
        n_fft = s * 4
        window = torch.hann_window(n_fft, device=x.device)
        x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
        y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()

        x_spec[x_spec.isnan()] = 0
        x_spec[x_spec.isinf()] = 0
        y_spec[y_spec.isnan()] = 0
        y_spec[y_spec.isinf()] = 0

        loss += (safe_log(x_spec) - safe_log(y_spec)).abs().mean()
    return loss / num_scales


global mel_spectrogram_modules
mel_spectrogram_modules = {}
def mel_spectrogram_loss(x, y, sample_rate=48000, n_fft=2048, hop_length=512, power=2.0, log=True):
    device = x.device
    if device not in mel_spectrogram_modules:
        mel_spectrogram_modules[device] = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, power=power).to(device)
    mel_spectrogram = mel_spectrogram_modules[device]

    x_mel = mel_spectrogram(x)
    y_mel = mel_spectrogram(y)
    if log:
        x_mel = safe_log(x_mel)
        y_mel = safe_log(y_mel)

        x_mel[x_mel.isnan()] = 0
        x_mel[x_mel.isinf()] = 0
        y_mel[y_mel.isnan()] = 0
        y_mel[y_mel.isinf()] = 0

    loss = F.l1_loss(x_mel, y_mel)
    return loss

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


def duration_discriminator_adversarial_loss(real_output, fake_output, text_mask):
    loss = (((fake_output - 1) ** 2) * text_mask).sum() / text_mask.sum()
    loss += ((real_output ** 2) * text_mask).sum() / text_mask.sum()
    return loss


def duration_generator_adversarial_loss(fake_output, text_mask):
    loss = ((fake_output ** 2) * text_mask).sum() / text_mask.sum()
    return loss

    
def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    n = min(len(fmap_real), len(fmap_fake))
    for r, f in zip(fmap_real, fmap_fake):
        f = f.float()
        r = r.float()
        loss += (f - r).abs().mean()
    return loss * (2 / n) 


def kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l