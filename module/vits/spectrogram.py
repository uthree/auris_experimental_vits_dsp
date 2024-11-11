import torch
import torch.nn as nn
import torch.nn.functional as F


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, n_fft//2 - 1, Frames]
def spectrogram(wave, n_fft, hop_size, power=2.0):
    dtype = wave.dtype
    wave = wave.to(torch.float)
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window).abs()
    spec = spec[:, :, 1:]
    spec = spec.to(dtype)
    return spec


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, n_fft//2 - 1, Frames]
# Convert to Cepstrum domain
def cepstrum(wave, n_fft, hop_size, power=None):
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window)
    cepstrum = torch.fft.ifft(torch.log(spec + 1e-6), dim=1)
    return torch.real(cepstrum)[:, :cepstrum.shape[1]//2+1, 1:]


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, n_fft//2 - 1, Frames]
# Convert to Complext Cepstrum domain
def complex_cepstrum(wave, n_fft, hop_size, power=None):
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window)
    cepstrum = torch.fft.ifft(torch.log(spec + 1e-6), dim=1)
    return cepstrum