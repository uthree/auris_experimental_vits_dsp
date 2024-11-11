import torch
import torch.nn as nn
import torch.nn.functional as F


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, n_fft//2-1, Frames]
# Compute autocorrelation of wave
def autocorrelation(wave, n_fft, hop_size, window_fn=torch.ones):
    window = window_fn(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, onesided=False, return_complex=True, window=window)
    corr = torch.fft.ifft(spec * torch.conj(spec), dim=1)
    return torch.real(corr)[:, :corr.shape[1]//2+1, 1:]
