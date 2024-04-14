import torch
import torch.nn.functional as F


# estimate energy from power spectrogram
# input:
#   spec: [BatchSize, n_fft//2+1, Length]
# output:
#   energy: [BatchSize, 1, Length]
def estimate_energy(spec):
    fft_bin = spec.shape[1]
    return spec.max(dim=1, keepdim=True).values / fft_bin
