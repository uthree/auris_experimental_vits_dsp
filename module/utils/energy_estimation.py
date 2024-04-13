import torch
import torch.nn.functional as F


# estimate energy from power spectrogram
# input:
#   spec: [BatchSize, n_fft//2+1, Length]
# output:
#   energy: [BatchSize, 1, Length]
def estimate_energy(spec):
    return spec.mean(dim=1, keepdim=True)

