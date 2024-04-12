import torch
import torch.nn.functional as F

# estimate energy from waveform
# input:
#   wf: [BatchSize, 1, Length]
# output:
#   energy: [BatchSize, 1,Length]
def estimate_energy_from_waveform(wf, frame_size=960):
    return F.max_pool1d(wf ** 2, kernel_size=frame_size, stride=frame_size)

# estimate energy from power spectrogram
# input:
#   spec: [BatchSize, n_fft//2+1, Length]
# output:
#   energy: [BatchSize, 1, Length]
def estimate_energy_from_spectrogram(spec, n_fft=3840):
    return spec.max(dim=1, keepdim=True).values / n_fft

