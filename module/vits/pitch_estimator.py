import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXtLayer


class PitchEstimator(nn.Module):
    def __init__(
            self,
            content_channels=192,
            internal_channels=256,
            num_layers=2,
            num_classes=512,
            classes_per_octave=48,
            min_frequency=20.0
        ):
        super().__init__()
        self.num_classes = num_classes
        self.classes_per_octave = classes_per_octave
        self.min_frequency = min_frequency

        self.input_layer = nn.Conv1d(content_channels, internal_channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(internal_channels) for _ in range(num_layers)])
        self.output_layer = nn.Conv1d(internal_channels, num_classes, 1)

    # spec: [BatchSize, fft_bin, Length]
    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x

    # f: [<Any shape allowed>]
    def freq2id(self, f):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        nc = self.num_classes
        return torch.ceil(torch.clamp(cpo * torch.log2(f / fmin), 0, nc-1)).to(torch.long)
    
    # ids: [<Any shape allowed>]
    def id2freq(self, ids):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        x = ids.to(torch.float)
        x = fmin * (2 ** (x / cpo))
        x[x <= self.min_frequency] = 0
        return x
    
    # z_p: [BatchSize, content_channels, Length]
    # spk: [BatchSize, speaker_embedding_dim, Length]
    # Outputs:
    #   f0: [BatchSize, 1, Length]
    #   energy: [BatchSize, 1, Length]
    def decode(self, logits, k=4):
        probs, indices = torch.topk(logits, k, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.id2freq(indices)
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.min_frequency] = 0
        return f0
    
    def infer(self, spec):
        logits = self.forward(spec)
        f0 = self.decode(logits)
        return f0