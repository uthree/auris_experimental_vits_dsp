import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import LayerNorm
from .convnext import ConvNeXtLayer

# this model estimates fundamental frequency (f0) and energy from z_prior
class ProsodyPredictor(nn.Module):
    def __init__(self,
                 content_channels=192,
                 speaker_embedding_dim=256,
                 internal_channels=256,
                 num_classes=512,
                 kernel_size=7,
                 num_layers=6,
                 mlp_mul=3,
                 min_frequency=20.0,
                 classes_per_octave=48,
                 ):
        super().__init__()
        self.content_channels = content_channels
        self.num_classes = num_classes
        self.classes_per_octave = classes_per_octave
        self.min_frequency = min_frequency

        self.content_input = nn.Conv1d(content_channels, internal_channels, 1)
        self.speaker_input = nn.Conv1d(speaker_embedding_dim, internal_channels, 1)
        self.input_norm = LayerNorm(internal_channels)
        self.mid_layers = nn.Sequential(
                *[ConvNeXtLayer(internal_channels, kernel_size, mlp_mul) for _ in range(num_layers)])
        self.output_norm = LayerNorm(internal_channels)
        self.to_f0_logits = nn.Conv1d(internal_channels, num_classes, 1)
        self.to_energy = nn.Conv1d(internal_channels, 1, 1)

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
        cpo = self.classes_per_octave
        nc = self.num_classes
        return torch.ceil(torch.clamp(cpo * torch.log2(f / fmin), 0, nc-1)).to(torch.long)
    
    # ids: [<Any shape allowed>]
    def id2freq(self, ids):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        nc = self.num_classes
        x = ids.to(torch.float)
        x = fmin * (2 ** (x / cpo))
        x[x <= self.f0_min] = 0
        return x
    
    # z_p: [BatchSize, content_channels, Length]
    # spk: [BatchSize, speaker_embedding_dim, Length]
    # Outputs:
    #   f0: [BatchSize, 1, Length]
    #   energy: [BatchSize, 1, Length]
    def infer(self, z_p, spk, k=4):
        logits, energy = self.forward(z_p, spk)
        probs, indices = torch.topk(logits, k, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.freq2id(indices)
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.min_frequency] = 0
        return f0, energy