import torch
import torch.nn as nn


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=8192, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embedding_dim)
    
    # ids: [BatchSize]
    # Output: [BatchSize, embedding_dim, 1]
    def forward(self, ids):
        x = self.embedding(x)
        x.unsqueeze(2)
        return x
