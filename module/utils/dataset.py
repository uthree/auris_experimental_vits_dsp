import torch
import torchaudio
from pathlib import Path


class VITSDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir='dataset_cache'):
        super().__init__()
