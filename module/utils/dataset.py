import os
import torch
import torchaudio
import json
from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader, random_split


class VitsDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir='dataset_cache', metadata='models/metadata.json'):
        super().__init__()
        self.root = Path(cache_dir)
        metadata = json.load(open(Path(metadata)))
        self.speakers = metadata['speakers']
        self.audio_file_paths = []
        self.speaker_ids = []
        self.metadata_paths = []
        for path in self.root.glob("*/*.wav"):
            self.audio_file_paths.append(path)
            spk = path.parent.name
            self.speaker_ids.append(self._speaker_id(spk))
            metadata_path = path.parent / (path.stem + ".pt")
            self.metadata_paths.append(metadata_path)

    def _speaker_id(self, speaker: str) -> int:
        return self.speakers.index(speaker)

    def __getitem__(self, idx):
        speaker_id = self.speaker_ids[idx]
        wf, sr = torchaudio.load(self.audio_file_paths[idx])
        wf = wf.mean(dim=0)
        metadata = torch.load(self.metadata_paths[idx])
        spec = metadata['spec'].squeeze(0)
        spec_len = metadata['spec_len'].item()
        f0 = metadata['f0'].squeeze(0)
        phoneme = metadata['phonemes'].squeeze(0)
        phoneme_len = metadata['phonemes_len'].item()
        language = metadata['language'].item()
        lm_feat = metadata['lm_feat'].squeeze(0)
        return wf, spec, spec_len, speaker_id, f0, phoneme, phoneme_len, lm_feat, language

    def __len__(self):
        return len(self.audio_file_paths)


class VitsDataModule(L.LightningDataModule):
    def __init__(
            self,
            cache_dir='dataset_cache',
            metadata='models/metadata.json',
            batch_size=1,
            num_workers=1,
            ):
        super().__init__()
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = VitsDataset(
                self.cache_dir,
                self.metadata)
        dataloader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=(os.name=='nt'))
        return dataloader
