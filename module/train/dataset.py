import torch
import torchaudio
import json
from pathlib import Path


class VITSDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir='dataset_cache', speaker_infomation='./models/speakers.json'):
        super().__init__()
        self.root = Path(cache_dir)
        self.speakers = json.load(open(Path(speaker_infomation)))
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
        metadata = torch.load(self.metadata_paths[idx])
        f0 = metadata['f0'].squeeze(0)
        phonemes = metadata['phonemes'].squeeze(0)
        phonemes_len = metadata['phonemes_len'].item()
        language = metadata['language'].item()
        lm_feat = metadata['lm_feat'].squeeze(0)
        lm_feat_len = metadata['lm_feat_len'].item()
        return wf, speaker_id, f0, phonemes, phonemes_len, lm_feat, lm_feat_len, language

    def __len__(self):
        return len(self.audio_file_paths)
