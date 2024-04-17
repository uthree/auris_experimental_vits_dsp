from typing import List, Union

import torch
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from module.vits import Vits, spectrogram
from module.g2p import G2PProcessor
from module.language_model import LanguageModel
from module.utils.config import load_json_file


class Infer:
    def __init__(self, checkpoint_path, config_path, metadata_path, device=torch.device('cpu')):
        self.device = device
        self.config = load_json_file(config_path)
        self.metadata = load_json_file(metadata_path)
        self.g2p = G2PProcessor()
        self.lm = LanguageModel(self.config.language_model.type, self.config.language_model.options)
        vits = Vits.load_from_checkpoint(checkpoint_path)
        self.generator = vits.generator.to(device)

        self.max_lm_tokens = self.config.infer.max_lm_tokens
        self.max_phonemes = self.config.infer.max_phonemes
        self.max_frames = self.config.infer.max_frames

        self.n_fft = self.config.infer.n_fft
        self.frame_size = self.config.infer.frame_size
        self.sample_rate = self.config.infer.sample_rate

    def speakers(self):
        return self.metadata.speakers

    def speaker_id(self, speaker):
        return self.speakers().index(speaker)

    def languages(self):
        return self.g2p.languages

    def language_id(self, language):
        return self.g2p.language_to_id(language)

    @torch.inference_mode()
    def text_to_speech(
            self,
            text: str,
            speaker: str,
            language: str,
            style_text: Union[None, str] = None
            ):
        spk = torch.LongTensor([self.speaker_id(speaker)])
        lm_feat, lm_feat_len = self.lm.encode([text], self.max_lm_tokens)
        if style_text is None:
            style_text = text
        phoneme, phoneme_len, lang = self.g2p.encode([style_text], [language], self.max_phonemes)

        device = self.device
        phoneme = phoneme.to(device)
        phoneme_len = phoneme_len.to(device)
        lm_feat = lm_feat.to(device)
        lm_feat_len = lm_feat_len.to(device)
        spk = spk.to(device)
        lang = lang.to(device)

        wf = self.generator.text_to_speech(
                phoneme,
                phoneme_len,
                lm_feat,
                lm_feat_len,
                lang,
                spk,
                )
        return wf.squeeze(0)

    # wf: [Channels, Length]
    def audio_reconstruction(self, wf: torch.Tensor, speaker:str):
        spk = torch.LongTensor([self.speaker_id(speaker)])
        wf = wf.sum(dim=0, keepdim=True)
        spec = spectrogram(wf, self.n_fft, self.frame_size)
        spec_len = torch.LongTensor([spec.shape[2]])

        device = self.device
        spec = spec.to(device)
        spec_len = spec_len.to(device)
        spk = spk.to(device)

        wf = self.generator.audio_reconstruction(spec, spec_len, spk)
        return wf.squeeze(0)

    def singing_voice_conversion(self):
        pass
