import torch
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from module.vits import Generator
from module.g2p import G2PProcessor
from module.language_model import LanguageModelProcessor
from module.utils.config import load_json_file


class AurisInfer():
    def __init__(self, generator_path, config_path, speaker_infomation_path='models/speakers.json', device=torch.device('cpu')):
        config = load_json_file(config_path)
        self.generator = Generator(config.generator)
        self.lm = LanguageModelProcessor(config.language_model.type, config.language_model.options)
        self.g2p = G2PProcessor()
        self.speakers = json.load(open(Path(speaker_infomation_path)))
        self.device = device

        # load model
        tensors = {}
        with safe_open(generator_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(device)
        self.generator.load_state_dict(tensors)

    # synthesize speech
    def text_to_speech(
            self,
            text,
            speaker,
            language,
            max_phonemes=100,
            max_lm_tokens=50
            ):
        device = self.device
        phoneme, phoneme_len, lang = self.g2p.encode([text], [language], max_phonemes)
        lm_feat, lm_feat_len = self.lm.encode([text], max_lm_tokens)
        spk = torch.LongTensor([self.speakers.index(speaker)])

        wf = self.generator.text_to_speech(
                phoneme.to(device),
                phoneme_len.to(device),
                lm_feat.to(device),
                lm_feat_len.to(device),
                spk.to(device),
                lang.to(device))
        return wf.squeeze(0)
