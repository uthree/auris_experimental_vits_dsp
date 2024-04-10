import torch
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from module.vits import Vits
from module.g2p import G2PProcessor
from module.language_model import LanguageModel
from module.utils.config import load_json_file


class Infer:
    def __init__(self, config_path, metadata_path):
        self.config = load_json_file(config)
        self.metadata = load_json_file(metadata_path)
        self.g2p = G2PProcessor()
        self.lm = LanguageModel(config.language_model.type, config.language_model.options)

    def audio_reconstruction(self):
        pass

    def text_to_speech(self):
        pass

    def singing_voice_conversion(self):
        pass
