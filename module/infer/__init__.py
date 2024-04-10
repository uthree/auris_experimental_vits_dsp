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
    def __init__(self, config, speaker_list)
