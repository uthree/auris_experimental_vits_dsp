from typing import Dict
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def save_tensors(tensors: Dict[str, torch.Tensor], path):
    save_file(tensors, path)

def load_tensors(path) -> Dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors