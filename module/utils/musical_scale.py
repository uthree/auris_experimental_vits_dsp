import torch


def freq2scale(f0: torch.Tensor):
    return 12 * torch.log2(f0 / 440)


def scale2freq(scales: torch.Tensor):
    return 440 * (2 ** (scale / 12))
