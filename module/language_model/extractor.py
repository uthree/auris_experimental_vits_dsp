import torch
from typing import List, Tuple

# base class of linguistic feature extractor
class LinguisticExtractor:
    def __init__(self, *args, **kwargs):
        pass

    # Output: [1, length, lm_dim]
    def extract(self, str) -> Tuple[torch.Tensor, int]:
        pass
