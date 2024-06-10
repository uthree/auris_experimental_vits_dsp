import torch
from typing import List, Tuple

# base class of linguistic feature extractor
class LinguisticExtractor:
    def __init__(self, *args, **kwargs):
        pass

    # Output: [1, lm_dim]
    def extract(self, str) -> torch.Tensor:
        pass
