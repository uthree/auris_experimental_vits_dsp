from .extractor import PhoneticExtractor
import pyopenjtalk
from typing import List


class JapaneseExtractor(PhoneticExtractor):
    def __init__(self):
        super().__init__()

    def g2p(self, text) -> List[str]:
        phonemes = pyopenjtalk.g2p(text).split(" ")
        new_phonemes = []
        for p in phonemes:
            if p == 'pau':
                new_phonemes.append('<pad>')
            else:
                new_phonemes.append(p)
        return new_phonemes

    def possible_phonemes(self):
        return ['I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd',
                'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
                'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh',
                't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']
