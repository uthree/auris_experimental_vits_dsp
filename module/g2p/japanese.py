from .extractor import PhoneticExtractor
import pyopenjtalk


class JapaneseExtractor(PhoneticExtractor):
    def __init__(self):
        super().__init__()

    def g2p(self, text):
        return ['pau'] + pyopenjtalk.g2p(text).split(" ")

    def possible_phonemes(self):
        return [' ', 'pau', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd',
                'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
                'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh',
                't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']
