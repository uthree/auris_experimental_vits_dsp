from .module import G2PModule
import pyopenjtalk


class JapaneseG2PModule(G2PModule):
    def __init__(self):
        super().__init__()

    def g2p(self, text):
        return pyopenjtalk.g2p(text).split(" ")

    def possible_phonemes(self):
        return [' ', 'pau', 'I', 'N', 'U', 'a', 'b', 'by', 'ch', 'cl', 'd',
                'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky',
                'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh',
                't', 'ts', 'ty', 'u', 'v', 'w', 'y', 'z']
