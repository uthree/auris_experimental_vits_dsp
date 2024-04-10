from .extractor import PhoneticExtractor
from g2p_en import G2p


class EnglishExtractor(PhoneticExtractor):
    def __init__(self):
        super().__init__()
        self.g2p_instance = G2p()

    def g2p(self, text):
        return self.g2p_instance(text)

    def possible_phonemes(self):
        phonemes = self.g2p_instance.phonemes
        phonemes.remove('<pad>')
        return phonemes
