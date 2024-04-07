from .module import G2PModule
from g2p_en import G2p


class EnglishG2PModule(G2PModule):
    def __init__(self):
        super().__init__()
        self.g2p_instance = G2p()

    def g2p(self, text):
        return self.g2p_instance(text)

    def possible_phonemes(self):
        return self.g2p_instance.phonemes
