from typing import List


# grapheme-to-phoneme module base class
# Create a module by inheriting this class for each language.
class PhoneticExtractor():
    def __init__(self, *args, **kwargs):
        pass

    # grapheme to phoneme
    def g2p(self, text: str) -> List[str]:
        raise "Not Implemented"

    def possible_phonemes(self) -> List[str]:
        raise "Not Implemented"
