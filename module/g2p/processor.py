from typing import List, Union, Tuple
import torch
from .module import G2PModule

from .japanese import JapaneseG2PModule
from .english import EnglishG2PModule


def unique(l: List):
    return list(set(l))


class G2PProcessor:
    def __init__(self):
        self.g2p_modules = {}

        # If you want to add a language, add processing here
        # ---
        self.g2p_modules['ja'] = JapaneseG2PModule()
        self.g2p_modules['en'] = EnglishG2PModule()
        # ---

        self.phoneme_vocabs = ['<pad>']
        for mod in self.g2p_modules.values():
            self.phoneme_vocabs += mod.possible_phonemes()
        self.phoneme_vocabs = unique(self.phoneme_vocabs)
        self.languages = ['unknown']
        self.languages += self.g2p_modules.keys()

    def grapheme_to_phoneme(self, text: Union[str, List[str]], language: Union[str, List[str]]):
        if type(text) == list:
            return self._g2p_multiple(text, language)
        elif type(text) == str:
            return self._g2p_single(text, language)

    def _g2p_single(self, text, language):
        mod = self.g2p_modules[language]
        return mod.g2p(text)

    def _g2p_multiple(self, text, language):
        result = []
        for t, l in zip(text, language):
            result.append(self._g2p_single(t, l))
        return result

    def phoneme_to_id(self, phonemes: Union[List[str], List[List[str]]], max_length: int):
        if type(phonemes[0]) == list:
            return self._p2id_multiple(phonemes, max_length)
        elif type(phonemes[0]) == str:
            return self._p2id_single(phonemes, max_length)

    def _p2id_single(self, phonemes: List[str], max_length: int):
        length = len(phonemes)
        ids = []
        for p in phonemes:
            if p in self.phoneme_vocabs:
                ids.append(self.phoneme_vocabs.index(p))
            else:
                print("warning: unknown phoneme.")
                ids.append(0)
        while len(ids) < max_length:
            ids.append(0) # padding
        if len(ids) > max_length:
            ids = ids[:max_length]
        return ids, length

    def _p2id_multiple(self, phonemes: List[List[str]], max_length: int):
        sequences = []
        lengths = []
        for s in phonemes:
            out, length = self._p2id_single(s, max_length)
            sequences.append(out)
            lengths.append(length)
        return sequences, lengths

    def language_to_id(self, languages: Union[str, List[str]]):
        if type(languages) == str:
            return self._l2id_single(languages)
        elif type(languages) == list:
            return self._l2id_multiple(languages)

    def _l2id_single(self, language):
        if language in self.languages:
            return self.languages.index(language)
        else:
            return 0 # unknown

    def _l2id_multiple(self, languages):
        result = []
        for l in languages:
            result.append(self._l2id_single(l))
        return result

    def encode(self, sentences: List[str], languages: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phonemes = self.grapheme_to_phoneme(sentences, languages)
        ids, lengths = self.phoneme_to_id(phonemes, max_length)
        language_ids = self.language_to_id(languages)

        ids = torch.LongTensor(ids)
        lengths = torch.LongTensor(lengths)
        language_ids = torch.LongTensor(language_ids)

        return ids, lengths, language_ids

