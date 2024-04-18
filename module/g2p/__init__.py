from typing import List, Union, Tuple
import torch

from .japanese import JapaneseExtractor
#from .english import EnglishExtractor


class G2PProcessor:
    def __init__(self):
        self.extractors = {}

        # If you want to add a language, add processing here
        # ---
        self.extractors['ja'] = JapaneseExtractor()
        #self.extractors['en'] = EnglishExtractor()
        # ---

        self.languages = []
        phoneme_vocabs = []
        for mod in self.extractors.values():
            phoneme_vocabs += mod.possible_phonemes()
        self.languages += self.extractors.keys()
        self.phoneme_vocabs = ['<pad>'] + phoneme_vocabs

    def grapheme_to_phoneme(self, text: Union[str, List[str]], language: Union[str, List[str]]):
        if type(text) == list:
            return self._g2p_multiple(text, language)
        elif type(text) == str:
            return self._g2p_single(text, language)

    def _g2p_single(self, text, language):
        mod = self.extractors[language]
        return ['<pad>'] + mod.g2p(text)

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
        length = min(length, max_length)
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
            return 0

    def _l2id_multiple(self, languages):
        result = []
        for l in languages:
            result.append(self._l2id_single(l))
        return result

    def id_to_phoneme(self, ids):
        if type(ids[0]) == list:
            return self._id2p_multiple(ids)
        elif type(ids[0]) == int:
            return self._id2p_single(ids)

    def _id2p_single(self, ids: List[int]) -> List[str]:
        phonemes = []
        for i in ids:
            if i < len(self.phoneme_vocabs):
                p = self.phoneme_vocabs[i]
            else:
                p = '<pad>'
            phonemes.append(p)
        return phonemes

    def _id2p_multiple(self, ids: List[List[int]]) -> List[List[str]]:
        results = []
        for s in ids:
            results.append(self._id2p_single(s))
        return results

    def encode(self, sentences: List[str], languages: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phonemes = self.grapheme_to_phoneme(sentences, languages)
        ids, lengths = self.phoneme_to_id(phonemes, max_length)
        language_ids = self.language_to_id(languages)

        ids = torch.LongTensor(ids)
        lengths = torch.LongTensor(lengths)
        language_ids = torch.LongTensor(language_ids)

        return ids, lengths, language_ids