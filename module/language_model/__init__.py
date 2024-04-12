import torch
from .rinna_roberta import RinnaRoBERTaExtractor


def get_extractor(typename):
    if typename == "rinna_roberta":
        return RinnaRoBERTaExtractor
    else:
        raise "Unknown linguistic extractor type"


class LanguageModel:
    def __init__(self, extractor_type, options):
        ext_constructor = get_extractor(extractor_type)
        self.extractor = ext_constructor(**options)

    def encode(self, sentences, max_length: int):
        if type(sentences) == list:
            return self._ext_multiple(sentences, max_length)
        elif type(sentences) == str:
            return self._ext_single(sentences, max_length)

    def _ext_single(self, sentence, max_length: int):
        features, length = self.extractor.extract(sentence)
        features = features.cpu()

        N, L, D = features.shape
        # add padding
        if L < max_length:
            pad = torch.zeros(N, max_length - L, D)
            features = torch.cat([pad, features], dim=1)
        # crop
        if L > max_length:
            features = features[:, :max_length, :]
        # length
        length = min(length, max_length)

        return features, length

    def _ext_multiple(self, sentences, max_length: int):
        lengths = []
        features = []
        for s in sentences:
            f, l = self._ext_single(s, max_length)
            features.append(f)
            lengths.append(l)
        features = torch.cat(features, dim=0)
        lengths = torch.LongTensor(lengths)
        return features, lengths
