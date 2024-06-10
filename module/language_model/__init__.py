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

    def encode(self, sentences):
        if type(sentences) == list:
            return self._ext_multiple(sentences)
        elif type(sentences) == str:
            return self._ext_single(sentences)

    def _ext_single(self, sentence):
        features  = self.extractor.extract(sentence)
        return features

    def _ext_multiple(self, sentences):
        features = []
        for s in sentences:
            f = self._ext_single(s)
            features.append(f)
        features = torch.cat(features, dim=0)
        return features
