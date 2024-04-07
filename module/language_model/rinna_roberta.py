import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .extractor import LinguisticExtractor


# feature extractor using rinna/japanese-roberta-base
# from: https://huggingface.co/rinna/japanese-roberta-base
class RinnaRoBERTaExtractor(LinguisticExtractor):
    def __init__(self, hf_repo: str, layer: int, device=None):
        super().__init__()
        self.layer = layer

        if device is None:
            if torch.cuda.is_available(): # CUDA available
                device = torch.device('cuda')
            elif torch.backends.mps.is_available(): # on macos
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device)
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, use_fast=False)
        self.tokenizer.do_lower_case = True
        self.model = AutoModelForMaskedLM.from_pretrained(hf_repo)
        self.model.to(device)

    def extract(self, text):
        # add [CLS] token
        text = "[CLS]" + text

        # tokenize
        tokens = self.tokenizer.tokenize(text)

        # convert to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # convert to tensor
        token_tensor = torch.LongTensor([token_ids]).to(self.device)

        # provide position id explicitly
        position_ids = list(range(0, token_tensor.shape[1]))
        position_id_tensor = torch.LongTensor([position_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                    input_ids=token_tensor,
                    position_ids=position_id_tensor,
                    output_hidden_states=True)
            features = outputs.hidden_states[self.layer]
        # return features and length
        return features, token_tensor.shape[1]
