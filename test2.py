from module.language_model.processor import LanguageModelProcessor

p = LanguageModelProcessor("rinna_roberta", options={"hf_repo":"rinna/japanese-roberta-base", "layer":12})
y, y_length = p.extract_linguistic_features(["aa", "aiueo"], 50)
print(y.shape, y_length.shape)
