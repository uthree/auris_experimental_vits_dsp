# test feature retrieval
import torch
from module.vits.feature_retrieval import match_features

source = torch.randn(2, 192, 100)
reference = torch.randn(2, 192, 100)

output = match_features(source, reference)
print(output.shape)
