import torch
from module.model_components.monotonic_align import maximum_path

neg_x_ent = torch.randn(1, 50, 10)
attn_mask = torch.ones(1, 50, 10)
path = maximum_path(neg_x_ent, attn_mask)
print(path.shape)
