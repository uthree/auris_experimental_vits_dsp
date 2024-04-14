# test feature retrieval
import torch
from module.vits.flow import Flow

flow = Flow()
z = torch.randn(2, 192, 100)
g = torch.randn(2, 256, 1)
z_mask = torch.ones(2, 1, 100)
z_p = flow(z, z_mask, g)
z_recon = flow(z_p, z_mask, g, reverse=True)
print((z_recon - z).abs())