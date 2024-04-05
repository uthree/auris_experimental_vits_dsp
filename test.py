import torch
from module.model_components.posterior_encoder import PosteriorEncoder

pe = PosteriorEncoder()
fft_bin = 3840 // 2 + 1
x = torch.randn(2, fft_bin, 100)
x_len = torch.Tensor([100, 100])
spk = torch.randn(2, 256, 1)

z, mean, logvar, z_mask, = pe(x, x_len, spk)
print(z.shape, mean.shape, logvar.shape, z_mask.shape)
