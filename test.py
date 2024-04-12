import torch
from module.vits.decoder import Decoder

d = Decoder(
        filter_down_interpolation='linear',
        filter_up_interpolation='linear',
        filter_resblock_type='3',
        filter_down_dilations=[[1, 2, 4]],
        filter_up_dilations=[[1, 3, 9, 27]],
        filter_up_kernel_sizes=[3],
        filter_down_kernel_size=3,
        )
z = torch.randn(2, 192, 100)
f0 = torch.randn(2, 1, 100)
spk = torch.randn(2, 256, 1)
dsp_out, fake, f0_logits = d.forward(z, f0, spk)
