import torch
from module.model_components.posterior_encoder import PosteriorEncoder
from module.model_components.decoder import Decoder
from module.model_components.flow import Flow
from module.model_components.text_encoder import TextEncoder
from module.model_components.audio_encoder import AudioEncoder
from module.model_components.duration_predictors import DurationPredictor, StochasticDurationPredictor


# test posterior encoder 
pe = PosteriorEncoder()
fft_bin = 3840 // 2 + 1
x = torch.randn(2, fft_bin, 100)
x_len = torch.LongTensor([100, 100])
spk = torch.randn(2, 256, 1)
z, mean, logvar, z_mask, = pe(x, x_len, spk)

# test decoder
dec = Decoder()
z = torch.randn(2, 192, 100)
f0 = torch.randn(2, 1, 100)
out = dec.infer(z)

# test flow
flow = Flow()
z_mask = torch.ones(2, 1, 100)
spk = torch.randn(2, 256, 1)
z_theta = flow.forward(z, z_mask, spk)

# test text encoder
te = TextEncoder()
text = torch.zeros(2, 10, dtype=torch.long)
text_length = torch.LongTensor([10, 10])
lm_features = torch.randn(2, 12, 768)
lm_features_length = torch.LongTensor([12, 12])
lang = torch.LongTensor([0, 0])
z, mean, logvar, z_mask = te(text, text_length, lm_features, lm_features_length, lang)

# audio encoder 
ae = AudioEncoder()
fft_bin = 3840 // 2 + 1
x = torch.randn(2, fft_bin, 100)
x_len = torch.LongTensor([100, 100])
z, mean, logvar, z_mask = ae(x, x_len)

# duration predictor
dp = DurationPredictor()
x = torch.randn(2, 192, 10)
x_mask = torch.ones(2, 1, 10)
dur = dp(x, x_mask)

# stochastic duration precictor
sdp = StochasticDurationPredictor()
x = torch.randn(2, 192, 10)
x_mask = torch.ones(2, 1, 10)
g = torch.randn(2, 10)

