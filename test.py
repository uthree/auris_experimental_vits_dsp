import torch
import torchaudio

from module.infer import AurisInfer
from module.utils.spectrogram import spectrogram

auris = AurisInfer("models/model.safetensors", "config/small.json")
wf = auris.text_to_speech("こんにちは！、これはテストメッセージです。", "jvs003", "ja")
torchaudio.save("output.wav", wf, sample_rate=48000)

vits = auris.vits
wf, sr = torchaudio.load("./dataset_cache/jvs001/0.wav")
spec = spectrogram(wf, 3840, 960)
spec_len = torch.LongTensor([10000])
g = torch.randn(1, 256, 1)
z, mean, logvar, mask = vits.generator.posterior_encoder(spec, spec_len, g)
wf = vits.generator.decoder.infer(z)
torchaudio.save("output_2.wav", wf.squeeze(1), sample_rate=48000)
