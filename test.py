import torch
import torchaudio

from module.infer import AurisInfer

auris = AurisInfer("models/model.safetensors", "config/small.json")
wf = auris.text_to_speech("こんにちは！", "jvs001", "ja")
print(wf.shape)
torchaudio.save("output.wav", wf, sample_rate=48000)
