import torch
import torchaudio

from module.infer import AurisInfer

auris = AurisInfer("models/model.safetensors", "config/small.json")
wf = auris.text_to_speech("こんにちは！、これはテストメッセージです。", "jvs003", "ja")
torchaudio.save("output.wav", wf, sample_rate=48000)
