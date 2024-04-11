from pathlib import Path
import argparse

import torch
import torchaudio
from torchaudio.functional import resample

from module.infer import Infer

parser = argparse.ArgumentParser(description="inference")
parser.add_argument('-c', '--config', default='config/base.json')
parser.add_argument('-t', '--task', choices=['tts', 'recon', 'svc', 'svs'], default='tts')
parser.add_argument('-s', '--speaker', default='jvs001')
parser.add_argument('-i', '--inputs', default='inputs')
parser.add_argument('-o', '--outputs', default='outputs')
parser.add_argument('-ckpt', '--checkpoint', default='models/vits.ckpt')
parser.add_argument('-meta', '--metadata', default='models/metadata.json')
args = parser.parse_args()

outputs_dir = Path(args.outputs)
# load model
infer = Infer(args.checkpoint, args.config, args.metadata)

# support audio formats
audio_formats = ['mp3', 'wav', 'ogg']

# make outputs directory if not exists
if not outputs_dir.exists():
    outputs_dir.mkdir()

if args.task == 'recon':
    print("task: Audio Reconstruction")
    # audio reconstruction task
    spk = args.speaker

    # get input path
    inputs_dir = Path(args.inputs)
    inputs = []
    for fmt in audio_formats:
        for path in inputs_dir.glob(f"*.{fmt}"):
            inputs.append(path)

    for path in inputs:
        print(f"Inferencing {path}")

        # load audio
        wf, sr = torchaudio.load(path)

        # resample
        if sr != infer.sample_rate:
            wf = resample(wf, sr)

        # infer
        spk = args.speaker
        wf = infer.audio_reconstruction(wf, spk).squeeze(1)

        # save
        save_path = outputs_dir / (path.stem + ".wav")
        torchaudio.save(save_path, wf, infer.sample_rate)
