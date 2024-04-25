from pathlib import Path
import argparse
import json

import torch
import torchaudio
from torchaudio.functional import resample

from module.infer import Infer

parser = argparse.ArgumentParser(description="inference")
parser.add_argument('-c', '--config', default='models/config.json')
parser.add_argument('-t', '--task', choices=['tts', 'recon', 'svc', 'svs'], default='tts')
parser.add_argument('-s', '--speaker', default='jvs001')
parser.add_argument('-i', '--inputs', default='inputs')
parser.add_argument('-o', '--outputs', default='outputs')
parser.add_argument('-m', '--model', default='models/generator.safetensors')
parser.add_argument('-meta', '--metadata', default='models/metadata.json')
args = parser.parse_args()

outputs_dir = Path(args.outputs)

# load model
infer = Infer(args.model, args.config, args.metadata)
device = infer.device

# support audio formats
audio_formats = ['mp3', 'wav', 'ogg']

# make outputs directory if not exists
if not outputs_dir.exists():
    outputs_dir.mkdir()

if args.task == 'recon':
    print("Task: Audio Reconstruction")
    # audio reconstruction task
    spk = args.speaker

    # get input path
    inputs_dir = Path(args.inputs)
    inputs = []

    # load files
    for fmt in audio_formats:
        for path in inputs_dir.glob(f"*.{fmt}"):
            inputs.append(path)

    # inference
    for path in inputs:
        print(f"Inferencing {path}")

        # load audio
        wf, sr = torchaudio.load(path)

        # resample
        if sr != infer.sample_rate:
            wf = resample(wf, sr)

        # infer
        spk = args.speaker
        wf = infer.audio_reconstruction(wf, spk).cpu()

        # save
        save_path = outputs_dir / (path.stem + ".wav")
        torchaudio.save(save_path, wf, infer.sample_rate)

elif args.task == 'tts':
    print("Task: Text to Speech")

    # get input path
    inputs_dir = Path(args.inputs)

    # load files
    inputs = []
    for path in inputs_dir.glob("*.json"):
        inputs.append(path)

    # inference
    for path in inputs:
        print(f"Inferencing {path}")
        t = json.load(open(path, encoding='utf-8'))
        for k, v in zip(t.keys(), t.values()):
            print(f"  Inferencing {k}")
            wf = infer.text_to_speech(**v).cpu()

            # save
            save_path = outputs_dir / (f"{path.stem}_{k}.wav")
            torchaudio.save(save_path, wf, infer.sample_rate)

elif args.task == 'svs':
    print("Task: Singing Voice Synthesis")

    inputs_dir = Path(args.inputs)

    # load score
    inputs = []
    for path in inputs_dir.glob("*.json"):
        inputs.append(path)

    # inference
    for path in inputs:
        print(f"Inferencing {path}")
        score = json.load(open(path, encoding='utf-8'))
        wf = infer.singing_voice_synthesis(score)

        # save
        save_path = outputs_dir / (path.stem + ".wav")
        torchaudio.save(save_path, wf, infer.sample_rate)

