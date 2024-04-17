from pathlib import Path
import argparse
import json

import torch
import torchaudio
from torchaudio.functional import resample

from module.infer import Infer

import gradio as gr

parser = argparse.ArgumentParser(description="inference")
parser.add_argument('-c', '--config', default='config/base.json')
parser.add_argument('-t', '--task', choices=['tts', 'recon', 'svc', 'svs'], default='tts')
parser.add_argument('-ckpt', '--checkpoint', default='models/vits.ckpt')
parser.add_argument('-meta', '--metadata', default='models/metadata.json')
parser.add_argument('-o', '--outputs', default='outputs')
args = parser.parse_args()

outputs_dir = Path(args.outputs)

# make outputs directory if not exists
if not outputs_dir.exists():
    outputs_dir.mkdir()

# load model
infer = Infer(args.checkpoint, args.config, args.metadata)
device = infer.device

demo = gr.Blocks()

def text_to_speech(text, style_text, speaker, language):
    if style_text == "":
        style_text = text
    wf = infer.text_to_speech(text, speaker, language, style_text)
    output_file_name = f"{speaker}_{text}.wav"
    save_path = outputs_dir / output_file_name
    torchaudio.save(save_path, wf, sample_rate=infer.sample_rate)
    return save_path

tts_demo = gr.Interface(fn=text_to_speech, inputs=[
    gr.Text(label="Text"),
    gr.Text(label="Style"),
    gr.Dropdown(infer.speakers(), label="Speaker", value=infer.speakers()[0]),
    gr.Dropdown(infer.languages(), label="Language", value=infer.languages()[0])],
    outputs=[gr.Audio()])

with demo:
    gr.TabbedInterface([tts_demo], ["Text-to-Speech"])

demo.launch(debug=True)