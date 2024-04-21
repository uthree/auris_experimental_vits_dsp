from pathlib import Path
import argparse
import torch
import numpy as np

from module.infer import Infer

import gradio as gr

parser = argparse.ArgumentParser(description="inference")
parser.add_argument('-c', '--config', default='models/config.json')
parser.add_argument('-t', '--task', choices=['tts', 'recon', 'svc', 'svs'], default='tts')
parser.add_argument('-m', '--model', default='models/generator.safetensors')
parser.add_argument('-meta', '--metadata', default='models/metadata.json')
parser.add_argument('-p', '--port', default=7860, type=int)
args = parser.parse_args()

# load model
infer = Infer(args.model, args.config, args.metadata)
device = infer.device

def text_to_speech(text, style_text, speaker, language, duration_scale, pitch_shift):
    if style_text == "":
        style_text = text
    wf = infer.text_to_speech(text, speaker, language, style_text, duration_scale, pitch_shift)
    wf = wf.squeeze(0).cpu().numpy()
    wf = (wf * 32768).astype(np.int16)
    sample_rate = infer.sample_rate
    return sample_rate, wf

demo = gr.Interface(text_to_speech, inputs=[
    gr.Text(label="Text"),
    gr.Text(label="Style"),
    gr.Dropdown(infer.speakers(), label="Speaker", value=infer.speakers()[0]),
    gr.Dropdown(infer.languages(), label="Language", value=infer.languages()[0]),
    gr.Slider(0.1, 3.0, 1.0, label="Duration Scale"),
    gr.Slider(-12.0, 12.0, 0.0, label="Pitch Shift"),
    ],
    outputs=[gr.Audio()])

demo.launch(debug=True, server_port=args.port)