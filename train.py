import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from module.model import VITS
from module.train.dataset import VITSDataset
from module.train.loss import LogMelSpectrogramLoss
from module.utils.spectrogram import spectrogram


parser = argparse.ArgumentParser(description="train audio reconstruction task")
parser.add_argument('-c', '--config', default='config/base.json')
args = parser.parse_args()
config = json.load(open(args.config))
model_path = Path('models') / 'model.safetensors'

device = torch.device(config['train']['device'])

def load_or_init_models(device=torch.device('cpu')):
    model = VITS(config)
    if os.path.exists(model_path):
        tensors = {}
        with safe_open(model_path) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        model.load_state_dict(tensors)
    model = model.to(device)
    return model

def save_models(model):
    print("Saving... (do not terminate this processs)")
    model = model.to(torch.device('cpu'))
    tensors = model.state_dict()
    save_file(tensors, model_path)
    print("Save Complete.")


model = load_or_init_models(device)
batch_size = config['train']['batch_size']
num_epoch = config['train']['epoch']
lr = config['train']['lr']
interval = config['train']['interval']
ds = VITSDataset(config['train']['cache'])
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
OptG = optim.AdamW(model.generator.parameters(), lr)
OptD = optim.AdamW(model.discriminator.parameters(), lr)

mel_loss = LogMelSpectrogramLoss().to(device)

step_count = 0

for epoch in range(num_epoch):
    tqdm.write(f" Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wf, spk_id, f0, phonemes, phonemes_len, lm_feat, lm_feat_len, lang) in enumerate(dl):
        N = wf.shape[0]

        wf = wf.to(device)
        f0 = f0.to(device)
        spk_id = spk_id.to(device)
        phonemes = phonemes.to(device)
        phonemes_len = phonemes_len.to(device)
        lm_feat = lm_feat.to(device)
        lm_feat_len = lm_feat_len.to(device)
        lang = lang.to(device)

        bar.update(N)
    if batch % interval == 0:
        save_models(model)

print("Training Complete!")
