import os
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from module.vits import VITS
from module.train.crop import crop_waveform
from module.train.dataset import VITSDataset
from module.train.loss import LogMelSpectrogramLoss, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss
from module.utils.spectrogram import spectrogram

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="train audio reconstruction task")
parser.add_argument('-c', '--config', default='config/base.json')
args = parser.parse_args()
config = json.load(open(args.config))

# load config
batch_size = config['train']['batch_size']
num_epoch = config['train']['epoch']
lr = config['train']['lr']
save_interval = config['train']['save_interval']
tensorboard_interval = config['train']['tensorboard_interval']
frame_size = config['generator']['decoder']['frame_size']
n_fft = config['generator']['decoder']['n_fft']
sample_rate = config['generator']['decoder']['sample_rate']
use_amp = config['train']['amp']

device = torch.device(config['train']['device'])

model_path = Path('models') / 'model.safetensors'

def load_or_init_models(device=torch.device('cpu')):
    model = VITS(config)
    if os.path.exists(model_path):
        tensors = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(device)
        model.load_state_dict(tensors)
        print(f"loaded model from {str(model_path)}")
    model = model.to(device)
    return model


def save_models(model):
    print("Saving... (do not terminate this processs)")
    tensors = model.state_dict()
    save_file(tensors, model_path)
    print("Save Complete.")



ds = VITSDataset(config['train']['cache'])
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

model = load_or_init_models(device)

OptG = optim.AdamW(model.generator.parameters(), lr)
OptD = optim.AdamW(model.discriminator.parameters(), lr)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

MelLoss = LogMelSpectrogramLoss().to(device)

step_count = 0

G = model.generator
D = model.discriminator

writer = SummaryWriter(log_dir="./logs")

print("Start training")
print("run `tensorboard --logdir logs` if you need show tensorboard")
for epoch in range(num_epoch):
    tqdm.write(f" Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wf, spec_len, spk_id, f0, phoneme, phoneme_len, lm_feat, lm_feat_len, lang) in enumerate(dl):
        N = wf.shape[0]

        wf = wf.to(device)
        spec_len = spec_len.to(device)
        f0 = f0.to(device)
        spk_id = spk_id.to(device)
        phoneme = phoneme.to(device)
        phoneme_len = phoneme_len.to(device)
        lm_feat = lm_feat.to(device)
        lm_feat_len = lm_feat_len.to(device)
        lang = lang.to(device)

        # Train generator
        OptG.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            wf = wf.squeeze(1) #[N, WaveformLength]

            spec = spectrogram(wf, n_fft, frame_size) # [N, fft_bin, Length]

            dsp_out, fake, lossG, crop_range, loss_dict = G(spec, spec_len, phoneme, phoneme_len, lm_feat, lm_feat_len, f0, spk_id, lang)

            real = crop_waveform(wf, crop_range, frame_size)
            loss_dsp = MelLoss(dsp_out, real) * 45.0
            loss_mel = MelLoss(fake, real) * 45.0

            logits_real, fmap_real = D(real)
            logits_fake, fmap_fake = D(fake)

            loss_adv = generator_adversarial_loss(logits_fake)
            loss_feat = feature_matching_loss(fmap_real, fmap_fake)

            lossG += loss_dsp + loss_mel + loss_adv + loss_feat

            loss_dict["DSP"] = loss_dsp.item()
            loss_dict["Mel"] = loss_mel.item()
            loss_dict["Feature Matching"] = loss_feat.item()
            loss_dict["Generator Adversarial"] = loss_adv.item()

        # backward
        scaler.scale(lossG).backward()
        nn.utils.clip_grad_norm_(G.parameters(), 1.0, 2.0)
        scaler.step(OptG)

        # Train Discriminator
        OptD.zero_grad()
        fake = fake.detach()

        with torch.cuda.amp.autocast(enabled=use_amp):
            # calculate adversarial loss
            logits_real, _ = D(real)
            logits_fake, _ = D(fake)
            lossD = discriminator_adversarial_loss(logits_real, logits_fake)
            loss_dict["Discriminator Adversarial"] = lossD.item()

        # backward
        scaler.scale(lossD).backward()
        nn.utils.clip_grad_norm_(D.parameters(), 1.0, 2.0)
        scaler.step(OptD)

        scaler.update()

        if batch % tensorboard_interval == 0:
            # write summary
            for k, v in zip(loss_dict.keys(), loss_dict.values()):
                writer.add_scalar(f"scalar/{k}", v, step_count)
            dsp_preview = torch.clamp(dsp_out[0].unsqueeze(0), -1.0, 1.0)
            fake_preview = torch.clamp(fake[0].unsqueeze(0), -1.0, 1.0)
            writer.add_audio("audio/DSP Output", dsp_preview, step_count, sample_rate)
            writer.add_audio("audio/GAN Output", fake_preview, step_count, sample_rate)

        tqdm.write(f"G: {lossG.item():.4f}, D: {lossD.item():.4f}")
        bar.set_description(f"Epoch: {epoch}, Step: {step_count}")
        bar.update(N)
        step_count += 1
        if batch % save_interval == 0:
            save_models(model)

print("Training Complete!")
