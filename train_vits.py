import os
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from module.vits import Generator, Discriminator
from module.train.crop import crop_waveform
from module.train.dataset import Dataset
from module.train.loss import LogMelSpectrogramLoss, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss
from module.train.crop import crop_waveform, decide_crop_range
from module.utils.spectrogram import spectrogram
from module.utils.config import load_json_file

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="train audio reconstruction task")
parser.add_argument('-c', '--config', default='config/base.json')
args = parser.parse_args()
config = load_json_file(args.config)

# load config
batch_size = config.train.batch_size
num_epoch = config.train.num_epoch
opt_lr = config.train.optimizer.lr
opt_betas = config.train.optimizer.betas
save_interval = config.train.save_interval
tensorboard_interval = config.train.tensorboard_interval
frame_size = config.train.frame_size
n_fft = config.train.n_fft
sample_rate = config.train.sample_rate
crop_frames = config.train.crop_frames
use_amp = config.train.use_amp
device = torch.device(config.train.device)

generator_path = Path('models') / 'generator.safetensors'
discriminator_path = Path('models') / 'discriminator.safetensors'

def load_tensors(model_path):
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(device)
    return tensors


def load_or_init_models(device=torch.device('cpu')):
    gen = Generator(config.generator)
    dis = Discriminator(config.discriminator)
    if os.path.exists(generator_path):
        print("loading generator...")
        gen.load_state_dict(load_tensors(generator_path))
    if os.path.exists(discriminator_path):
        print("loading discriminator...")
        dis.load_state_dict(load_tensors(discriminator_path))
    gen = gen.to(device)
    dis = dis.to(device)
    return gen, dis


def save_models(gen, dis):
    print("Saving... (do not terminate this processs)")
    save_file(gen.state_dict(), generator_path)
    save_file(dis.state_dict(), discriminator_path)
    print("Save Complete.")


# Setup loss
MelLoss = LogMelSpectrogramLoss(sample_rate).to(device)

ds = VITSDataset(config.train.cache)
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

# Setup models
G, D = load_or_init_models(device)

# Setup optimizer, scaler
OptG = optim.AdamW(G.parameters(), opt_lr, opt_betas)
OptD = optim.AdamW(D.parameters(), opt_lr, opt_betas)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

writer = SummaryWriter(log_dir="./logs")

print("Start training")
print("run `tensorboard --logdir logs` if you need show tensorboard")

step_count = 0
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

            # convert to spectrogram
            spec = spectrogram(wf, n_fft, frame_size) # [N, fft_bin, Length]

            # decide crop range
            crop_range = decide_crop_range(spec.shape[2], crop_frames)
            dsp_out, fake, lossG, loss_dict = G(spec, spec_len, phoneme, phoneme_len, lm_feat, lm_feat_len, f0, spk_id, lang, crop_range)

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
            save_models(G, D)

print("Training Complete!")
save_models(G, D)
cleanup()
