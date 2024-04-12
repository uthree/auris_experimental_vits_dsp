import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator
from .discriminator import Discriminator
from .loss import mel_spectrogram_loss, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss
from .crop import crop_features, crop_waveform, decide_crop_range
from .spectrogram import spectrogram


class Vits(L.LightningModule):
    def __init__(
            self,
            config,
            ):
        super().__init__()
        self.generator = Generator(config.generator)
        self.discriminator = Discriminator(config.discriminator)
        self.config = config

        # disable automatic optimization
        self.automatic_optimization = False
        # save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch):
        wf, spec_len, spk, f0, phoneme, phoneme_len, lm_feat, lm_feat_len, lang = batch
        wf = wf.squeeze(1) # [Batch, WaveLength]
        # get optimizer
        opt_g, opt_d = self.optimizers()

        # spectrogram
        n_fft = self.generator.posterior_encoder.n_fft
        frame_size = self.generator.posterior_encoder.frame_size
        spec = spectrogram(wf, n_fft, frame_size)

        # decide crop range
        crop_range = decide_crop_range(spec.shape[2], 32)

        # crop real waveform
        real = crop_waveform(wf, crop_range, frame_size)

        # start tracking gradient G.
        self.toggle_optimizer(opt_g)

        # calculate loss
        dsp_out, fake, lossG, loss_dict = self.generator.forward(
                spec, spec_len, phoneme, phoneme_len, lm_feat, lm_feat_len, f0, spk, lang, crop_range)

        loss_dsp = mel_spectrogram_loss(dsp_out, real)
        loss_mel = mel_spectrogram_loss(fake, real)
        logits_fake, fmap_fake = self.discriminator(fake)
        _, fmap_real = self.discriminator(real)
        loss_feat = feature_matching_loss(fmap_real, fmap_fake)
        loss_adv = generator_adversarial_loss(logits_fake)

        lossG += loss_mel * 45.0 + loss_dsp + loss_feat + loss_adv
        self.manual_backward(lossG)
        opt_g.step()
        opt_g.zero_grad()

        # stop tracking gradient G.
        self.untoggle_optimizer(opt_g)

        # start tracking gradient D.
        self.toggle_optimizer(opt_d)

        # calculate loss
        fake = fake.detach()
        logits_fake, _ = self.discriminator(fake)
        logits_real, _ = self.discriminator(real)

        lossD = discriminator_adversarial_loss(logits_real, logits_fake)
        self.manual_backward(lossD)
        opt_d.step()
        opt_d.zero_grad()

        # stop tracking gradient D.
        self.untoggle_optimizer(opt_d)

        # write log
        loss_dict['Mel'] = loss_mel.item()
        loss_dict['Generator Adversarial'] = loss_adv.item()
        loss_dict['DSP'] = loss_dsp.item()
        loss_dict['Feature Matching'] = loss_feat.item()
        loss_dict['Discriminator Adversarial'] = lossD.item()

        for k, v in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"loss/{k}", v)

    def configure_optimizers(self):
        lr = self.config.optimizer.lr
        betas = self.config.optimizer.betas

        opt_g = optim.AdamW(self.generator.parameters(), lr, betas)
        opt_d = optim.AdamW(self.discriminator.parameters(), lr, betas)
        return opt_g, opt_d
