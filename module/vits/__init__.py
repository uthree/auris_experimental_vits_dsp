import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator
from .discriminator import Discriminator
from .duration_discriminator import DurationDiscriminator
from .loss import mel_spectrogram_loss, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss, duration_discriminator_adversarial_loss, duration_generator_adversarial_loss
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
        self.duration_discriminator = DurationDiscriminator(**config.duration_discriminator)
        self.config = config

        # disable automatic optimization
        self.automatic_optimization = False
        # save hyperparameters
        self.save_hyperparameters()

    def training_step(self, batch):
        wf, spec_len, spk, f0, phoneme, phoneme_len, lm_feat, lm_feat_len, lang = batch
        wf = wf.squeeze(1) # [Batch, WaveLength]
        # get optimizer
        opt_g, opt_d, opt_dd = self.optimizers()

        # spectrogram
        n_fft = self.generator.posterior_encoder.n_fft
        frame_size = self.generator.posterior_encoder.frame_size
        spec = spectrogram(wf, n_fft, frame_size)

        # decide crop range
        crop_range = decide_crop_range(spec.shape[2], self.config.segment_size)

        # crop real waveform
        real = crop_waveform(wf, crop_range, frame_size)

        # start tracking gradient G.
        self.toggle_optimizer(opt_g)

        # calculate loss
        lossG, loss_dict, (text_encoded, text_mask, fake_log_duration, real_log_duration, spk_emb, dsp_out, fake) = self.generator(
                spec, spec_len, phoneme, phoneme_len, lm_feat, lm_feat_len, f0, spk, lang, crop_range)

        loss_dsp = mel_spectrogram_loss(dsp_out, real)
        loss_mel = mel_spectrogram_loss(fake, real)
        logits_fake, fmap_fake = self.discriminator(fake)
        _, fmap_real = self.discriminator(real)
        loss_feat = feature_matching_loss(fmap_real, fmap_fake)
        loss_adv = generator_adversarial_loss(logits_fake)
        dur_logit_fake = self.duration_discriminator(text_encoded, text_mask, fake_log_duration, spk_emb)
        loss_dadv = duration_generator_adversarial_loss(dur_logit_fake, text_mask)

        lossG += loss_mel * 10.0 + loss_dsp + loss_feat + loss_adv + loss_dadv
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

        # start tracking gradient Duration Discriminator
        self.toggle_optimizer(opt_dd)

        fake_log_duration = fake_log_duration.detach()
        real_log_duration = real_log_duration.detach()
        text_mask = text_mask.detach()
        text_encoded = text_encoded.detach()
        spk_emb = spk_emb.detach()
        dur_logit_real = self.duration_discriminator(text_encoded, text_mask, real_log_duration, spk_emb)
        dur_logit_fake = self.duration_discriminator(text_encoded, text_mask, fake_log_duration, spk_emb)

        lossDD = duration_discriminator_adversarial_loss(dur_logit_real, dur_logit_fake, text_mask)
        self.manual_backward(lossDD)
        opt_dd.step()
        opt_dd.zero_grad()

        # stop tracking gradient Duration Discriminator
        self.untoggle_optimizer(opt_dd)

        # write log
        loss_dict['Mel'] = loss_mel.item()
        loss_dict['Generator Adversarial'] = loss_adv.item()
        loss_dict['DSP'] = loss_dsp.item()
        loss_dict['Feature Matching'] = loss_feat.item()
        loss_dict['Discriminator Adversarial'] = lossD.item()
        loss_dict['Duration Generator Adversarial'] = loss_dadv.item()
        loss_dict['Duration Discriminator Adversarial'] = lossDD.item()

        for k, v in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"loss/{k}", v)

    def configure_optimizers(self):
        lr = self.config.optimizer.lr
        betas = self.config.optimizer.betas

        opt_g = optim.AdamW(self.generator.parameters(), lr, betas)
        opt_d = optim.AdamW(self.discriminator.parameters(), lr, betas)
        opt_dd = optim.AdamW(self.duration_discriminator.parameters(), lr, betas)
        return opt_g, opt_d, opt_dd
