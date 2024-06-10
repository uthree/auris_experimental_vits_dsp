import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

from .generator import Generator
from .discriminator import Discriminator
from .crop import crop_features, crop_waveform, decide_crop_range
from module.utils.loss import multiscale_stft_loss, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss, kl_divergence_loss
from .monotonic_align import maximum_path


def search_path(z_p, m_p, logs_p, x_mask, y_mask, mas_noise_scale=0.01):
    with torch.no_grad():
        o_scale = torch.exp(-2 * logs_p)  # [b, d, t]
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t]
        logp2 = torch.matmul(-0.5 * (z_p**2).mT, o_scale)  # [b, t', d] x [b, d, t] = [b, t', t]
        logp3 = torch.matmul(z_p.mT, (m_p * o_scale))  # [b, t', d] x [b, d, t] = [b, t', t]
        logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1], keepdim=True)  # [b, 1, t]
        logp = logp1 + logp2 + logp3 + logp4  # [b, t', t]

        if mas_noise_scale > 0.0:
            epsilon = torch.std(logp) * torch.randn_like(logp) * mas_noise_scale
            logp = logp + epsilon

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # [b, 1, t] * [b, t', 1] = [b, t', t]
        attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t', t] maximum_path_cuda
    return attn


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
        wf, spec, spec_len, speaker_id, f0, phoneme, phoneme_len, lm_feat, language = batch

        # get optimizer
        opt_g, opt_d = self.optimizers()

        # aliases
        G, D = self.generator, self.discriminator

        # decide crop range
        crop_range = decide_crop_range(spec.shape[2], self.config.segment_size)

        # crop real waveform
        real = crop_waveform(wf, crop_range, self.config.generator.decoder.frame_size)

        # start tracking gradient G.
        self.toggle_optimizer(opt_g)

        g = G.speaker_embedding(speaker_id)
        z, m_q, logs_q, spec_mask = G.posterior_encoder(spec, spec_len, g)
        z_p = G.flow(z, spec_mask, g)
        text_encoded, m_p, logs_p, text_mask = G.text_encoder(phoneme, phoneme_len, lm_feat, g, language)
        MAS_path = search_path(z_p, m_p, logs_p, text_mask, spec_mask)
        m_p = torch.matmul(MAS_path.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        duration = MAS_path.sum(2).to(torch.float)
        loss_sdp = G.stochastic_duration_predictor(text_encoded, text_mask, duration, g).mean()
        pred_log_dur = G.duration_predictor.forward(text_encoded, text_mask, g)
        log_dur = torch.log(duration + 1e-6)
        loss_dp = ((log_dur - pred_log_dur) ** 2).mean()
        z_crop = crop_features(z, crop_range)
        f0_crop = crop_features(f0, crop_range)
        fake = G.decoder(z_crop, g, f0_crop).squeeze(1)
        logits_fake, fmap_fake = D(fake)
        logits_real, fmap_real = D(real)
        loss_stft = multiscale_stft_loss(fake, real)
        loss_feat = feature_matching_loss(fmap_real, fmap_fake)
        loss_adv = generator_adversarial_loss(logits_fake)
        loss_kl = kl_divergence_loss(z_p, logs_q, m_p, logs_p, spec_mask)

        # calculate loss
        loss_G = loss_stft + loss_adv + loss_feat + loss_dp + loss_sdp + loss_kl
        self.manual_backward(loss_G)
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

        loss_D = discriminator_adversarial_loss(logits_real, logits_fake)
        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()

        # stop tracking gradient D.
        self.untoggle_optimizer(opt_d)

        # write log
        loss_dict = dict()
        loss_dict['STFT'] = loss_stft.item()
        loss_dict['Generator Adversarial'] = loss_adv.item()
        loss_dict['Feature Matching'] = loss_feat.item()
        loss_dict['Discriminator Adversarial'] = loss_D.item()
        loss_dict['KL Divergence'] = loss_kl.item()

        for k, v in zip(loss_dict.keys(), loss_dict.values()):
            self.log(f"loss/{k}", v)

    def configure_optimizers(self):
        lr = self.config.optimizer.lr
        betas = self.config.optimizer.betas

        opt_g = optim.AdamW(self.generator.parameters(), lr, betas)
        opt_d = optim.AdamW(self.discriminator.parameters(), lr, betas)
        return opt_g, opt_d
