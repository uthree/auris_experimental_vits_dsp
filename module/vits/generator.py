import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .posterior_encoder import PosteriorEncoder
from .prior_encoder import PriorEncoder
from .speaker_embedding import SpeakerEmbedding
from .flow import Flow
from .crop import crop_features
from module.utils.energy_estimation import estimate_energy


class Generator(nn.Module):
    # initialize from config
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(**config.decoder)
        self.posterior_encoder = PosteriorEncoder(**config.posterior_encoder)
        self.prior_encoder = PriorEncoder(config.prior_encoder)
        self.speaker_embedding = SpeakerEmbedding(**config.speaker_embedding)

    # training pass
    #
    # spec: [BatchSize, fft_bin, Length]
    # spec_len: [BatchSize]
    # phone: [BatchSize, NumPhonemes]
    # phone_len: [BatchSize]
    # lm_feat: [BatchSize, lm_dim, NumLMfeatures]
    # lm_feat_len: [BatchSize]
    # f0: [Batchsize, 1, Length]
    # spk: [BatchSize]
    # lang: [BatchSize]
    # crop_range: Tuple[int, int]
    #
    # Outputs:
    #   dsp_out: [BatchSize, Length * frame_size]
    #   fake: [BatchSize, Length * frame_size]
    #   lossG: [1]
    #   loss_dict: Dict[str: float]
    #
    def forward(
            self,
            spec,
            spec_len,
            phoneme,
            phoneme_len,
            lm_feat,
            lm_feat_len,
            f0,
            spk,
            lang,
            crop_range
            ):

        spk = self.speaker_embedding(spk)
        z, m_q, logs_q, spec_mask = self.posterior_encoder.forward(spec, spec_len, spk)
        energy = estimate_energy(spec)
        loss_prior, loss_dict_prior, (text_encoded, text_mask, fake_log_duration, real_log_duration) = self.prior_encoder.forward(spec_mask, z, logs_q, phoneme, phoneme_len, lm_feat, lm_feat_len, lang, f0, energy, spk)
        
        z_crop = crop_features(z, crop_range)
        f0_crop = crop_features(f0, crop_range)
        energy_crop = crop_features(energy, crop_range)
        dsp_out, fake, loss_decoder, loss_dict_decoder = self.decoder.forward(z_crop, f0_crop, energy_crop, spk)

        loss_dict = (loss_dict_decoder | loss_dict_prior) # merge dict
        loss = loss_prior + loss_decoder

        return loss, loss_dict, (text_encoded, text_mask, fake_log_duration, real_log_duration, spk.detach(), dsp_out, fake)

    @torch.no_grad()
    def text_to_speech(
            self,
            phoneme,
            phoneme_len,
            lm_feat,
            lm_feat_len,
            lang,
            spk,
            noise_scale=0.6,
            max_frames=2000,
            use_sdp=True,
            duration_scale=1.0,
            pitch_shift=0.0,
            energy_scale=1.0,
            ):
        spk = self.speaker_embedding(spk)
        z = self.prior_encoder.text_to_speech(
                phoneme, phoneme_len, lm_feat, lm_feat_len, lang, spk,
                noise_scale=noise_scale,
                max_frames=max_frames,
                use_sdp=use_sdp,
                duration_scale=duration_scale)
        f0, energy = self.decoder.estimate_pitch_energy(z, spk)
        pitch = torch.log2((f0 + 1e-6) / 440.0) * 12.0
        pitch += pitch_shift
        f0 = 440.0 * 2 ** (pitch / 12.0)
        energy = energy * energy_scale
        fake = self.decoder.infer(z, spk, f0=f0, energy=energy)
        return fake

    @torch.no_grad()
    def audio_reconstruction(self, spec, spec_len, spk):
        # embed speaker
        spk = self.speaker_embedding(spk)

        # encode linear spectrogram and speaker infomation
        z, m_q, logs_q, spec_mask = self.posterior_encoder(spec, spec_len, spk)

        # decode
        fake = self.decoder.infer(z, spk)
        return fake

    @torch.no_grad()
    def singing_voice_conversion(self):
        pass # TODO: write this

    @torch.no_grad()
    def singing_voice_synthesis(self):
        pass # TODO: write this
