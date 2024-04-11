import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .posterior_encoder import PosteriorEncoder
from .flow import Flow
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .speaker_embedding import SpeakerEmbedding
from .duration_predictors import DurationPredictor, StochasticDurationPredictor
from .crop import crop_features

from .monotonic_align import maximum_path

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    padding_shape = [[0, 0], [1, 0], [0, 0]]
    padding = [item for sublist in padding_shape[::-1] for item in sublist]

    path = path - F.pad(path, padding)[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path


def kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum((kl * z_mask).mean(dim=1))
    l = kl / torch.sum(z_mask)
    return l


def pitch_estimation_loss(logits, label):
    num_classes = logits.shape[1]
    device = logits.device
    weight = torch.ones(num_classes, device=device)
    weight[0] = 1e-2
    return F.cross_entropy(logits, label, weight)


# run Monotonic Alignment Search (MAS).
# MAS associates phoneme sequences with sounds.
def search_path(z_p, m_p, logs_p, text_mask, spec_mask, mas_noise_scale=0.1):
    with torch.no_grad():
        # calculate nodes
        # b = batch size, d = feature dim, t = text length, t' = spec length
        s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
        neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True) # [b, 1, t]
        neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).mT, s_p_sq_r) # [b, t', d] x [b, d, t] = [b, t', t]
        neg_cent3 = torch.matmul(z_p.mT, (m_p * s_p_sq_r)) # [b, t', s] x [b, d, t] = [b, t', t]
        neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True) # [b, 1, t]
        neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4 # [b, t', t]

        # add noise
        if mas_noise_scale > 0.0:
            eps = torch.std(neg_cent) * torch.randn_like(neg_cent) * mas_noise_scale
            neg_cent += eps

        # mask unnecessary nodes, run D.P.
        MAS_node_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(-1) # [b, 1, t] * [b, t', 1] = [b, t', t]
        MAS_path = maximum_path(neg_cent, MAS_node_mask.squeeze(1)).unsqueeze(1).detach() # [b, 1, 't, t]
    return MAS_path


class Generator(nn.Module):
    # initialize from config
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(**config.decoder)
        self.posterior_encoder = PosteriorEncoder(**config.posterior_encoder)
        self.flow = Flow(**config.flow)
        self.audio_encoder = AudioEncoder(**config.audio_encoder)
        self.text_encoder = TextEncoder(**config.text_encoder)
        self.speaker_embedding = SpeakerEmbedding(**config.speaker_embedding)
        self.duration_predictor = DurationPredictor(**config.duration_predictor)
        self.stochastic_duration_predictor = StochasticDurationPredictor(**config.stochastic_duration_predictor)


    # reconstruction training pass
    # spec: [BatchSize, fft_bin, Length]
    # spec_len: [BatchSize]
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
    def train_recon(
            self,
            spec,
            spec_len,
            f0,
            spk,
            crop_range
            ):
        # embed speaker
        spk = self.speaker_embedding(spk)

        # encode linear spectrogram and speaker infomation
        z, m_q, logs_q, spec_mask = self.posterior_encoder(spec, spec_len, spk)

        # KL Divergence loss
        loss_kl = recon_kl_divergence_loss(m_q, logs_q, spec_mask)

        # decoder losses
        z_sliced = crop_features(z, crop_range)
        f0_sliced = crop_features(f0, crop_range)
        f0_logit, dsp_out, fake = self.decoder(z_sliced, f0_sliced, spk)

        # pitch estimation loss
        f0_label = self.decoder.pitch_estimator.freq2id(f0_sliced).squeeze(1)
        loss_pe = pitch_estimation_loss(f0_logit, f0_label)

        loss_dict = {
                "PitchEstimator": loss_pe.item(),
                "KL Divergence": loss_kl.item(),
                }

        lossG = loss_pe + loss_kl
        return dsp_out, fake, lossG, loss_dict


    # vits training pass
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
    def train_vits(
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
        # encode text
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(phoneme, phoneme_len, lm_feat, lm_feat_len, lang)
        
        # embed speaker
        spk = self.speaker_embedding(spk)
        # encode linear spectrogram and speaker infomation
        z, m_q, logs_q, spec_mask = self.posterior_encoder(spec, spec_len, spk)
        # take z and embedded speaker, get z_p to use in MAS
        z_p = self.flow(z, spec_mask, spk)

        # search path
        MAS_path = search_path(z_p, m_p, logs_p, text_mask, spec_mask)

        # calculate KL divergence loss
        m_p = torch.matmul(MAS_path.squeeze(1), m_p.mT).mT
        logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.mT).mT
        loss_kl = kl_divergence_loss(z_p, logs_q, m_p, logs_p, spec_mask)

        # calculate audio encoder loss
        z_ae, m_ae, logs_ae, _ = self.audio_encoder(spec, spec_len)
        loss_ae = (m_ae - m_p.detach()).abs().mean() + (logs_ae - logs_p.detach()).abs().mean()

        # calculate duration each phonemes
        duration = MAS_path.sum(2)
        loss_sdp = self.stochastic_duration_predictor(
                text_encoded, 
                text_mask, 
                w=duration,
                g=spk
                ).sum() / text_mask.sum()
        logw_ = torch.log(duration + 1e-6) * text_mask
        logw = self.duration_predictor(text_encoded, text_mask, spk)
        loss_dp = torch.sum((logw - logw_) ** 2) / torch.sum(text_mask)

        # decoder losses
        z_sliced = crop_features(z, crop_range)
        f0_sliced = crop_features(f0, crop_range)
        f0_logit, dsp_out, fake = self.decoder(z_sliced, f0_sliced, spk)

        # pitch estimation loss
        f0_label = self.decoder.pitch_estimator.freq2id(f0_sliced).squeeze(1)
        loss_pe = pitch_estimation_loss(f0_logit, f0_label)

        loss_dict = {
                "StochasticDurationPredictor": loss_sdp.item(),
                "DurationPredictor": loss_dp.item(),
                "PitchEstimator": loss_pe.item(),
                "KL Divergence": loss_kl.item(),
                "Audio Encoder": loss_ae.item()
                }

        lossG = loss_sdp + loss_dp + loss_pe + loss_kl + loss_ae

        return dsp_out, fake, lossG, loss_dict


    @torch.no_grad()
    def text_to_speech(
            self,
            phoneme,
            phoneme_len,
            lm_feat,
            lm_feat_len,
            spk,
            lang,
            noise_scale=0.6,
            max_frames=2000,
            use_sdp=True
            ):
        # encode text
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(phoneme, phoneme_len, lm_feat, lm_feat_len, lang)
        spk = self.speaker_embedding(spk)

        # predict duration
        if use_sdp:
            log_duration = self.stochastic_duration_predictor(text_encoded, text_mask, g=spk, reverse=True)
            duration = torch.exp(log_duration)
        else:
            duration = self.duration_predictor(text_encoded, text_mask, spk)
        duration = torch.ceil(duration)
        spec_len = torch.clamp_min(torch.sum(duration, dim=(1, 2)), 1).long()
        spec_len = torch.clamp_max(spec_len, max_frames)
        spec_mask = sequence_mask(spec_len).unsqueeze(1).to(text_mask.dtype)
        MAS_node_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(-1)
        MAS_path = generate_path(duration, MAS_node_mask).float()

        # projection
        m_p = torch.matmul(MAS_path.squeeze(1), m_p.mT).mT
        logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.mT).mT

        # sample from gaussian
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # crop max frames
        if z_p.shape[2] > max_frames:
            z_p = z_p[:, :, :max_frames]

        # add speaker infomation
        z = self.flow(z_p, spec_mask, spk, reverse=True)

        # decode
        fake = self.decoder.infer(z, spk)
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
