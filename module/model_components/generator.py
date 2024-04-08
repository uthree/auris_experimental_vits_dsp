import random
import math

import torch
import torch.nn as nn

from .decoder import Decoder
from .posterior_encoder import PosteriorEncoder
from .flow import Flow
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .speaker_embedding import SpeakerEmbedding
from .duration_predictors import DurationPredictor, StochasticDurationPredictor

from .monotonic_align import maximum_path
from module.train.slice import slice_z, decide_slice_area


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


class Generator(nn.Module):
    # initialize from config
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(**config['decoder'])
        self.posterior_encoder = PosteriorEncoder(**config['posterior_encoder'])
        self.flow = Flow(**config['flow'])
        self.audio_encoder = AudioEncoder(**config['audio_encoder'])
        self.text_encoder = TextEncoder(**config['text_encoder'])
        self.speaker_embedding = SpeakerEmbedding(**config['speaker_embedding'])
        self.duration_predictor = DurationPredictor(**config['duration_predictor'])
        self.stochastic_duration_predictor = StochasticDurationPredictor(**config['stochastic_duration_predictor'])
        self.slice_frames = config['slice_frames']
    
    # calculate loss
    # spec: [BatchSize, fft_bin, Length]
    # spec_len: [BatchSize]
    # phone: [BatchSize, NumPhonemes]
    # phone_len: [BatchSize]
    # lm_feat: [BatchSize, lm_dim, NumLMfeatures]
    # lm_feat_len: [BatchSize]
    # f0: [Batchsize, 1, Length]
    # spk: [BatchSize]
    # lang: [BatchSize]
    #
    # Outputs:
    #   loss_sdp: [1]
    #   loss_dp: [1]
    #   f0_logits: [BatchSize, num_f0_classes, Length]
    #   dsp_out: [BatchSize, Length * frame_size]
    #   fake: [BatchSize, Length * frame_size]
    #   slice_area: tuple[int, int]
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
            ):
        # encode text
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(phoneme, phoneme_len, lm_feat, lm_feat_len, lang)
        
        # embed speaker
        spk = self.speaker_embedding(spk)
        # encode linear spectrogram and speaker infomation
        z, m_q, logs_q, spec_mask = self.posterior_encoder(spec, spec_len, spk)
        # take z and embedded speaker, get z_p to use in MAS
        z_p = self.flow(z, spec_mask, spk)

        # run Monotonic Alignment Search (MAS).
        # MAS associates phoneme sequences with sounds.
        with torch.no_grad():
            # calculate nodes using D.P.
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).mT, s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.mT, (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            # mask unnecessary nodes, run D.P.
            MAS_node_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(-1)
            MAS_path = maximum_path(neg_cent, MAS_node_mask.squeeze(1)).unsqueeze(1).detach()

        # calculate duration each phonemes
        duration = MAS_path.sum(2)
        loss_sdp = self.stochastic_duration_predictor(
                text_encoded.detach(), 
                text_mask, 
                w=duration,
                g=spk
                ).mean()
        logw_ = torch.log(duration + 1e-6) * text_mask
        logw = self.duration_predictor(text_encoded.detach(), text_mask)
        loss_dp = (torch.sum((logw - logw_) ** 2, dim=(1, 2)) / torch.sum(text_mask)).mean()

        m_p = torch.matmul(MAS_path.squeeze(1), m_p.mT).mT
        logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.mT).mT

        # slice randomly
        area = decide_slice_area(z.shape[2], self.slice_frames)

        # decoder losses
        z_sliced = slice_z(z, area)
        f0_sliced = slice_z(f0, area)

        f0_logits, dsp_out, fake = self.decoder(z_sliced, f0_sliced)

        return loss_sdp, loss_dp, f0_logits, dsp_out, fake, MAS_path, text_mask, spec_mask, area, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def text_to_speech(self):
        pass # TODO: write this

    @torch.no_grad()
    def singing_voice_conversion(self):
        pass # TODO: write this
