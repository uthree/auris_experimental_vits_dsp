import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .duration_predictors import StochasticDurationPredictor, DurationPredictor
from .monotonic_align import maximum_path
from .text_encoder import TextEncoder
from .flow import Flow
from module.utils.energy_estimation import estimate_energy



# run Monotonic Alignment Search (MAS).
# MAS associates phoneme sequences with sounds.
# 
# z_p: [b, d, t']
# m_p: [b, d, t]
# logs_p: [b, d, t]
# text_mask: [b, 1, t]
# spec_mask: [b, 1, t']
# Output: [b, 1, t', t]
def search_path(z_p, m_p, logs_p, text_mask, spec_mask, mas_noise_scale=0.0):
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
        MAS_node_mask = text_mask.unsqueeze(2) * spec_mask.unsqueeze(-1) # [b, 1, 't, t]
        MAS_path = maximum_path(neg_cent, MAS_node_mask.squeeze(1)).unsqueeze(1).detach() # [b, 1, 't, t]
    return MAS_path


class PriorEncoder(nn.Module):
    def __init__(
            self,
            config,
        ):
        super().__init__()
        self.flow = Flow(**config.flow)
        self.text_encoder = TextEncoder(**config.text_encoder)
        self.duration_predictor = DurationPredictor(**config.duration_predictor)
        self.stochastic_duration_predictor = StochasticDurationPredictor(**config.stochastic_duration_predictor)

    def forward(self, spec_mask, z, logs_q, phoneme, phoneme_len, lm_feat, lm_feat_len, lang, spk):
        # encode text
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(phoneme, phoneme_len, lm_feat, lm_feat_len, spk, lang)

        # remove speaker infomation
        z_p = self.flow(z, spec_mask, spk)
        
        # search path
        MAS_path = search_path(z_p, m_p, logs_p, text_mask, spec_mask)

        # KL Divergence loss
        m_p = torch.matmul(MAS_path.squeeze(1), m_p.mT).mT
        logs_p = torch.matmul(MAS_path.squeeze(1), logs_p.mT).mT
        loss_kl = kl_divergence_loss(z_p, logs_q, m_p, logs_p, spec_mask)

        # calculate duration each phonemes
        duration = MAS_path.sum(2)
        loss_sdp = self.stochastic_duration_predictor(
            text_encoded,
            text_mask,
            w=duration,
            g=spk
        ).sum() / text_mask.sum()

        logw_y = torch.log(duration + 1e-6) * text_mask
        logw_x = self.duration_predictor(text_encoded, text_mask, spk)
        loss_dp = torch.sum(((logw_x - logw_y) ** 2) * text_mask) / torch.sum(text_mask)

        # predict duration
        fake_log_duration = self.duration_predictor(text_encoded, text_mask, spk)
        real_log_duration = logw_y

        loss_dict = {
            "StochasticDurationPredictor": loss_sdp.item(),
            "DurationPredictor": loss_dp.item(),
            "KL Divergence": loss_kl.item(),
        }

        loss = loss_sdp + loss_dp + loss_kl
        return loss, loss_dict, (text_encoded.detach(), text_mask, fake_log_duration, real_log_duration)
    
    def text_to_speech(self, phoneme, phoneme_len, lm_feat, lm_feat_len, lang, spk, noise_scale=0.6, max_frames=2000, use_sdp=True, duration_scale=1.0):
        # encode text
        text_encoded, m_p, logs_p, text_mask = self.text_encoder(phoneme, phoneme_len, lm_feat, lm_feat_len, spk, lang)

        # predict duration
        if use_sdp:
            log_duration = self.stochastic_duration_predictor(text_encoded, text_mask, g=spk, reverse=True)
        else:
            log_duration = self.duration_predictor(text_encoded, text_mask, spk)
        duration = torch.exp(log_duration)
        duration = duration * text_mask * duration_scale
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

        return z