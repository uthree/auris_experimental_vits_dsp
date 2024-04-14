import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import RelativePositionTransformerDecoder


class TextEncoder(nn.Module):
    def __init__(
            self,
            num_phonemes=512,
            num_languages=256,
            internal_channels=256,
            content_channels=192,
            n_heads=4,
            lm_dim=768,
            kernel_size=1,
            dropout=0.0,
            window_size=4,
            num_layers=4,
            ):
        super().__init__()
        self.lm_proj = nn.Linear(lm_dim, internal_channels)
        self.phoneme_embedding = nn.Embedding(num_phonemes, internal_channels)
        self.language_embedding = nn.Embedding(num_languages, internal_channels)
        self.transformer = RelativePositionTransformerDecoder(
                internal_channels,
                internal_channels * 4,
                n_heads,
                num_layers,
                kernel_size,
                dropout,
                window_size
                )
        self.post = nn.Conv1d(internal_channels, content_channels * 2, 1)

    # Note:
    #   x: Phoneme IDs
    #   x_length: length of phoneme sequence, for generating mask
    #   y: language model's output features
    #   y_length: length of language model's output features, for generating mask
    #   lang: Language ID
    #
    # x: [BatchSize, Length_x]
    # x_length: [BatchSize]
    # y: [BatchSize, Length_y, lm_dim]
    # y_length [BatchSize]
    # lang: [BatchSize]
    # 
    # Outputs:
    #   z: [BatchSize, content_channels, Length_x]
    #   mean: [BatchSize, content_channels, Length_x]
    #   logvar: [BatchSize, content_channels, Length_x]
    #   z_mask: [BatchSize, 1, Length_x]
    def forward(self, x, x_length, y, y_length, lang):
        # generate mask
        # x mask
        max_length = x.shape[1]
        progression = torch.arange(max_length, dtype=x_length.dtype, device=x_length.device)
        x_mask = (progression.unsqueeze(0) < x_length.unsqueeze(1))
        x_mask = x_mask.unsqueeze(1).to(y.dtype)
        z_mask = x_mask

        # y mask
        max_length = y.shape[1]
        progression = torch.arange(max_length, dtype=y_length.dtype, device=y_length.device)
        y_mask = (progression.unsqueeze(0) < y_length.unsqueeze(1))
        y_mask = y_mask.unsqueeze(1).to(y.dtype)

        # pass network
        y = self.lm_proj(y).mT # [B, C, L_y] where C = internal_channels, L_y = Length_y, L_x = Length_x, B = BatchSize
        x = self.phoneme_embedding(x) # [B, L_x, C]
        lang = self.language_embedding(lang) # [B, C]
        lang = lang.unsqueeze(1) # [B, 1, C]
        x = x + lang # language conditioning
        x = x.mT # [B, C, L_x]
        x = self.transformer(x, x_mask, y, y_mask) # [B, C, L_x]
        x = self.post(x) * x_mask # [B, 2C, L_x]
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(logvar) * z_mask
        return z, mean, logvar, z_mask
