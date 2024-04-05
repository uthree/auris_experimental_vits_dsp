import torch
import torch.nn as nn
import random


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# Shifted Absolute Positional Embedding (https://www.jstage.jst.go.jp/article/jnlp/29/1/29_248/_pdf/-char/ja)
# To avoid learning absolute position information, a random position shift is added to the position embedding during learning.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.max_len = max_len
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1) # [BatchSize, max_len, d_model]
        self.register_buffer('pe', pe)

    # x: [BatchSize, d_model, Length]
    # Output: [BatchSize, d_model, Length]
    def forward(self, x):
        if self.training:
            x_len = x.shape[1]
            begin = random.randint(0, self.max_len - x_len)
            end = begin + x_len
        else:
            begin = 0
            end = x.shape[1]
        emb = self.pe[:, :, begin:end]
        return x + emb


# encode phonemes with language model's features, language embedding
class TextEncoder(nn.Module):
    def __init__(self,
                 num_phonemes=256,
                 num_languages=256,
                 lm_dim=768,
                 d_model=256,
                 content_channels=192,
                 nhead=4,
                 num_layers=4):
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_phonemes, d_model)
        self.language_embedding = nn.Embedding(num_languages, d_model * 2)
        self.lm_proj = nn.Linear(lm_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.post = nn.Linear(d_model, content_channels * 2)
    
    # x: [BatchSize, PhonemeLength]
    # x_length: [BatchSize]
    # y: [BatchSize, LMFeatureLength, lm_dim]
    # y_length: [BatchSize]
    # language_id: [BatchSize]
    #
    # Outputs:
    #   z: [BatchSize, content_channels, Length]
    #   mean: [BatchSize, content_channels, Length]
    #   logvar: [BatchSize, content_channels, Length]
    #   z_mask: [BatchSize, 1, PhonemeLength]
    def forward(self, x, x_length, y, y_length, language_id):
        # generate mask
        # x mask
        max_Length = x.shape[1]
        progression = torch.arange(max_length, dtype=x_length.dtype, device=x_length.device)
        x_mask = (progression.unsqueeze(0) < x_length.unsqueeze(1)) #[BatchSize, PhonemeLength]

        # y mask
        max_length = y.shape[1]
        progression = torch.arange(max_length, dtype=y_length.dtype, device=y_length.dtype)
        y_mask = (progression.unsqueeze(0) < y_length.unsqueeze(1)) # [BatchSize, LMFeatureLength]

        # output (z) mask
        z_mask = x_mask.to(x.dtype).unsqueeze(1) # [BatchSize, 1, PhonemeLength]

        # pass network
        x = self.phoneme_embedding(x)
        x = self.positional_encoding(x)
        lang_emb = self.language_embedding(language_id).unsqueeze(1)
        scale, shift = torch.chunk(lang_emb, 2, dim=2)
        x = x * scale + shift
        y = self.lm_proj(y)
        x = self.transformer(
                x,
                y,
                src_key_padding_mask=torch.logical_not(x_mask),
                memory_key_padding_mask=torch.logical_not(y_mask))
        x = self.post(x).transpose(1, 2) # [BatchSize, Length, content_channels * 2]
        mean, logvar = torch.chunk(x, 2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(logvar)
        return z, mean, logvar, z_mask
