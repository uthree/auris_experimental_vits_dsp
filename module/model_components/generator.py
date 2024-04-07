import torch
import torch.nn as nn

from .decoder import Decoder
from .posterior_encoder import PosteriorEncoder
from .flow import Flow
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .speaker_embedding import SpeakerEmbedding


class Generator(nn.Module):
    # initialize from config
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config['decoder'])
        self.posterior_encoder = PosteriorEncoder(config['posterior_encoder'])
        self.flow = Flow(config['flow'])
        self.audio_encoder = AudioEncoder(config['audio_encoder'])
        self.text_encoder = TextEncoder(config['text_encoder'])
        self.speaker_embedding = SpeakerEmbedding(config['speaker_embedding'])
