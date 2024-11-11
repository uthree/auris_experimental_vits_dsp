from pathlib import Path

import torch
import torchaudio
from torchaudio.functional import resample

from module.g2p import G2PProcessor
from module.language_model import LanguageModel
from module.utils.f0_estimation import estimate_f0


# for dataset preprocess
class Preprocessor:
    def __init__(self, config):
        self.g2p = G2PProcessor()
        self.lm = LanguageModel(config.language_model.type, config.language_model.options)
        self.max_phonemes = config.preprocess.max_phonemes
        self.lm_max_tokens = config.preprocess.lm_max_tokens
        self.pitch_estimation = config.preprocess.pitch_estimation
        self.max_waveform_length = config.preprocess.max_waveform_length
        self.sample_rate = config.preprocess.sample_rate
        self.frame_size = config.preprocess.frame_size
        self.config = config
        self.stt_pipeline = None
        
    def init_stt_pipeline(self, launguage: str = "ja"):
        if self.config.preprocess.get("stt", None) is None:
            return

        if self.config.preprocess.stt.type == "wav2vec2":
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            processor = Wav2Vec2Processor.from_pretrained(self.config.preprocess.stt.model)
            model = Wav2Vec2ForCTC.from_pretrained(self.config.preprocess.stt.model)
            self.stt_pipeline = (processor, model)
        elif self.config.preprocess.stt.type == "whisper":
            from transformers import pipeline
            model_id = self.config.preprocess.stt.model
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
            generate_kwargs = {"language": launguage, "task": "transcribe"}
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
                model_kwargs=model_kwargs
            )
            # processor = AutoProcessor.from_pretrained(model_id)
            self.stt_pipeline = (None, lambda x: pipe(x, generate_kwargs=generate_kwargs))
            
    def stt(self, waveform):
        if self.stt_pipeline is None:
            return None

        processor, model = self.stt_pipeline
        if processor is not None:
            inputs = processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        else:
            inputs = waveform
        transcription = model(inputs)["text"]
        return transcription

    def write_cache(self, waveform_path: Path, transcription: str, language: str, speaker_name: str, data_name: str):
        # load waveform file
        wf, sr = torchaudio.load(waveform_path)

        # resampling
        if sr != self.sample_rate:
            wf = resample(wf, sr, self.sample_rate) # [Channels, Length_wf]

        # mix down
        wf = wf.sum(dim=0) # [Length_wf]

        # get length frame size
        spec_len = torch.LongTensor([wf.shape[0] // self.frame_size])

        # padding
        if wf.shape[0] < self.max_waveform_length:
            wf = torch.cat([wf, torch.zeros(self.max_waveform_length - wf.shape[0])])

        # crop
        if wf.shape[0] > self.max_waveform_length:
            wf = wf[:self.max_waveform_length]

        wf = wf.unsqueeze(0) # [1, Length_wf]

        # estimate f0(pitch)
        f0 = estimate_f0(wf, self.sample_rate, self.frame_size, self.pitch_estimation)

        # get phonemes
        phonemes, phonemes_len, language = self.g2p.encode([transcription], [language], self.max_phonemes)

        # get lm features
        lm_feat, lm_feat_len = self.lm.encode([transcription], self.lm_max_tokens)

        # to dict
        metadata = {
                "spec_len": spec_len,
                "f0": f0,
                "phonemes": phonemes,
                "phonemes_len": phonemes_len,
                "language": language,
                "lm_feat": lm_feat,
                "lm_feat_len": lm_feat_len,
                }

        # get target dir.
        cache_dir = Path(self.config.preprocess.cache)
        subdir = cache_dir / speaker_name

        # check exists subdir
        if not subdir.exists():
            subdir.mkdir()

        audio_path = subdir / (data_name + ".wav")
        metadata_path = subdir / (data_name + ".pt")

        # save
        torchaudio.save(audio_path, wf, self.sample_rate)
        torch.save(metadata, metadata_path)
