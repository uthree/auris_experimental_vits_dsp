from typing import List, Union

import torch
import math
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from module.vits import Generator, spectrogram
from module.utils.safetensors import load_tensors
from module.g2p import G2PProcessor
from module.language_model import LanguageModel
from module.utils.config import load_json_file


class Infer:
    def __init__(self, safetensors_path, config_path, metadata_path, device=torch.device('cpu')):
        self.device = device
        self.config = load_json_file(config_path)
        self.metadata = load_json_file(metadata_path)
        self.g2p = G2PProcessor()
        self.lm = LanguageModel(self.config.language_model.type, self.config.language_model.options)

        # load generator
        generator = Generator(self.config.vits.generator)
        generator.load_state_dict(load_tensors(safetensors_path))
        generator = generator.to(self.device)
        self.generator = generator

        self.max_lm_tokens = self.config.infer.max_lm_tokens
        self.max_phonemes = self.config.infer.max_phonemes
        self.max_frames = self.config.infer.max_frames

        self.n_fft = self.config.infer.n_fft
        self.frame_size = self.config.infer.frame_size
        self.sample_rate = self.config.infer.sample_rate

    def speakers(self):
        return self.metadata.speakers

    def speaker_id(self, speaker):
        return self.speakers().index(speaker)

    def languages(self):
        return self.g2p.languages

    def language_id(self, language):
        return self.g2p.language_to_id(language)

    @torch.inference_mode()
    def text_to_speech(
            self,
            text: str,
            speaker: str,
            language: str,
            style_text: Union[None, str] = None,
            duration_scale=1.0,
            pitch_shift=0.0,
            ):
        spk = torch.LongTensor([self.speaker_id(speaker)])
        if style_text is None:
            style_text = text
        lm_feat, lm_feat_len = self.lm.encode([style_text], self.max_lm_tokens)
        phoneme, phoneme_len, lang = self.g2p.encode([text], [language], self.max_phonemes)

        device = self.device
        phoneme = phoneme.to(device)
        phoneme_len = phoneme_len.to(device)
        lm_feat = lm_feat.to(device)
        lm_feat_len = lm_feat_len.to(device)
        spk = spk.to(device)
        lang = lang.to(device)

        wf = self.generator.text_to_speech(
                phoneme,
                phoneme_len,
                lm_feat,
                lm_feat_len,
                lang,
                spk,
                duration_scale=duration_scale,
                pitch_shift=pitch_shift,
                )
        return wf.squeeze(0)

    # wf: [Channels, Length]
    @torch.inference_mode()
    def audio_reconstruction(self, wf: torch.Tensor, speaker:str):
        spk = torch.LongTensor([self.speaker_id(speaker)])
        wf = wf.sum(dim=0, keepdim=True)
        spec = spectrogram(wf, self.n_fft, self.frame_size)
        spec_len = torch.LongTensor([spec.shape[2]])

        device = self.device
        spec = spec.to(device)
        spec_len = spec_len.to(device)
        spk = spk.to(device)

        wf = self.generator.audio_reconstruction(spec, spec_len, spk)
        return wf.squeeze(0)

    def singing_voice_synthesis(self, score):
        parts = score['parts']
        for part_name, part in zip(parts.keys(), parts.values()):
            print(f"processing {part_name}")
            self._svs_generate_part(part)

    # TODO: コメントを英語にする、いつかやる。多分。
    def _svs_generate_part(self, part):
        language = part['language']
        style_text = part['style_text']
        speaker = part['speaker']
        notes = part['notes']

        # notes をonset でソート
        notes.sort(key=lambda x: x['onset'])

        # get begin and end time
        # 開始時刻[秒]と終了時刻[秒]をノート一覧から探す。もっとも小さいonsetが開始時刻で最も大きいoffsetが終了時刻。
        t_begin = None
        t_end = None
        # それと歌詞情報を取得する
        part_phonemes = [] # このパートの音素列
        note_phoneme_indices = [] # 各ノート毎の(開始index, 終了index)
        for note in notes:
            b = note['onset']
            e = note['offset']
            if t_begin is None:
                t_begin = b
            elif b < t_begin:
                t_begin = b
            if t_end is None:
                t_end = e
            elif b > t_end:
                t_end = e
            
            # 歌詞情報の処理
            note_phonemes = self.g2p.grapheme_to_phoneme(note['lyrics'], language)
            note_phoneme_indices.append((len(part_phonemes), len(part_phonemes) + len(note_phonemes) - 1))
            part_phonemes.extend(note_phonemes)
        # 音素の数
        num_phonemes = len(part_phonemes)
        # パートの長さを求める
        part_length = t_end - t_begin
        # 1秒間に何フレームか
        fps = self.sample_rate / self.frame_size
        # 生成するフレーム数
        num_frames = math.ceil(part_length * fps)
        # ピッチ列のバッファ。この段階ではまだMIDIのスケール。-infに近い値で埋めておく。(self._midi2f0(-inf) = 0なので、発声がない区間を0Hzにしたい。)
        pitch = torch.full([num_frames], -1e10)
        # エネルギー列のバッファ。 これは初期値0
        energy = torch.full([num_frames], 0.0)
        # Duration
        duration = torch.full([num_phonemes], 0.0)
        # 話者をエンコードする
        speaker_id = self.speaker_id(speaker)
        speaker_id = torch.LongTensor([speaker_id])
        spk = self.generator.speaker_embedding(speaker_id)
        # 言語をエンコードする
        lang_id = self.language_id(language)
        lang = torch.LongTensor([lang_id])

        # 音素とテキスト. LMの特徴量をエンコードする
        phonemes = torch.LongTensor(self.g2p.phoneme_to_id(part_phonemes)).unsqueeze(1)
        phonemes_len = torch.LongTensor([phonemes.shape[1]])
        lm_feat, lm_feat_len = self.lm.encode([style_text], self.max_lm_tokens)
        text_encoded, text_mean, text_logvar, text_mask = self.generator.prior_encoder.text_encoder(phonemes, phonemes_len, lm_feat, lm_feat_len, spk, lang)
        # durationを推定する
        log_dur = self.generator.prior_encoder.stochastic_duration_predictor(text_encoded, text_mask, g=spk, reverse=True)
        duration = torch.ceil(torch.exp(log_dur)).to(torch.long)

        # ノートごとに処理する
        for i, note in enumerate(notes):
            phoneme_begin, phoneme_end = note_phoneme_indices[i]

            # ノートの始点と終点をフレーム単位に変換
            onset = round((note['onset'] - t_begin) * fps)
            offset = round((note['offset'] - t_begin) * fps)

            # 代入
            pitch[onset:offset] = float(note['pitch'])
            energy[onset:offset] = float(note['energy'])

            # TODO: ビブラートとかフォールとかenergyの調整とか

        print(pitch)


    def _f02midi(self, f0):
        return torch.log2(f0 / 440.0) * 12.0 + 69.0

    def _midi2f0(self, n):
        return 440.0 * 2 ** ((n - 69.0) / 12.0)
