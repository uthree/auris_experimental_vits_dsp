from pathlib import Path
from .processor import Preprocessor
from tqdm import tqdm


def process_category(path: Path, category, processor: Preprocessor, speaker_name, config):
    print(f"Ppocessing {str(path)}")
    audio_dir = path / "wav24kHz16bit"
    transcription_path = path / "transcripts_utf8.txt"
    with open(transcription_path, encoding='utf-8') as f:
        transcription_text = f.read()

    counter = 0
    for metadata in tqdm(transcription_text.split("\n")):
        s = metadata.split(":")
        if len(s) >= 2:
            audio_file_name, transcription = s[0], s[1]
            audio_file_path = audio_dir / (audio_file_name + ".wav")
            if not audio_file_path.exists():
                continue
            processor.write_cache(
                    audio_file_path,
                    transcription,
                    'ja',
                    speaker_name,
                    f"{category}_{counter}"
                    )
            counter += 1

def preprocess_jvs(jvs_root: Path, config):
    processor = Preprocessor(config)
    cache_dir = Path(config['preprocess']['cache'])
    for subdir in jvs_root.glob("*/"):
        if subdir.is_dir():
            print(f"Processing {subdir}")
            speaker_name = subdir.name
            process_category(subdir / "nonpara30", "nonpara30", processor, speaker_name, config)
            process_category(subdir / "parallel100", "paralell100", processor, speaker_name, config)
