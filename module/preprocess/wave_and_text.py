from pathlib import Path
from .processor import Preprocessor
from tqdm import tqdm


LANGUAGE = 'ja'

def process_speaker(subdir: Path, processor):
    speaker_name = subdir.stem
    counter = 0
    for wave_file in tqdm(subdir.rglob("*.wav")):
        text_file = wave_file.parent / (wave_file.stem + ".txt")
        if text_file.exists():
            with open(text_file) as f:
                text = f.read()
        else:
            text = processor.stt(str(wave_file))
            with open(text_file, 'w') as f:
                f.write(text)
        processor.write_cache(
            wave_file,
            text,
            LANGUAGE,
            speaker_name,
            f"{speaker_name}_{counter}",
        )
        counter += 1


def preprocess_wave_and_text(root: Path, config):
    processor = Preprocessor(config)
    processor.init_stt_pipeline(LANGUAGE)
    for subdir in root.glob("*/"):
        if subdir.is_dir():
            print(f"Processing {subdir.stem}")
            process_speaker(subdir, processor)