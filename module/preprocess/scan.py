import json
from pathlib import Path
from module.g2p import G2PProcessor


# create metadata
def scan_cache(config):
    cache_dir = Path(config.preprocess.cache)
    models_dir = Path(config.train.save.models_dir)
    metadata_path = models_dir / "metadata.json"
    if not models_dir.exists():
        models_dir.mkdir()

    speaker_names = []
    for subdir in cache_dir.glob("*"):
        if subdir.is_dir():
            speaker_names.append(subdir.name)
    speaker_names = sorted(speaker_names)
    g2p = G2PProcessor()
    phonemes = g2p.phoneme_vocabs
    languages = g2p.languages
    num_harmonics = config.vits.generator.decoder.num_harmonics
    sample_rate = config.vits.generator.decoder.sample_rate
    frame_size = config.vits.generator.decoder.frame_size
    metadata = {
            "speakers": speaker_names, # speaker list
            "phonemes": phonemes,
            "languages": languages,
            "num_harmonics": num_harmonics,
            "sample_rate": sample_rate,
            "frame_size": frame_size
            }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)