import json
from pathlib import Path


# create metadata
def scan_cache(config):
    cache_dir = Path(config.preprocess.cache)
    models_dir = Path("models")
    metadata_path = models_dir / "metadata.json"
    if not models_dir.exists():
        models_dir.mkdir()

    speaker_names = []
    for subdir in cache_dir.glob("*"):
        if subdir.is_dir():
            speaker_names.append(subdir.name)
    speaker_names = sorted(speaker_names)
    metadata = {
            "speakers": speaker_names # speaker list
            }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
