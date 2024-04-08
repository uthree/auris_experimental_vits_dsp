import json
from pathlib import Path


def scan_cache(config):
    models_dir = Path("models")
    cache_dir = Path(config['preprocess']['cache'])
    if not models_dir.exists():
        models_dir.mkdir()
    speakers_path = models_dir / "speakers.json"

    names = []
    for subdir in cache_dir.glob("*"):
        if subdir.is_dir():
            names.append(subdir.name)
    names = sorted(names)

    with open(speakers_path, 'w') as f:
        json.dump(names, f)
