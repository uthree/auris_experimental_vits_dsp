import json
from pathlib import Path


def scan_cache(config):
    cache_dir = Path(config['preprocess']['cache'])
    speaker_list_path = Path("speaker_list.json")

    names = []
    for subdir in cache_dir.glob("*"):
        if subdir.is_dir():
            names.append(subdir.name)
    names = sorted(names)

    with open(speaker_list_path, 'w') as f:
        json.dump(names, f)
