import os
import argparse
import json
from pathlib import Path
import shutil

from module.preprocess.jvs import preprocess_jvs
from module.preprocess.wave_and_text import preprocess_wave_and_text
from module.preprocess.scan import scan_cache
from module.utils.config import load_json_file


def get_preprocess_method(dataset_type):
    if dataset_type == 'jvs':
        return preprocess_jvs
    if dataset_type == 'wav-txt':
        return preprocess_wave_and_text
    else:
        raise "Unknown dataset type"


parser = argparse.ArgumentParser("preprocess")
parser.add_argument('type')
parser.add_argument('root_dir')
parser.add_argument('-c', '--config', default='./config/base.json')
parser.add_argument('--scan-only', default=False, type=bool)

args = parser.parse_args()

config = load_json_file(args.config)
root_dir = Path(args.root_dir)
dataset_type = args.type

cache_dir = Path(config.preprocess.cache)
if not cache_dir.exists():
    cache_dir.mkdir()

preprocess_method = get_preprocess_method(dataset_type)

if not args.scan_only:
    print(f"Start preprocess type={dataset_type}, root={str(root_dir)}")
    preprocess_method(root_dir, config)

print(f"Scaning dataset cache")
scan_cache(config)
shutil.copy(args.config, f'{config.train.save.models_dir}/config.json')

print(f"Complete!")
