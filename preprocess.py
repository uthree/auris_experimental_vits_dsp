import os
import argparse
import json
from pathlib import Path

from module.preprocess.jvs import preprocess_jvs

from module.preprocess.scan import scan_cache


def get_preprocess_method(dataset_type):
    if dataset_type == 'jvs':
        return preprocess_jvs
    else:
        raise "Unknown dataset type"


parser = argparse.ArgumentParser("preprocess")
parser.add_argument('type')
parser.add_argument('root_dir')
parser.add_argument('-c', '--config', default='./config/base.json')

args = parser.parse_args()

config = json.load(open(args.config))
root_dir = Path(args.root_dir)
dataset_type = args.type

cache_dir = Path(config['preprocess']['cache'])
if not cache_dir.exists():
    cache_dir.mkdir()

preprocess_method = get_preprocess_method(dataset_type)

print(f"Start preprocess type={dataset_type}, root={str(root_dir)}")
preprocess_method(root_dir, config)

print(f"Scaning dataset cache")
scan_cache(config)

print(f"Complete!")
