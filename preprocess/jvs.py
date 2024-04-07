import argparse
import os
import torch
import pathlib
import json

from module.utils.f0_estimation import estimate_f0


parser = argparse.ArgumentParser(description="preprocess script for JVS corpus")
parser.add_argument('jvs_corpus_root', default='jvs_ver1')
parser.add_argument('-c', '--config', default='./config/base.json')


