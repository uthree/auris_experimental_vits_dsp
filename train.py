import os
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L
import pytorch_lightning
from module.utils.config import load_json_file
from module.vits import Vits
from module.utils.dataset import VitsDataModule

parser = argparse.ArgumentParser(description="train")
parser.add_argument('-c', '--config', default='config/base.json')
parser.add_argument('-t', '--task', default='vits', choices=['vits', 'recon'])
parser.add_argument('-ckpt', '--checkpoint', default=None)
args = parser.parse_args()

config = load_json_file(args.config)
task = args.task
dm = VitsDataModule(**config.train.data_module)
model = Vits(config, task)
if args.checkpoint:
    print(f"loading checkpoint from {args.checkpoint}")
    model = Vits.load_from_checkpoint(args.checkpoint)
model.task = task # set task
print("if you need to check tensorboard, run `tensorboard -logdir lightning_logs`")
trainer = L.Trainer(**config.train.trainer)
trainer.fit(model, dm)
