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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-c', '--config', default='config/base.json')
    args = parser.parse_args()

    class SaveCheckpoint(L.Callback):
        def __init__(self, models_dir, interval=200):
            super().__init__()
            self.models_dir = Path(models_dir)
            self.interval = interval
            if not self.models_dir.exists():
                self.models_dir.mkdir()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx % self.interval == 0:
                ckpt_path = self.models_dir / "vits.ckpt"
                trainer.save_checkpoint(ckpt_path)

    config = load_json_file(args.config)
    dm = VitsDataModule(**config.train.data_module)
    model_path = Path(config.train.save.models_dir) / "vits.ckpt"

    if model_path.exists():
        print(f"loading checkpoint from {model_path}")
        model = Vits.load_from_checkpoint(model_path)
    else:
        print("initialize model")
        model = Vits(config.vits)

    print("if you need to check tensorboard, run `tensorboard -logdir lightning_logs`")
    cb_save_checkpoint = SaveCheckpoint(config.train.save.models_dir, config.train.save.interval)
    trainer = L.Trainer(**config.train.trainer, callbacks=[cb_save_checkpoint])
    trainer.fit(model, dm)
