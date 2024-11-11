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
from module.vits import Vits, VitsGenerator, VitsPitchEnergy
from module.utils.dataset import VitsDataModule
from module.utils.safetensors import save_tensors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-c', '--config', default='config/base.json')
    parser.add_argument('-pt', '--pretraining', choices=['pitchenergy', 'gen', 'none'], default='none', help="Which or not to do pretraining")
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
                generator_path = self.models_dir / "generator.safetensors"
                save_tensors(pl_module.generator.state_dict(), generator_path)

    config = load_json_file(args.config)
    dm = VitsDataModule(**config.train.data_module)
    model_path = Path(config.train.save.models_dir) / "vits.ckpt"
    
    model_loaded = False
    model_class = Vits
    if args.pretraining == 'gen':
        model_class = VitsGenerator
    elif args.pretraining == 'pitchenergy':
        model_class = VitsPitchEnergy
        
    if model_path.exists():
        print(f"{model_class.__name__}: loading checkpoint from {model_path}")
        model = model_class.load_from_checkpoint(model_path, strict=False)
        model_loaded = True
    else:
        print(f"{model_class.__name__}: initialize generator")
        model = model_class(config.vits)
        model = model.train()
            
    # load pretrained weights if provided
    if config.get("pretrained", None) is not None and model_loaded == False:
        pretrained_generator_path = config.pretrained.get("generator")
        if Path(pretrained_generator_path).exists():
            print(f"Loading pretrained weights from {pretrained_generator_path}")
            if pretrained_generator_path.endswith(".safetensors"):
                from module.utils.safetensors import load_tensors
                model.load_state_dict(load_tensors(pretrained_generator_path), strict=False)
            else:
                model.load_state_dict(torch.load(pretrained_generator_path), strict=False)

    print("if you need to check tensorboard, run `tensorboard -logdir lightning_logs`")
    cb_save_checkpoint = SaveCheckpoint(config.train.save.models_dir, config.train.save.interval)
    trainer = L.Trainer(**config.train.trainer, callbacks=[cb_save_checkpoint])
    trainer.fit(model, dm)
